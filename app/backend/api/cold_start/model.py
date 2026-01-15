import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from copy import deepcopy

class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_fields, num_heads=1, dropout_rate=0.):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_fields = num_fields
        self.attention_dim = embedding_dim

        self.W_q = nn.Linear(embedding_dim, embedding_dim * num_heads, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim * num_heads, bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim * num_heads, bias=False)

        self.W_out = nn.Linear(embedding_dim * num_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.scale = embedding_dim ** -0.5

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        # x: [batch, num_fields, dim]
        batch_size = x.size(0)

        query = self.W_q(x).view(batch_size, self.num_fields, self.num_heads, self.embedding_dim).transpose(1, 2)
        key = self.W_k(x).view(batch_size, self.num_fields, self.num_heads, self.embedding_dim).transpose(1, 2)
        value = self.W_v(x).view(batch_size, self.num_fields, self.num_heads, self.embedding_dim).transpose(1, 2)

        # Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        if self.dropout:
            attn = self.dropout(attn)

        out = torch.matmul(attn, value) # [batch, heads, fields, dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, self.num_fields, -1)
        out = self.W_out(out)

        return self.layer_norm(out + x) # Residual

class GraphGenerator(nn.Module):
    def __init__(self, input_dim, num_fields, device):
        super().__init__()
        self.num_fields = num_fields
        self.generators = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim + num_fields, num_fields),
            nn.ReLU(),
            nn.Linear(num_fields, num_fields),
            nn.ReLU(),
            nn.Linear(num_fields, num_fields)
        ) for _ in range(num_fields)])
        self.device = device

    def forward(self, feature_emb):
        # feature_emb: [batch, input_dim]
        graph_fields = []
        for i in range(self.num_fields):
            field_index = torch.tensor([i]).to(self.device)
            field_onehot = F.one_hot(field_index, num_classes=self.num_fields).repeat(feature_emb.shape[0], 1).float()
            # Input: Item embedding + target field onehot
            inp = torch.cat([feature_emb, field_onehot], dim=1)
            graph_field = self.generators[i](inp)
            graph_fields.append(graph_field)

        # Stack: [batch, num_fields, num_fields]
        graph = torch.cat([g.unsqueeze(1) for g in graph_fields], dim=1)

        graph = torch.mean(graph, dim=0, keepdim=True)
        return graph # [1, N, N] -> will be broadcasted or repeated

class GNN_Layer(nn.Module):
    def __init__(self, num_fields, embed_dim, gnn_layers=2, device='cpu'):
        super().__init__()
        self.num_fields = num_fields
        self.gnn_layers = gnn_layers
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        self.relu = nn.LeakyReLU()

        self.attention = MultiHeadSelfAttention(embed_dim, num_fields)

    def forward(self, graphs, x):
        h = x
        for _ in range(self.gnn_layers):
            # Message Passing: A * H * W
            # H_W = H @ W: [batch, N, dim]
            h_w = self.W(h)

            # A @ H_W
            # graphs: [batch, N, N]
            h_new = torch.matmul(graphs, h_w)
            h = self.relu(h_new) + h # Residual

        # Self Attention Aggregation
        h = self.attention(h)
        return h

class GNN(nn.Module):
    def __init__(self, description, embed_dim, gnn_layers=2, device='cpu', item_id_name='item_id'):
        super().__init__()
        self.device = device
        self.item_id_name = item_id_name
        self.description = description
        self.embed_dim = embed_dim
        self.num_fields = 0

        self.emb_layer = nn.ModuleDict()

        # Parse fields
        self.field_names = []
        for name, (size, type) in description.items():
            if name == 'label' or name == item_id_name:
                continue
            self.field_names.append(name)
            self.num_fields += 1
            if type == 'seq':
                self.emb_layer[name] = nn.Embedding(size, embed_dim, padding_idx=0)
            elif type == 'spr':
                self.emb_layer[name] = nn.Embedding(size, embed_dim)
            elif type == 'ctn':
                self.emb_layer[name] = nn.Linear(1, embed_dim)

        # ID Embedding (trained via MAML)
        self.emb_layer[item_id_name] = nn.Embedding(description[item_id_name][0], embed_dim)

        # Base Global Graph (Learnable)
        self.graph_dict = nn.Parameter(torch.randn(self.num_fields, self.num_fields))

        self.gnn = GNN_Layer(self.num_fields, embed_dim, gnn_layers, device)

        # Prediction
        self.fc_score = nn.Linear(embed_dim, 1, bias=False)
        self.fc_weight = nn.Sequential(nn.Linear(self.num_fields * embed_dim, self.num_fields, bias=False), nn.Sigmoid())

    def forward(self, x_dict, graphs=None):
        bsz = x_dict[self.item_id_name].shape[0]

        # 1. Embed Features
        embs = []
        for name in self.field_names:
            val = x_dict[name].to(self.device)
            if self.description[name][1] == 'seq':
                # Sum pooling for sequence
                e = self.emb_layer[name](val).sum(dim=1)
            elif self.description[name][1] == 'ctn':
                e = self.emb_layer[name](val.float().unsqueeze(-1))
            else:
                e = self.emb_layer[name](val)
            embs.append(e)

        # [batch, num_fields, dim]
        x_emb = torch.stack(embs, dim=1)

        # 2. Graph
        if graphs is None:
            # global graph
            graphs = self.graph_dict.unsqueeze(0).repeat(bsz, 1, 1)
            graphs = torch.sigmoid(graphs) # Normalized?

        # 3. GNN
        h = self.gnn(graphs, x_emb)

        # 4. Prediction (Attentional)
        # h: [batch, num_fields, dim]
        score = self.fc_score(h).squeeze(-1) # [batch, num_fields]

        # Attention weights
        flat_h = h.view(bsz, -1)
        weights = self.fc_weight(flat_h) # [batch, num_fields]

        logit = (score * weights).sum(dim=1, keepdim=True)
        return torch.sigmoid(logit)

class EmerG(nn.Module):
    def __init__(self, model, item_features_dim, device, inner_lr=0.01):
        super().__init__()
        self.model = model
        self.device = device
        self.inner_lr = inner_lr
        self.criterion = nn.BCELoss()

        # Graph Generator
        # Input: Item ID Emb + Item Features (Author, Year, Publisher) flattened
        # item_features_dim: Dimension of (ItemEmb + AuthorEmb + YearEmb + PublisherEmb)
        self.graph_gen = GraphGenerator(item_features_dim, model.num_fields, device)

        self.keep_weight = {}

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())

    def load_parameters(self):
        self.model.load_state_dict(self.keep_weight)

    def generate_graph(self, x_dict):
        # Extract Item Features for Generator
        # We need to concat [ItemID_Emb, Author_Emb, Year_Emb, Publisher_Emb]
        # In our dataset, item features are: author, year, publisher. (title_len skipped for now)

        # Ensure x_dict items are on device
        item_id = x_dict[self.model.item_id_name].to(self.device)
        item_emb = self.model.emb_layer[self.model.item_id_name](item_id)

        feats = []
        feats.append(item_emb)

        # Fixed list of item features to expect
        item_feat_names = ['author', 'year', 'publisher']
        for name in item_feat_names:
            if name in x_dict:
                val = x_dict[name].to(self.device)
                e = self.model.emb_layer[name](val)
                feats.append(e)

        # Concat
        feature_emb = torch.cat(feats, dim=1) # [batch, total_dim]

        return self.graph_gen(feature_emb)

    def forward(self, support_x, support_y, query_x):
        # 1. Generate Graph using Support Set (should be same item)
        # We expect support_x to contain interactions for the SAME item.
        graphs = self.generate_graph(support_x) # [1, N, N]
        graphs = graphs.repeat(support_x[self.model.item_id_name].shape[0], 1, 1)

        # 2. Fine-tune Item ID Embedding on Support Set
        pred = self.model(support_x, graphs)
        loss = self.criterion(pred, support_y.view(-1, 1))

        # fast adaptation: we skip it for now or implement manual grad
        # In this step we just verify forward pass with generated graph

        graphs_query = graphs[0:1].repeat(query_x[self.model.item_id_name].shape[0], 1, 1)
        pred_query = self.model(query_x, graphs_query)

        return pred_query, loss
