import unittest
import torch
from app.backend.api.cold_start.model import EmerG, GNN

class TestModel(unittest.TestCase):
    def test_model_initialization(self):
        description = {
            'user_id': (100, 'spr'),
            'item_id': (50, 'spr'),
            'location': (20, 'seq'),
            'age': (7, 'spr'),
            'author': (30, 'spr'),
            'year': (10, 'spr'),
            'publisher': (15, 'spr')
        }
        embed_dim = 32

        # Base Model
        gnn = GNN(description, embed_dim, gnn_layers=2)

        # Meta Model
        item_feat_dim = embed_dim * 4 # Item Emb + 3 Item Features
        model = EmerG(gnn, item_feat_dim, device='cpu')

        self.assertIsInstance(model, EmerG)

    def test_forward_pass(self):
        description = {
            'user_id': (100, 'spr'),
            'item_id': (50, 'spr'),
            'location': (20, 'seq'),
            'age': (7, 'spr'),
            'author': (30, 'spr'),
            'year': (10, 'spr'),
            'publisher': (15, 'spr')
        }
        embed_dim = 32
        gnn = GNN(description, embed_dim, gnn_layers=2)
        # Assuming item features passed to graph gen are: item_id_emb + author_emb + year_emb + publisher_emb
        # Each is embed_dim. Total 4 * embed_dim.
        model = EmerG(gnn, embed_dim * 4, device='cpu')

        # Fake Batch
        bsz = 4
        x_dict = {
            'user_id': torch.randint(0, 100, (bsz,)),
            'item_id': torch.randint(0, 50, (bsz,)),
            'location': torch.randint(0, 20, (bsz, 5)),
            'age': torch.randint(0, 7, (bsz,)),
            'author': torch.randint(0, 30, (bsz,)),
            'year': torch.randint(0, 10, (bsz,)),
            'publisher': torch.randint(0, 15, (bsz,))
        }
        y = torch.randint(0, 2, (bsz,)).float()

        # Forward (MAML step: Support -> Query)
        # Using same batch as support and query for checking shape
        pred, loss = model(x_dict, y, x_dict)

        self.assertEqual(pred.shape, (bsz, 1))
        self.assertTrue(torch.is_tensor(loss))

if __name__ == '__main__':
    unittest.main()
