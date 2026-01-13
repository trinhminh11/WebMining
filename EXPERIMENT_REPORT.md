# Experiment Report: Books Recommendation Systems

## Group 21

|         **Name**    | **MSSV** |
|---------------------|----------|
| Trinh The Minh      | 20225513 |
| Dao Xuan Quang Minh | 20225449 |
| Nguyen Minh Duong   | 20225439 |
| Bui Anh Duong       | 20225489 |
| Vu Ngoc Ha          | 20225490 |

---

## 1. Source Code and Dataset Links

### Source Code: [WebMining Project](https://github.com/trinhminh11/WebMining)


### Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| **Book-Crossing** | Book ratings dataset with ~1.1M ratings from 278K users on 271K books | [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) |


**Data Split Configuration:**
- Train/Test Split: 75% / 25%
- Sample Fraction: 30% of data used for training efficiency
- Minimum Rating Threshold: 1

---

## 2. Evaluation Metrics

### Precision@K
Precision@K measures the proportion of recommended items in the top-K that are actually relevant to the user.

$$\text{Precision@K} = \frac{|\{\text{relevant items}\} \cap \{\text{top-K recommended}\}|}{K}$$

**Interpretation**: Higher values indicate better accuracy in top-K recommendations.

### Recall@K
Recall@K measures the proportion of relevant items that appear in the top-K recommendations.

$$\text{Recall@K} = \frac{|\{\text{relevant items}\} \cap \{\text{top-K recommended}\}|}{|\{\text{relevant items}\}|}$$

**Interpretation**: Higher values indicate better coverage of user interests.

### Additional Metrics (Cold-Start Evaluation)
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
- **F1-Score**: Harmonic mean of precision and recall

---

## 3. Evaluated Methods

### 3.1 Matrix Factorization (MF / SVD)
**Description**: A classic collaborative filtering technique that decomposes the user-item interaction matrix into latent factor matrices.

**Key Characteristics**:
- Uses Singular Value Decomposition (SVD)
- Learns latent representations for users and items
- Predicts ratings by reconstructing the matrix

**Loss Function**:
$$\mathcal{L} = \sum_{(u,i) \in \mathcal{O}} (r_{ui} - \mathbf{p}_u^T \mathbf{q}_i)^2 + \lambda(||\mathbf{p}_u||^2 + ||\mathbf{q}_i||^2)$$

Where:
- $r_{ui}$ is the actual rating
- $\mathbf{p}_u$, $\mathbf{q}_i$ are latent vectors for user $u$ and item $i$
- $\lambda$ is the regularization parameter

---

### 3.2 Bayesian Personalized Ranking (BPR)
**Description**: A pairwise learning approach for personalized ranking that optimizes the posterior probability of user preferences.

**Key Characteristics**:
- Implicit feedback model
- Learns to rank items based on pairwise comparisons
- Assumes positive items should rank higher than unobserved items

**Loss Function** (BPR-OPT):
$$\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) + \lambda ||\Theta||^2$$

Where:
- $\hat{x}_{uij} = \hat{x}_{ui} - \hat{x}_{uj}$ is the difference between positive and negative item scores
- $\sigma$ is the sigmoid function
- $D_S$ is the training set of triplets (user, positive item, negative item)

---

### 3.3 Bilateral Variational Autoencoder (BiVAE)
**Description**: A deep learning approach using variational autoencoders to learn both user and item latent representations simultaneously.

**Key Characteristics**:
- Generative model with probabilistic latent representations
- Learns bilateral (user-side and item-side) representations
- Uses variational inference for optimization

**Loss Function** (Evidence Lower Bound - ELBO):
$$\mathcal{L}_{BiVAE} = \mathbb{E}_{q}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) || p(z))$$

Where:
- The first term is the reconstruction loss
- The second term is the KL divergence regularization
- $\beta$ controls the trade-off between reconstruction and regularization

---

### 3.4 Light Graph Convolutional Network (LightGCN)
**Description**: A simplified graph convolutional network specifically designed for collaborative filtering, removing unnecessary feature transformations.

**Key Characteristics**:
- Uses neighborhood aggregation on user-item bipartite graph
- Removes non-linear activations for simplicity
- Layer combination for multi-scale representations

**Propagation Rule**:
$$\mathbf{e}_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|} \sqrt{|\mathcal{N}_i|}} \mathbf{e}_i^{(k)}$$

$$\mathbf{e}_i^{(k+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i|} \sqrt{|\mathcal{N}_u|}} \mathbf{e}_u^{(k)}$$

**Loss Function** (BPR Loss):
$$\mathcal{L} = \sum_{(u,i,j) \in \mathcal{O}} -\ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda ||\mathbf{E}^{(0)}||^2$$

---

### 3.5 EmerG (Cold-Start Model)
**Description**: A meta-learning approach for cold-start CTR prediction that learns item-specific feature interaction patterns using hypernetworks and Graph Neural Networks.

**Reference**: *"Warming Up Cold-Start CTR Prediction by Learning Item-Specific Feature Interactions"* (KDD 2024)

**Key Characteristics**:
- Uses hypernetworks to generate item-specific feature graphs
- Customized GNN for feature interaction capture
- Meta-learning strategy for fast adaptation to new items

---

## 4. Model Performance Comparison

### 4.1 Standard Recommendation Performance (Books Dataset)

| Model | Precision@10 | Recall@10 | Rank |
|-------|-------------|-----------|------|
| **LightGCN** | **0.0175** | **0.0803** | 1 |
| BPR | 0.0101 | 0.0453 | 2 |
| BiVAE | 0.0067 | 0.0292 | 3 |
| MF | 0.0047 | 0.0191 | 4 |

![image info](Books/app/backend/api/checkpoints/model_comparison_metrics.png)

> [!IMPORTANT]
> **Key Finding**: LightGCN significantly outperforms all other models on the Books dataset, achieving approximately 73% higher Precision@10 compared to the second-best model (BPR).

---

### 4.2 Cold-Start Performance Comparison (MovieLens-100K)

Cold-start evaluation tests model performance across four phases:
- **Cold**: No training data for new items (0 interactions)
- **Warm-A**: First batch of interactions (~10 samples)
- **Warm-B**: Additional interactions (~20 samples)
- **Warm-C**: More interactions (~30 samples)

#### Precision@10 (Cold-Start Phases)

| Phase | EmerG | LightGCN |
|-------|-------|----------|
| Cold | **0.2325** | 0.000 |
| Warm-A | **0.2509** | 0.0027 |
| Warm-B | **0.2604** | 0.0049 |
| Warm-C | **0.2654** | 0.0228 |

#### Recall@10 (Cold-Start Phases)

| Phase | EmerG | LightGCN |
|-------|-------|----------|
| Cold | **0.1749** | 0.000 |
| Warm-A | **0.1522** | 0.0019 |
| Warm-B | **0.1546** | 0.0038 |
| Warm-C | **0.1561** | 0.0282 |

```
Cold-Start Precision@10 Comparison:

EmerG Performance:
Cold    ████████████████████████ 0.2325
Warm-A  █████████████████████████ 0.2509
Warm-B  ██████████████████████████ 0.2604
Warm-C  ███████████████████████████ 0.2654

LightGCN Performance:
Cold    │ 0.000
Warm-A  │ 0.0027
Warm-B  │ 0.0049
Warm-C  ██ 0.0228
```

> [!CAUTION]
> **Critical Observation**: Traditional models like LightGCN fail completely in pure cold-start scenarios (Cold phase), achieving 0% precision. EmerG maintains strong performance even without any interaction data, demonstrating the effectiveness of meta-learning approaches for cold-start problems.

---

## 5. Loss Functions Summary

| Model | Loss Function | Description |
|-------|---------------|-------------|
| **MF (SVD)** | MSE + L2 Regularization | Minimizes squared error between predicted and actual ratings |
| **BPR** | BPR-OPT (Pairwise Ranking) | Maximizes the difference between positive and negative item scores |
| **BiVAE** | ELBO (Reconstruction + KL) | Variational lower bound with reconstruction and regularization terms |
| **LightGCN** | BPR Loss + Embedding Regularization | Pairwise ranking loss with embedding regularization |
| **EmerG** | BCE + Meta-Learning Loss | Binary cross-entropy with meta-optimization across tasks |

---

### LightGCN Training Loss Curve

![image info](Books/app/backend/api/checkpoints/LightGCN_loss.png)

---

## 6. Hyperparameters

### 6.1 Matrix Factorization (SVD)

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `n_factors` | 150 | Number of latent factors |
| `n_epochs` | 30 | Training epochs |
| `lr_all` | 0.005 | Learning rate for all parameters |
| `reg_all` | 0.02 | Regularization for all parameters |
| `random_state` | 42 | Random seed |

### 6.2 BPR

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `k` | 100 | Number of latent factors |
| `max_iter` | 200 | Maximum iterations |
| `learning_rate` | 0.01 | SGD learning rate |
| `lambda_reg` | 0.001 | Regularization coefficient |

### 6.3 BiVAE

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `k` | 50 | Latent dimension |
| `encoder_structure` | [100] | Hidden layer sizes |
| `n_epochs` | 50 | Training epochs |
| `batch_size` | 128 | Mini-batch size |
| `learning_rate` | 0.001 | Optimizer learning rate |
| `use_gpu` | True | GPU acceleration |

### 6.4 LightGCN

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `n_layers` | 3 | Number of GCN layers |
| `latent_dim` | 64 | Embedding dimension |
| `n_epochs` | 50 | Training epochs |
| `batch_size` | 1024 | Batch size for BPR sampling |
| `learning_rate` | 0.005 | Adam optimizer LR |
| `decay` | 0.0001 | L2 regularization weight |

### 6.5 EmerG (Cold-Start)

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `batch_size` | 512 | Batch size |
| `lr` | 0.001 | Outer learning rate |
| `lr_inner` | 0.01 | Inner loop learning rate |
| `meta_lr` | 0.001 | Meta-learning rate |
| `warm_lr` | 0.01 | Warm-up phase learning rate |
| `pretrain_epochs` | 2 | Pretraining epochs |
| `melu_epochs` | 11 | Meta-learning epochs |
| `epoch` | 11 | Fine-tuning epochs per item |
| `weight_decay` | 2.92e-07 | Weight decay coefficient |
| `gnn_layers` | 3 | Number of GNN layers |
| `embed_dim` | 16 | Feature embedding dimension |

---

## 7. Conclusion

### Key Findings

1. **Standard Recommendation**: LightGCN achieves the best performance on the Books dataset, demonstrating the effectiveness of graph neural networks for collaborative filtering.

2. **Cold-Start Challenge**: Traditional models (MF, BPR, BiVAE, LightGCN) completely fail in cold-start scenarios, achieving near-zero or zero performance when new items have no interaction history.

3. **Meta-Learning Solution**: EmerG significantly outperforms all traditional methods in cold-start scenarios by leveraging meta-learning and item-specific feature graphs.

4. **Trade-offs**: While LightGCN excels in warm-start scenarios with sufficient data, EmerG is essential for systems that need to handle new items effectively.

### Recommendations

- **For systems with stable item catalogs**: Use LightGCN for best performance
- **For systems with frequent new items**: Implement EmerG or similar meta-learning approaches
- **Hybrid approach**: Consider combining both methods based on item interaction history

---

## Training Details

| Model | Training Time | Dataset Size (30% sample) |
|-------|---------------|---------------------------|
| MF | ~0.5 seconds | ~300K ratings |
| BPR | ~1.7 seconds | ~300K ratings |
| BiVAE | ~20 seconds | ~300K ratings |
| LightGCN | ~17 seconds | ~300K ratings |

---

