# Graph Neural Networks (GNN)

## Graph Basics

Q1: ***How are graphs represented? What are the adjacency matrix, degree matrix and Laplacian?***

- **A graph is usually represented as a node feature matrix plus a connectivity structure: the adjacency matrix records who is connected to whom, the degree matrix records how many edges each node has, and the Laplacian is their difference, used to measure how smooth a signal is over the graph.**
- Adjacency matrix $A$: 1 where an edge exists (edge weight for weighted graphs); symmetric for undirected graphs.
- Degree matrix $D$: diagonal, with each node's degree on the diagonal.
- Laplacian $L=D-A$ is positive semi-definite; the symmetric normalized version $I-D^{-1/2}AD^{-1/2}$ is common; the number of zero eigenvalues equals the number of connected components.
- In practice large graphs are almost always stored sparsely (edge list `edge_index`, CSR), because a dense adjacency matrix needs $O(N^2)$ memory.

Q2: ***What task levels exist in graph learning? How is readout done for graph-level tasks?***

- **Graph tasks come in three levels: node-level (e.g. node classification), edge-level (e.g. link prediction) and graph-level (e.g. molecular property prediction). Graph-level tasks need a readout that summarizes all node representations into one graph vector, and it must not depend on node ordering.**
- Node-level: put a classification head directly on the last-layer node representations.
- Edge-level: score the two endpoint representations, e.g. dot product, Hadamard product, or concatenation followed by an MLP.
- Graph-level readout: sum / mean / max pooling is common, and sum is the most expressive; you can also concatenate readouts from every layer (as GIN does), or use attention pooling or hierarchical pooling (DiffPool, SAGPool).

## Message Passing and Classic Models

Q3: ***What is the message passing framework?***

- **Message passing is the unifying framework of most spatial GNNs: in each layer, every node collects "messages" from its neighbors, combines them with an order-independent aggregation, and then merges the result with its own previous representation to get a new one.**
- Three components: a message function, a permutation-invariant aggregation (sum/mean/max/attention), and an update function (MLP, GRU, etc.).
- Stacking $L$ layers lets each node see its neighbors within $L$ hops.
- Parameters are shared across all nodes and independent of graph size; node-level outputs are equivariant to node reordering, and graph-level outputs after readout are invariant.

```python
# Minimal PyG-style message passing layer
import torch
from torch_geometric.nn import MessagePassing

class SimpleConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=self.lin(x))

    def message(self, x_j):        # x_j: source node features
        return x_j
```

Q4: ***What is the difference between spectral and spatial GNNs?***

- **Spectral GNNs use the eigendecomposition of the Laplacian to define convolution in the "graph frequency" domain; spatial GNNs aggregate directly over node neighborhoods. GCN bridges the two, and today industry and most new research use the spatial view.**
- Early spectral methods need an eigendecomposition ($O(N^3)$), their filters are not local, and they are tied to one graph, so they cannot transfer to new graphs.
- ChebNet approximates the filter with a polynomial, avoiding the eigendecomposition and acting only on $K$-hop neighbors; GCN is its first-order simplification.
- Spatial methods (GraphSAGE, GAT, GIN) are naturally local, support sampled mini-batches and inductive learning, and handle directed, heterogeneous and edge-featured graphs more easily.
- The spectral view is still useful: it explains why GCN is a low-pass filter and why it over-smooths.

Q5: ***Derive/explain the GCN propagation rule $H'=\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW)$***

- **Each GCN layer applies a linear transform, takes a degree-normalized weighted average of the features of the neighbors and the node itself, and then applies a nonlinearity. It comes from a first-order approximation of ChebNet, plus "add self-loops + renormalize" to keep stacked layers numerically stable.**
- Self-loops ($\tilde A=A+I$): keep the node's own information during aggregation.
- Symmetric normalization: prevents feature scale from exploding at high-degree nodes, while accounting for the degrees of both source and target.
- It is essentially a low-pass filter that makes neighboring features more and more alike, which is the root of over-smoothing.
- Limitations: transductive, full-graph training is memory-heavy, and neighbor weights are fixed by degree rather than learned.

Q6: ***What is the core idea of GraphSAGE? Why is it inductive?***

- **The core of GraphSAGE is "sample and aggregate": each layer samples only a fixed number of neighbors, summarizes them with an aggregation function, concatenates the result with the node's own representation and transforms it. It learns a function for "how to aggregate from neighbors" rather than one embedding per node, so it applies directly to new nodes and new graphs; that is what inductive means.**
- A fixed fan-out (e.g. 25, 10) makes the cost independent of graph size and enables mini-batch training.
- Aggregator options: Mean, LSTM, Pooling (element-wise max after an MLP).
- Concatenating self and neighbor representations instead of adding them acts like a skip connection and mitigates over-smoothing.
- Well suited to industrial settings where users and items are constantly added; also supports unsupervised training with random-walk co-occurrence and negative sampling.

Q7: ***How does GAT compute attention coefficients? What does multi-head attention do?***

- **For each edge, GAT concatenates the transformed features of the two endpoints, scores them with a small vector plus LeakyReLU, applies softmax over that node's neighbors only to get attention weights, and then aggregates neighbors by those weights. Multi-head attention uses several independent heads, which stabilizes training and enriches the representation.**
- Softmax only over neighbors (masked attention), so cost is proportional to the number of edges.
- Multi-head: concatenate heads in hidden layers, average them in the output layer.
- Compared with GCN, neighbor weights are learned from features (not fixed by degree), more interpretable, and naturally inductive.
- Follow-up: GATv2 points out that the original GAT is "static attention" (every node ranks neighbors the same way) and moves LeakyReLU before the scoring vector to get dynamic attention.

Q8: ***What is GIN? How does GNN expressiveness relate to the WL test?***

- **A message passing GNN can distinguish graphs at most as well as the 1-WL graph isomorphism test, and only reaches that bound when both aggregation and update are injective. GIN is designed to meet that requirement: sum aggregation plus an MLP update, so it reaches the 1-WL bound.**
- 1-WL test: repeatedly update each node's color to a hash of "own color + multiset of neighbor colors"; if the final color histograms differ, the graphs are not isomorphic.
- Mean and max are not injective (e.g. they cannot tell $\{a,b\}$ from $\{a,a,b,b\}$); sum can.
- Limits of 1-WL: cannot distinguish certain regular graphs or count triangles and cycles; higher-order GNNs, subgraph GNNs, positional encodings or random features, and structural features can go beyond it.
- More expressive does not mean better generalization; on real node classification GCN/SAGE often perform about as well as GIN.
- Key formula: $h_v'=\mathrm{MLP}\big((1+\epsilon)h_v+\sum_{u\in\mathcal N(v)}h_u\big)$

## Problems of Deep GNNs

Q9: ***What is over-smoothing? How can it be mitigated?***

- **Over-smoothing means that as depth grows, all node representations become more and more alike until they cannot be told apart, and performance drops. The cause is that GCN-style models are low-pass filters: repeatedly averaging over neighbors wipes out differences, which is why GCN is usually best at 2–3 layers.**
- Residual and skip connections: ResGCN, JKNet (combines all layer outputs), GCNII (initial residual, mixing some of the initial features into every layer).
- Decouple propagation from transformation: APPNP (personalized PageRank propagation), SGC.
- Normalization and regularization: PairNorm, DropEdge.
- Concatenate self and neighbor (GraphSAGE) or use gated updates; smoothness can be measured with metrics such as Dirichlet energy.

Q10: ***What is over-squashing? How does it differ from over-smoothing?***

- **Over-squashing means information from distant nodes has trouble getting through: as layers increase, the number of nodes in the receptive field grows exponentially but must be compressed into a fixed-size vector, and bottlenecks in the graph topology (bridges, tree-like structure) make it worse. Over-smoothing is a depth problem where "all representations become the same"; over-squashing is a topology problem where "distant information cannot get through".**
- Consequence: poor results on tasks that need long-range dependencies.
- Remedies: graph rewiring (adding edges, curvature-based rewiring, a fully connected last layer), adding a virtual node connected to all nodes, and global attention with Graph Transformers.
- A larger hidden dimension helps but does not fix it; there is some trade-off between the two problems.

Q11: ***What other tricks are used to train deep GNNs?***

- **The main tools are residual/dense connections, normalization, regularization such as random edge dropping, and decoupling propagation from transformation; but often you do not need depth at all, and 2–3 layers plus precomputed multi-hop features are enough.**
- Architecture: residual, initial residual and dense connections (DeepGCN trains 56+ layers); JK connections let each node adaptively choose its receptive field.
- Normalization and regularization: BatchNorm, LayerNorm, PairNorm; DropEdge, DropNode, Dropout.
- Decoupling: transform features with an MLP first, then run multi-step parameter-free propagation (APPNP, SGC, SIGN), both deep and fast.
- Training habits: smaller learning rate, weight decay, early stopping; small graph benchmarks have high variance, so report mean and std over multiple random seeds.

## Tasks and Training

Q12: ***How is link prediction done? What negative sampling strategies exist?***

- **Link prediction first uses a GNN to get node representations, then a decoder (dot product or MLP) scores node pairs; real edges are positives and sampled non-edges are negatives, trained with a binary classification or ranking loss. The key points are choosing good negatives and making sure the graph used for message passing does not contain the edges being predicted.**
- Losses: BCE, BPR, InfoNCE; evaluation with AUC, Hits@K, MRR, Recall@K/NDCG@K.
- Avoid leakage: split edges into message edges and supervision edges, and split temporal graphs by time.
- Negative sampling: uniform random (simple but mostly too easy), degree/popularity-based, in-batch negatives (need popularity bias correction), and hard negatives (high-scoring but not interacted items, which can be added from easy to hard as a curriculum).
- Follow-up: subgraph-based methods like SEAL capture structures such as common neighbors, making up for what pure node embeddings miss.

Q13: ***How do DeepWalk and node2vec work? How do they differ from GNNs?***

- **DeepWalk runs random walks on the graph to get node sequences, treats them as "sentences", and learns embeddings with Skip-gram so that frequently co-occurring nodes get similar embeddings; node2vec makes the walk biased, with two parameters controlling whether it leans toward BFS or DFS. The fundamental difference from GNNs is that these are shallow embeddings: one lookup vector per node.**
- node2vec: leaning toward DFS captures communities (homophily); leaning toward BFS captures similarity of structural roles.
- In theory these methods are equivalent to implicit matrix factorization.
- Versus GNNs: random-walk embeddings are transductive, their parameter count grows with the number of nodes, and they ignore node features; GNNs share parameters, use features and can be inductive.
- Industrial use: pretrained embeddings or retrieval features, e.g. EGES adds item side information on top to tackle cold start.

Q14: ***What are heterogeneous graphs? How are they handled?***

- **Heterogeneous graphs have multiple node types and edge types (e.g. user–item–shop). The idea is to let the model tell types apart: either model along meta-paths, or give each relation its own parameters and combine them with attention.**
- Meta-paths: e.g. U-I-U; metapath2vec runs random walks along meta-paths.
- R-GCN: one weight matrix per relation; with many relations, basis decomposition shares parameters to avoid over-parameterization.
- HAN: node-level attention within a meta-path + semantic-level attention across meta-paths.
- HGT: Transformer-style attention parameterized by node/edge type; in practice use PyG's `HeteroData` or DGL's `HeteroGraphConv`.

## Scalability and Industrial Applications

Q15: ***How do GNNs scale to graphs with hundreds of millions of nodes?***

- **The core problem is that full-graph training does not fit in GPU memory, and with mini-batches the multi-layer neighborhood explodes exponentially. The fixes are sampling (neighbors, per layer, or subgraphs), precomputing propagation offline, and distributed systems with feature caching.**
- Node-wise sampling: GraphSAGE uses fixed fan-out; PinSage picks important neighbors with random walks. Layer-wise sampling: FastGCN, LADIES.
- Subgraph sampling: ClusterGCN partitions the graph into clusters and trains each batch on the subgraph of a few clusters; GraphSAINT samples random subgraphs with unbiased normalization.
- Precomputation/decoupling: SGC and SIGN precompute multi-hop features offline so training is just an MLP; GNNAutoScale reuses historical embeddings.
- Systems level: graph partitioning + distributed sampling, caching hot features, CPU/GPU cooperation, and layer-wise full-graph inference at serving time to avoid redundant computation.

Q16: ***What are the key designs of PinSage?***

- **PinSage is the GraphSAGE variant Pinterest deployed on a 3-billion-node graph. Its key designs are: choosing important neighbors with random walks and aggregating with weights, pipelining CPU sampling with GPU computation, training with hard negatives from easy to hard, and generating embeddings in batch with MapReduce for ANN retrieval.**
- Neighborhood: run short random walks from a node, take the Top-T most visited nodes as neighbors, and use visit counts as aggregation weights.
- Training: producer–consumer mini-batches, with the CPU doing sampling and feature fetching and the GPU doing computation.
- Negatives: items in the middle of the PageRank ranking serve as hard negatives, with their share increased epoch by epoch, together with a max-margin ranking loss.
- Inference: MapReduce avoids redundant computation, and all embeddings go into an ANN index.

Q17: ***How are GNNs used in recommender systems? Why does LightGCN remove feature transformation and nonlinearity?***

- **Recommendation can be modeled as a user–item bipartite graph, where multi-hop connections (user → item → similar user → item) are the collaborative filtering signal, and a GNN encodes them into embeddings through message passing. LightGCN found that in collaborative filtering with only ID embeddings and no semantic features, feature transformation and nonlinearity not only do not help but make training harder, so it keeps only normalized neighbor aggregation and then takes a weighted sum of the embeddings from all layers.**
- NGCF is the predecessor with feature transformation and nonlinearity; LightGCN's only trainable parameters are the layer-0 ID embeddings, trained with BPR loss.
- Combining layer embeddings acts like a residual and mitigates over-smoothing.
- Follow-ups: SGL and SimGCL add contrastive learning to improve the long tail; UltraGCN approximates infinite layers with a constraint loss.
- Deployment: mostly used for retrieval (ANN search over GNN embeddings) or as features for ranking; challenges are dynamic updates, cold start and latency, so embeddings are usually computed offline and updated daily/hourly.

Q18: ***How are GNNs applied in financial risk control/fraud detection? What are the difficulties?***

- **Users, devices, IPs, bank cards and merchants become nodes, and transactions, logins and shared devices become edges, forming a heterogeneous graph. Fraudsters often operate in rings that share devices and IPs, and graph structure exposes links you cannot see from individual features, which helps detect fraudulent accounts, fraudulent transactions and fraud rings.**
- Representative methods: GEM (heterogeneous device graph), CARE-GNN / PC-GNN (handle camouflage and class imbalance), temporal GNNs (TGN, TGAT).
- Heterophily: fraudsters deliberately connect to normal users to camouflage themselves, so neighbor filtering or high-pass filtering (H2GCN, FAGCN) is needed.
- Extreme class imbalance and adversarial evolution: resampling, focal loss, continual training and time-based validation splits.
- Real-time and interpretability: commonly offline graph embeddings + real-time features; regulators require explanations, which GNNExplainer can provide by showing subgraph evidence.

Q19: ***What is a Graph Transformer? What are its pros and cons versus message passing GNNs?***

- **A Graph Transformer treats nodes as tokens and applies global self-attention, so a single layer can connect any two nodes, which alleviates over-squashing and goes beyond the 1-WL limit. But a Transformer cannot see edges by itself, so the key is injecting graph structure; the cost is $O(N^2)$ computation.**
- Injecting structure: Laplacian or random-walk positional encodings; Graphormer uses degree, shortest-path distance and edge information as attention biases.
- Hybrid architecture: GraphGPS, where each layer = local message passing + global attention + FFN, is a common framework.
- Cons: large graphs need linear or sparse attention (Performer, Exphormer, SGFormer); prone to overfitting on small data and relies on good positional encodings.
- Good fit: graph-level tasks on small graphs such as molecules, and long-range dependency tasks; for large-scale node classification, well-tuned classic GNNs remain strong baselines.

Q20: ***What is the homophily assumption? How do you adapt GNNs to heterophilous graphs?***

- **Homophily means connected nodes tend to share labels and have similar features, and low-pass models like GCN assume it by default. On heterophilous graphs (e.g. fraud detection, dating networks), averaging over neighbors blurs the signal and can even do worse than an MLP, so the model needs to separate self from neighbors, look at farther neighbors, or learn to look at differences in a "high-pass" way.**
- Separate self and neighbor representations (concatenate rather than add), as in H2GCN.
- Multi-hop aggregation: 2-hop neighbors may actually be more homophilous.
- Adaptive filters: FAGCN and GPR-GNN learn filter coefficients that can be negative; or use signed/gated messages.
- Rebuild the graph (e.g. a kNN graph from feature similarity); in practice, measure the homophily ratio first and always use an MLP as a baseline.
- Key formula (edge homophily ratio): $h=\frac{|\{(u,v)\in E:y_u=y_v\}|}{|E|}$

Q21: ***What are common GNN frameworks and engineering practices?***

- **The mainstream frameworks are PyTorch Geometric (PyG) and DGL, with GraphLearn, GraphScope and DGL Distributed for large-scale distributed training. The most important engineering practice is to build a strong baseline first and strictly prevent data leakage before deciding whether a GNN is worth it.**
- Use sparse operators (`scatter`, SpMM) and CSR to speed up neighbor sampling.
- Build an MLP / LightGBM + graph statistic features baseline first, then measure the GNN's gain.
- Split data to avoid leakage: split by time, and for link prediction separate message edges from supervision edges; repeat experiments over multiple seeds on small datasets.
- Serving: generate embeddings in offline batch + ANN retrieval, or real-time subgraph sampling + a low-latency inference service.
