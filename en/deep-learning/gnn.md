# Graph Neural Networks (GNN)

## Graph Basics

Q1: ***How are graphs represented? What are the adjacency matrix, degree matrix and Laplacian?***

- Graph $G=(V,E)$ with $|V|=N$ nodes and node feature matrix $X\in\mathbb{R}^{N\times d}$.
- **Adjacency matrix** $A\in\{0,1\}^{N\times N}$: $A_{ij}=1$ if edge $(i,j)$ exists; edge weights for weighted graphs. $A$ is symmetric for undirected graphs.
- **Degree matrix** $D=\mathrm{diag}(d_1,\dots,d_N)$, $d_i=\sum_j A_{ij}$.
- **Laplacian**:
    - Combinatorial Laplacian $L=D-A$
    - Symmetric normalized $L_{sym}=I-D^{-1/2}AD^{-1/2}$, eigenvalues in $[0,2]$
    - Random-walk normalized $L_{rw}=I-D^{-1}A$
- $L$ is positive semi-definite, $x^\top Lx=\frac{1}{2}\sum_{i,j}A_{ij}(x_i-x_j)^2$ measures the **smoothness** of a signal on the graph; the multiplicity of eigenvalue 0 = number of connected components.
- In practice large graphs are almost always stored **sparsely** (COO edge list `edge_index`, CSR) because $A$ is extremely sparse and dense $O(N^2)$ storage is infeasible.

Q2: ***What task levels exist in graph learning? How is readout done for graph-level tasks?***

- **Node-level**: node classification/regression (e.g. paper classification, user risk detection); put a classification head on the last-layer node representation $h_v^{(L)}$.
- **Edge-level**: link prediction, edge classification (e.g. whether a user interacts with an item, whether a transaction is fraudulent); use $f(h_u,h_v)$ such as dot product, concatenation + MLP, or Hadamard product.
- **Graph-level**: graph classification/regression (e.g. molecular property prediction); requires a **readout (pooling)** that aggregates the node set into a graph vector:
    - Simple permutation-invariant pooling: $h_G=\sum_v h_v$ / mean / max; sum is the most expressive (it can distinguish multiset sizes)
    - Concatenate readouts from every layer (as in GIN): $h_G=\Vert_{l}\,\mathrm{SUM}(\{h_v^{(l)}\})$
    - Attention pooling, Set2Set, hierarchical pooling (DiffPool, SAGPool, TopK pooling)
- **Readout must be permutation-invariant**, since node ordering is arbitrary.

## Message Passing and Classic Models

Q3: ***What is the message passing framework?***

- Most spatial GNNs can be unified as an **MPNN** (Gilmer et al., 2017):
$$m_v^{(l)}=\bigoplus_{u\in\mathcal{N}(v)}\phi^{(l)}\big(h_v^{(l-1)},h_u^{(l-1)},e_{uv}\big)$$
$$h_v^{(l)}=\psi^{(l)}\big(h_v^{(l-1)},m_v^{(l)}\big)$$
- Three ingredients: **message function** $\phi$, **permutation-invariant aggregation** $\bigoplus$ (sum/mean/max/attention), **update function** $\psi$ (MLP, GRU, etc.).
- The receptive field of an $L$-layer GNN is the **$L$-hop neighborhood**.
- Properties: **permutation-equivariant** with respect to node reordering (node-level outputs), **permutation-invariant** after readout (graph-level outputs); parameters are shared across all nodes and independent of graph size.

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

- **Spectral**: based on graph signal processing. Eigendecompose $L_{sym}=U\Lambda U^\top$, graph Fourier transform $\hat x=U^\top x$, graph convolution defined as $g_\theta * x=U g_\theta(\Lambda)U^\top x$.
    - Spectral CNN (Bruna 2014): learns $g_\theta(\Lambda)$ directly; needs $O(N^3)$ eigendecomposition, filters are non-localized and tied to a specific graph structure, so they cannot transfer to new graphs.
    - **ChebNet**: approximates $g_\theta(\Lambda)\approx\sum_{k=0}^{K}\theta_k T_k(\tilde\Lambda)$ with order-$K$ Chebyshev polynomials, avoids eigendecomposition, and filters are $K$-hop **localized**.
    - **GCN**: obtained from ChebNet with $K=1$ plus further simplification.
- **Spatial**: aggregation defined directly on node neighborhoods (GraphSAGE, GAT, GIN, MPNN).
    - Advantages: naturally local, supports batching/sampling, **inductive** learning, directed/heterogeneous/edge-featured graphs, easier to scale.
- Bottom line: GCN bridges the two; industry and most new research use the spatial view; the spectral view remains useful for analyzing low-pass filtering and over-smoothing.

Q5: ***Derive/explain the GCN propagation rule $H'=\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW)$***

- Origin: first-order ChebNet approximation with $\lambda_{max}\approx2$, $K=1$: $g_\theta * x\approx\theta_0x+\theta_1(L-I)x=\theta_0x-\theta_1D^{-1/2}AD^{-1/2}x$; setting $\theta=\theta_0=-\theta_1$ gives $\theta(I+D^{-1/2}AD^{-1/2})x$.
- **Renormalization trick**: $I+D^{-1/2}AD^{-1/2}$ has eigenvalues in $[0,2]$, which is numerically unstable when stacked, so use $\tilde A=A+I$ (self-loops) and $\tilde D_{ii}=\sum_j\tilde A_{ij}$:
$$H^{(l+1)}=\sigma\big(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)}\big)$$
- Node view: $h_v'=\sigma\Big(\sum_{u\in\mathcal{N}(v)\cup\{v\}}\frac{1}{\sqrt{\tilde d_u\tilde d_v}}W h_u\Big)$
- Why **self-loops**: retain the node's own information. Why **symmetric normalization**: prevent feature scale from exploding at high-degree nodes, while accounting for the degree of both source and target (compared with row normalization $\tilde D^{-1}\tilde A$, i.e. mean aggregation).
- It is essentially a **low-pass filter**: it smooths features of neighboring nodes, which is also the root cause of over-smoothing.
- Limitations: **transductive**, full-graph training is memory-heavy, aggregation weights are fixed by degree, isotropic.

Q6: ***What is the core idea of GraphSAGE? Why is it inductive?***

- **SAmple and aggreGatE** (Hamilton et al., 2017):
$$h_{\mathcal N(v)}^{(l)}=\mathrm{AGG}\big(\{h_u^{(l-1)},u\in\mathcal{S}(v)\}\big),\quad h_v^{(l)}=\sigma\big(W^{(l)}[h_v^{(l-1)}\,\Vert\,h_{\mathcal N(v)}^{(l)}]\big)$$
  usually followed by $\ell_2$ normalization.
- **Neighbor sampling**: sample a fixed $S_l$ neighbors per layer (e.g. 25, 10); cost is $O(\prod_l S_l)$, independent of graph size, enabling mini-batch training.
- **Aggregators**: Mean, LSTM (on randomly permuted neighbors; not strictly permutation-invariant), Pooling (element-wise max over an MLP), GCN-style aggregation.
- **Inductive**: it learns a function for "how to aggregate from neighbors" rather than a per-node embedding, so it can directly infer on **new nodes/new graphs** (only their features and neighbors are needed), well suited for industrial settings where users and items are constantly added.
- Self and neighbor representations are **concatenated** (not summed), acting like a skip connection that mitigates over-smoothing.
- Supports an unsupervised loss: positives from random-walk co-occurrence + negative sampling.

Q7: ***How does GAT compute attention coefficients? What does multi-head attention do?***

- For each edge $(i,j)$ (Veličković et al., 2018):
$$e_{ij}=\mathrm{LeakyReLU}\big(\mathbf a^\top[Wh_i\,\Vert\,Wh_j]\big),\quad \alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k\in\mathcal N(i)\cup\{i\}}\exp(e_{ik})}$$
$$h_i'=\sigma\Big(\sum_{j\in\mathcal N(i)\cup\{i\}}\alpha_{ij}Wh_j\Big)$$
- Softmax only over neighbors (**masked attention**), complexity $O(|E|)$.
- **Multi-head**: $K$ independent heads, concatenated in hidden layers and averaged in the output layer; stabilizes training and increases expressiveness.
- Versus GCN: neighbor weights are **learned adaptively from features** instead of fixed by degree (anisotropic), more interpretable, naturally inductive.
- Follow-up: **GATv2** (Brody et al., 2022) shows the original GAT is "static attention" — because $\mathbf a$ is applied linearly before LeakyReLU, the ranking of neighbors is the same for every query node; GATv2 uses $\mathbf a^\top\mathrm{LeakyReLU}(W[h_i\Vert h_j])$ to achieve dynamic attention.

Q8: ***What is GIN? How does GNN expressiveness relate to the WL test?***

- **1-WL test**: iteratively update each node's color to hash(own color, multiset of neighbor colors); if the final color histograms of two graphs differ, they are non-isomorphic.
- Xu et al. (2019, "How Powerful are GNNs?") prove: **the discriminative power of message passing GNNs is upper-bounded by 1-WL**; the bound is reached when both aggregation and update functions are **injective**.
- Mean / max aggregation is not injective: mean cannot distinguish $\{a,b\}$ from $\{a,a,b,b\}$, max cannot distinguish $\{a,b\}$ from $\{a,a,b\}$; **sum can**.
- **GIN**:
$$h_v^{(l)}=\mathrm{MLP}^{(l)}\Big((1+\epsilon^{(l)})h_v^{(l-1)}+\sum_{u\in\mathcal N(v)}h_u^{(l-1)}\Big)$$
  sum aggregation + MLP (universal approximation guarantees injectivity); graph-level readout concatenates per-layer sums.
- Limits of 1-WL: cannot distinguish certain regular graphs, cannot count triangles/cycles; ways around it: **k-WL / higher-order GNNs**, **subgraph GNNs**, adding **random features or positional encodings** (Laplacian PE, random-walk PE), structural features (node degree, cycle counts).
- Note: more expressive does not mean better generalization; on real node classification GCN/SAGE often match GIN.

## Problems of Deep GNNs

Q9: ***What is over-smoothing? How can it be mitigated?***

- Symptom: as depth increases, all node representations converge to the same value, nodes become indistinguishable and performance drops. GCN is usually best at 2–3 layers.
- Cause: GCN is a low-pass filter; repeatedly multiplying by $\hat A=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ makes $\hat A^k x$ converge as $k\to\infty$ to the subspace of the dominant eigenvector (which depends only on degree), wiping out information; measurable by **Dirichlet energy** or MAD.
- Remedies:
    - **Residual/skip connections**: ResGCN, **JKNet** (concatenate/max-pool all layer outputs), GCNII (**initial residual** $(1-\alpha)\hat AH+\alpha H^{(0)}$ + identity mapping)
    - **Decouple propagation from transformation**: **APPNP** (personalized PageRank propagation $Z^{(k+1)}=(1-\alpha)\hat AZ^{(k)}+\alpha H$), SGC
    - **Normalization**: PairNorm, DiffGroupNorm
    - **Random edge dropping**: DropEdge
    - Concatenating self and neighbor (GraphSAGE), gated updates

Q10: ***What is over-squashing? How does it differ from over-smoothing?***

- Symptom: the number of neighbors within $L$ hops grows exponentially with $L$, but must be compressed into a **fixed-size vector**, so information from distant nodes cannot propagate effectively and long-range tasks suffer (Alon & Yahav, 2021).
- The root cause is **topological bottlenecks** (tree-like structure, bridges, negatively curved regions), measurable via the sensitivity of the Jacobian $\partial h_v^{(L)}/\partial x_u$.
- Difference: over-smoothing is "all representations become the same" (a depth problem); over-squashing is "distant information cannot get through" (a topology bottleneck problem); there is some trade-off between the two.
- Remedies:
    - **Graph rewiring**: a fully-adjacent last layer (FA layer), curvature-based rewiring (SDRF), expander graphs
    - Add a virtual node connected to all nodes
    - **Graph Transformers** with global attention
    - Larger hidden dimension (mitigates but does not solve)

Q11: ***What other tricks are used to train deep GNNs?***

- **Residual / initial residual / dense connections** (DeepGCN brings ResNet, DenseNet and dilated convolution ideas, training 56+ layers).
- **Normalization**: BatchNorm, LayerNorm, GraphNorm, PairNorm.
- **Regularization**: DropEdge, DropNode, Dropout, DropMessage.
- **Decoupling**: feature transformation with an MLP first, then multi-step parameter-free propagation (APPNP, SGC, SIGN precomputing $\hat A^kX$), both deep and fast.
- **JK connections**: let each node adaptively choose its receptive field size.
- **Depth vs receptive field**: often you don't need depth; 2–3 layers + precomputed multi-hop features is enough.
- Training stability: smaller learning rate, weight decay, early stopping (small graph benchmarks have high variance; report mean and std over multiple random seeds).

## Tasks and Training

Q12: ***How is link prediction done? What negative sampling strategies exist?***

- Pipeline: GNN encoder produces node representations $z_u,z_v$ → decoder score $s(u,v)=z_u^\top z_v$ or $\mathrm{MLP}(z_u\odot z_v)$ → binary classification or ranking loss.
- Losses: BCE (positive vs negative edges), **BPR** $-\log\sigma(s(u,i)-s(u,j))$, InfoNCE.
- **Data leakage trap**: the graph used for message passing during training must not contain the supervision edges being predicted (split into message edges and supervision edges), otherwise the model "sees the answer". Temporal graphs must be split by time.
- **Negative sampling strategies**:
    - Uniform random sampling (simple, but mostly too easy)
    - Degree/popularity-based sampling ($\propto \deg^{0.75}$, as in word2vec)
    - **In-batch negatives**, requiring popularity bias correction (logQ correction)
    - **Hard negatives**: same category, high-scoring but not interacted, PinSage uses mid-ranked nodes by PageRank; curriculum from easy to hard
- Evaluation: AUC, Hits@K, MRR, Recall@K/NDCG@K.
- Follow-up: subgraph-based methods like SEAL capture structures like common neighbors, addressing the weakness of pure node embeddings (e.g. GCN) in link prediction, which "cannot distinguish automorphic nodes".

Q13: ***How do DeepWalk and node2vec work? How do they differ from GNNs?***

- **DeepWalk** (2014): run truncated **random walks** on the graph to generate node sequences, treat them as "sentences", and learn embeddings with **Skip-gram + negative sampling/hierarchical softmax**; nodes that co-occur get similar embeddings.
- **node2vec** (2016): **biased second-order random walk** with parameters $p$ (return) and $q$ (in-out):
    - After moving from $t$ to $v$, the unnormalized probability of next going to $x$: back to $t$ is $1/p$; $x$ adjacent to $t$ is $1$; otherwise $1/q$
    - $q<1$ favors **DFS**, capturing **homophily (communities)**; $q>1$ favors **BFS**, capturing **structural equivalence**
- Theoretically DeepWalk/LINE/node2vec are equivalent to implicit **matrix factorization** (NetMF).
- Differences from GNNs:
    - Random-walk methods are **shallow embeddings** (one lookup vector per node), $O(Nd)$ parameters, **transductive**, do not use node features
    - GNNs share parameters, use features, and can be inductive
- Industrial use: pretrained embeddings/retrieval features (e.g. EGES adds item side information on top of node2vec to tackle cold start).

Q14: ***What are heterogeneous graphs? How are they handled?***

- Heterogeneous graphs (HIN): multiple **node types** and **edge types**, e.g. user–item–shop, author–paper–venue.
- Methods:
    - **Meta-paths**: e.g. U-I-U, A-P-A; metapath2vec does random walks along meta-paths
    - **R-GCN**: one weight matrix per relation $h_i'=\sigma\big(W_0h_i+\sum_r\sum_{j\in\mathcal N_i^r}\frac{1}{c_{i,r}}W_rh_j\big)$; with many relations, use **basis decomposition** $W_r=\sum_b a_{rb}V_b$ or block-diagonal decomposition to avoid over-parameterization
    - **HAN**: node-level attention (within a meta-path) + semantic-level attention (across meta-paths)
    - **HGT**: Transformer-style attention parameterized by node/edge type, supports dynamic heterogeneous graphs
- In practice: PyG's `HeteroData` + `to_hetero`, DGL's `HeteroGraphConv` aggregate per relation and then combine.

## Scalability and Industrial Applications

Q15: ***How do GNNs scale to graphs with hundreds of millions of nodes?***

- Bottleneck of full-graph GCN: the whole graph and all layer activations must fit in GPU memory; with mini-batches, $L$-layer neighborhoods **explode exponentially**.
- **Node-wise sampling**: GraphSAGE uses fixed fan-out per layer; PinSage picks the Top-K important neighbors by random-walk visit counts.
- **Layer-wise sampling**: FastGCN, LADIES sample each layer independently to bound the number of nodes per layer.
- **Subgraph sampling**:
    - **ClusterGCN**: partition the graph into clusters with METIS; each batch trains on the induced subgraph of a few clusters; many intra-cluster edges and few inter-cluster edges, memory depends only on the subgraph; randomly combine multiple clusters to reduce bias
    - GraphSAINT: random node/edge/walk subgraph sampling with normalization to keep estimates unbiased
- **Precomputation/decoupling**: SGC, SIGN compute $\hat A^kX$ offline, so training is just an MLP, extremely scalable.
- **Historical embeddings**: GNNAutoScale etc. reuse stale neighbor representations.
- **Systems level**: graph partitioning + distributed sampling (DGL distributed, GraphLearn, PyG + feature store), **feature caching** for hot nodes, CPU/GPU cooperation, **layer-wise full-graph inference** at serving time to avoid redundant computation.

Q16: ***What are the key designs of PinSage?***

- Pinterest's industrial GraphSAGE variant (2018), trained on a 3-billion-node pin–board bipartite graph.
- **Random-walk-based neighborhoods**: run short random walks from a node, take the Top-T neighbors by visit count, and use normalized visit counts as **importance weights** for weighted aggregation (importance pooling).
- **Producer–consumer mini-batching**: CPU does sampling and feature fetching, GPU does computation, pipelined in parallel.
- **Curriculum learning + hard negatives**: items ranked 2000–5000 by PageRank serve as hard negatives, with the proportion of hard negatives increased epoch by epoch.
- **Max-margin ranking loss**: $\max(0,\,z_q^\top z_{n}-z_q^\top z_{i}+\Delta)$.
- **MapReduce inference**: avoids redundant computation; embeddings are generated in full and put into an ANN index for retrieval.

Q17: ***How are GNNs used in recommender systems? Why does LightGCN remove feature transformation and nonlinearity?***

- Recommendation can be modeled as a **user–item bipartite graph**; high-order connectivity ($u\to i\to u'\to i'$) is the collaborative signal.
- **NGCF** (2019): message passing on the bipartite graph with feature transformation, nonlinearity and an $e_i\odot e_u$ interaction term.
- **LightGCN** (He et al., 2020) found that in collaborative filtering with only ID embeddings (no semantic features), feature transformation and nonlinearity are **useless or even harmful** (they make training harder), and simplifies to:
$$e_u^{(k+1)}=\sum_{i\in\mathcal N_u}\frac{1}{\sqrt{|\mathcal N_u|}\sqrt{|\mathcal N_i|}}e_i^{(k)},\quad e_u=\sum_{k=0}^{K}\alpha_ke_u^{(k)}$$
  The only trainable parameters are the layer-0 ID embeddings, $\alpha_k$ is usually $1/(K+1)$; the layer combination acts like a residual and mitigates over-smoothing; trained with **BPR loss**.
- Follow-ups: SGL, SimGCL (add contrastive learning/noise perturbation to reduce popularity bias and improve the long tail), UltraGCN (approximates infinite layers with a constraint loss).
- Industrial deployment: mostly used for **retrieval** (two-tower/ANN search over GNN embeddings) or to provide graph embedding features to the ranking model; challenges are dynamic graph updates, cold start and online latency, so embeddings are typically computed offline + updated daily/hourly.

Q18: ***How are GNNs applied in financial risk control/fraud detection? What are the difficulties?***

- Graph construction: users, devices, IPs, phone numbers, bank cards and merchants as nodes; transactions/logins/shared devices as edges, forming a heterogeneous graph. Fraudsters often form **rings** sharing devices/IPs; graph structure exposes associations invisible from individual features.
- Tasks: node classification (fraudulent accounts), edge classification (fraudulent transactions), subgraph/community detection (ring mining), temporal link prediction.
- Representative methods: GEM (Alibaba, heterogeneous device graph), CARE-GNN / PC-GNN (handle **camouflage** and class imbalance), temporal GNNs (TGN, TGAT).
- Difficulties:
    - **Heterophily**: fraudsters deliberately connect to normal users to camouflage themselves, breaking the homophily assumption → neighbor filtering, separating same-class/different-class neighbors, high-pass filtering (e.g. H2GCN, FAGCN)
    - **Extreme class imbalance**: resampling, focal loss, neighbor oversampling
    - **Label delay and adversarial evolution**: fraud patterns change quickly; requires continual training and time-based validation splits
    - **Real-time**: online inference requires low latency; commonly offline graph embeddings + real-time features, or real-time subgraph sampling
    - **Interpretability**: regulators require explanations (GNNExplainer, presenting subgraph evidence)

Q19: ***What is a Graph Transformer? What are its pros and cons versus message passing GNNs?***

- Treat nodes as tokens and apply **global self-attention**, so a single layer can model any node pair, alleviating over-squashing and the 1-WL limit.
- The key is **injecting graph structure** (otherwise the Transformer cannot see edges):
    - **Positional encodings**: Laplacian eigenvector PE (must handle sign/basis ambiguity, e.g. SignNet), random-walk PE (RWSE)
    - **Structural encodings / attention bias**: **Graphormer** (2021) adds centrality encoding (degree), spatial encoding (shortest-path distance bias), edge encoding
    - **Hybrid architectures**: **GraphGPS** (2022) with each layer = local MPNN + global attention + FFN, a common general-purpose framework
- Cons: global attention is $O(N^2)$; large graphs need linear attention (Performer, sparse/expander-graph attention like Exphormer, NodeFormer, SGFormer); prone to overfitting on small data, relies on good PEs.
- Good fit: molecules/small-graph graph-level tasks (strong on OGB-LSC etc.), long-range dependency tasks; for large-scale node classification, well-tuned classic GNNs remain strong baselines.

Q20: ***What is the homophily assumption? How do you adapt GNNs to heterophilous graphs?***

- **Homophily**: connected nodes tend to share labels/similar features; edge homophily ratio $h=\frac{|\{(u,v)\in E:y_u=y_v\}|}{|E|}$. Low-pass models like GCN implicitly assume it.
- On **heterophilous graphs** (e.g. fraud detection, protein interaction, dating networks), neighbor averaging confuses the signal and can even be worse than an MLP.
- Improvements:
    - **Separate self and neighbor representations** (concatenate rather than sum), H2GCN
    - **Higher-order neighbors / multi-hop aggregation**: 2-hop neighbors may be more homophilous
    - **High-pass/adaptive filters**: FAGCN, GPR-GNN (learns polynomial filter coefficients, which can be negative)
    - Signed/gated messages: allow negative weights
    - Graph reconstruction: kNN graph based on feature similarity
- Interview practice: measure $h$ first, and always compare against an **MLP baseline**.

Q21: ***What are common GNN frameworks and engineering practices?***

- Frameworks: **PyTorch Geometric (PyG)**, **DGL**; for large-scale distributed: GraphLearn, GraphScope, DGL Distributed.
- Practical tips:
    - Sparse operations (`scatter`, `SpMM`), CSR to speed up neighbor sampling
    - Build an **MLP / LightGBM + graph statistic features** baseline first, then judge the gain from a GNN
    - Split data to avoid leakage: split by time; for link prediction separate message edges from supervision edges
    - Repeat experiments over multiple seeds on small datasets
    - At serving time, generate embeddings in offline batch + ANN retrieval, or real-time subgraph sampling + low-latency inference service
