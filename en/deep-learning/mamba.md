# Mamba

Q1: ***What is a State Space Model (SSM)?***

- State space models come from control theory and use a hidden state $h(t)\in\mathbb{R}^N$ to map a 1D input signal $x(t)$ to an output $y(t)$:

$$h'(t)=Ah(t)+Bx(t),\qquad y(t)=Ch(t)+Dx(t)$$

- $A\in\mathbb{R}^{N\times N}$ is the **state transition matrix**, determining how the hidden state evolves and "forgets" over time; $B$ determines how the input is written into the state; $C$ determines how the output is read from the state; $D$ acts as a skip connection (often omitted or treated as a residual).
- In deep learning, SSMs apply one SSM independently to each input channel (or with shared structure) and then mix channels with linear layers, serving as the **sequence-mixing layer** of a sequence model that can replace RNNs or attention.

Q2: ***How is an SSM discretized? What is Zero-Order Hold (ZOH)?***

- Real sequences are discrete, so a **step size** $\Delta$ is introduced to convert the continuous parameters $(\Delta,A,B)$ into discrete parameters $(\bar{A},\bar{B})$:

$$h_t=\bar{A}h_{t-1}+\bar{B}x_t,\qquad y_t=Ch_t$$

- **Zero-Order Hold (ZOH)**: assume the input stays constant within each sampling interval $\Delta$; integrating the differential equation exactly gives:

$$\bar{A}=\exp(\Delta A),\qquad\bar{B}=(\Delta A)^{-1}(\exp(\Delta A)-I)\cdot\Delta B$$

- Other discretization methods include the bilinear transform (used in early S4). Mamba uses ZOH for $\bar{A}$ and often the simpler approximation $\bar{B}\approx\Delta B$.
- $\Delta$ can be understood as the **time resolution**: the larger $\Delta$, the more the state is "refreshed" by the current input; the smaller $\Delta$, the more the state tends to stay as it is.

Q3: ***What are the recurrent and convolutional views of an SSM? What is each used for?***

- **Recurrent view**: update step by step via $h_t=\bar{A}h_{t-1}+\bar{B}x_t$; each step has constant cost and only a fixed-size state is stored, which suits **autoregressive inference**, but training cannot be parallelized along the sequence.
- **Convolutional view**: when the parameters **do not change over time** (linear time-invariant, LTI), unrolling the recurrence gives:

$$y_t=\sum_{k=0}^{t}C\bar{A}^k\bar{B}x_{t-k}\quad\Rightarrow\quad y=x*\bar{K},\quad\bar{K}=(C\bar{B},C\bar{A}\bar{B},\dots,C\bar{A}^{L-1}\bar{B})$$

- That is, the output is the convolution of the input with a length-$L$ **SSM convolution kernel**, which can be computed with FFT in $O(L\log L)$ for **parallel training**.
- Conclusion: LTI SSMs can "train as a convolution, infer as a recurrence", getting the best of both; but Mamba's selective SSM is no longer LTI, so it cannot use the convolutional view and uses a parallel scan instead.

Q4: ***What are the main contributions of S4? What does HiPPO initialization do?***

- Naive SSMs have two problems: a randomly initialized $A$ struggles to remember long-range history (vanishing/exploding gradients), and computing the convolution kernel $\bar{K}$ requires repeated matrix powers, which is very expensive.
- **HiPPO** (Gu et al., 2020): derives a special class of $A$ matrices from the viewpoint of "approximating the entire input history online with orthogonal polynomials (e.g., Legendre polynomials)"; initializing with it lets the hidden state optimally compress the history, greatly improving **long-range dependency** modeling.
- **S4** (Gu et al., 2021, "Efficiently Modeling Long Sequences with Structured State Spaces"):
    - Parameterizes $A$ in a structured **Normal Plus Low-Rank (NPLR)** form so the convolution kernel can be computed efficiently and with numerical stability;
    - Performs very well on long-sequence benchmarks such as Long Range Arena, including the Path-X task that all previous models struggled to solve.
- Later simplifications: **S4D** shows that a diagonal $A$ (with a suitable initialization) achieves close results; **S5** uses a MIMO SSM with a parallel scan.

Q5: ***What is Mamba's selective SSM? How does it differ from S4?***

- The core of Mamba (Gu & Dao, 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces") is the **selection mechanism**: the SSM parameters **depend on the current input**:

$$B_t=\text{Linear}_B(x_t),\quad C_t=\text{Linear}_C(x_t),\quad\Delta_t=\text{softplus}(\text{Linear}_\Delta(x_t)+b)$$

- $A$ is still an input-independent parameter (a diagonal matrix), but because $\bar{A}_t=\exp(\Delta_tA)$, the state transition also varies with the input.
- Difference from S4: S4's parameters are **the same at every time step** (LTI) and process inputs the same way regardless of content; Mamba is **time-varying (input-dependent)**, so the model can **selectively remember or ignore** information based on content.
- Connection to gated RNNs: the paper notes that with $N=1$, $A=-1$, $B=1$, the selective SSM reduces to the gated form $h_t=(1-g_t)h_{t-1}+g_tx_t$ with $g_t=\sigma(\text{Linear}(x_t))$, so $\Delta$ plays a role similar to an RNN gate.

Q6: ***Why does selectivity matter? What do the selective copying and induction heads tasks show?***

- **Limitation of LTI models**: the convolution kernel is fixed, so they can only process information based on **relative position**, not **content**; they cannot "remember important tokens and skip irrelevant ones".
- **Selective Copying**: the tokens to be remembered are at random positions interspersed with irrelevant noise tokens. Because the spacing is not fixed, LTI models cannot solve it with a fixed kernel, whereas a selective SSM can decide based on content whether to write into the state.
- **Induction Heads**: upon seeing "... A B ... A", output "B" — content-based associative recall, one of the important mechanisms behind in-context learning in large language models. In the paper, Mamba solves this task and **extrapolates to sequences much longer than those seen in training**.
- **Intuition for $\Delta$**: large $\Delta_t$ → the state is reset and focuses on the current input; small $\Delta_t$ → the current input is ignored and the existing state is kept. $B_t$ and $C_t$ control what is written into and read out of the state.
- Essence: **selectivity = content-based compression of context**, an ability attention has and LTI SSMs lack.

Q7: ***How is Mamba's hardware-aware parallel scan implemented?***

- The selective SSM is no longer LTI and cannot be parallelized with convolution/FFT; computing the recurrence directly is slow, and the unrolled state has size $(B,L,D,N)$ — $N$ times larger than the input — making it expensive to write to GPU memory.
- Mamba's **hardware-aware algorithm**:
    - **Parallel scan**: the linear recurrence $h_t=\bar{A}_th_{t-1}+\bar{B}_tx_t$ is associative, so it can be computed with a work-efficient **parallel prefix sum (Blelloch scan)** in $O(\log L)$ parallel depth;
    - **Kernel fusion**: only the parameters $(\Delta,A,B,C)$ are read from slow **HBM** into fast **SRAM**; discretization, the scan and multiplication by $C$ are done in SRAM, and only the final output $y$ is written back to HBM, **never materializing the full hidden state in HBM**, which greatly reduces memory traffic;
    - **Recomputation**: intermediate states are not stored for the backward pass but recomputed, trading compute for memory, so that memory usage is comparable to a Transformer using FlashAttention.
- The idea is similar to FlashAttention: the bottleneck is **memory bandwidth** rather than compute.

Q8: ***How do Mamba and Transformers differ in complexity?***

- **Training**:
    - Transformer self-attention costs $O(L^2d)$;
    - Mamba is implemented via a scan, with compute growing **linearly** with sequence length: $O(LdN)$.
- **Inference (autoregressive generation)**:
    - Transformers must maintain a **KV cache** whose size grows linearly with the generated length $L$; each new token attends to all previous tokens, with per-step cost $O(L)$;
    - Mamba only maintains a **fixed-size state** $(D\times N)$; per-step compute and memory are **constant** $O(1)$, independent of context length, giving higher inference throughput and lower memory for long-sequence generation.
- Cost: a fixed-size state means the history must be **lossily compressed**, whereas attention can access the entire history losslessly (see Q11).

Q9: ***What does the Mamba block look like?***

- The Mamba block merges the H3 block with a gated MLP (GLU-like); the network is a stack of **homogeneous Mamba blocks** and **no longer needs separate MLP/FFN layers**:
    1. After LayerNorm/RMSNorm, the input is **expanded** by a linear layer (expansion factor usually 2) and split into two branches;
    2. **Main branch**: a short **causal 1D convolution** (causal conv1d) → SiLU activation → **selective SSM**;
    3. **Gate branch**: SiLU activation;
    4. The two branches are **multiplied element-wise** (gating) and projected back to the original dimension by a linear layer;
    5. Plus a residual connection.
- $A$ uses a diagonal real parameterization, commonly $A=-\exp(A_{\log})$ to ensure stability (negative eigenvalues, decaying state), with initialization following S4D-Real.
- The short convolution provides local token mixing, the SSM handles long-range dependencies, and gating increases expressiveness.

Q10: ***What are Mamba-2 and SSD (Structured State Space Duality)?***

- Mamba-2 (Dao & Gu, 2024, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality") proposes the **SSD framework**:
    - $A_t$ is restricted to a **scalar times the identity** ($A_t=a_tI$), so the whole sequence transformation can be written as a matrix multiplication $y=Mx$, where

$$M_{ji}=C_j^\top\left(\prod_{k=i+1}^{j}a_k\right)B_i,\quad j\ge i$$

- $M$ is a **semiseparable matrix**.
- **Duality**:
    - **Linear (recurrent) form**: computed via the SSM recurrence, with complexity linear in $L$;
    - **Quadratic (attention) form**: let $Q=C$, $K=B$, $V=x$; then $y=(L\circ QK^\top)V$, where $L$ is a lower-triangular mask matrix built from the cumulative decay $\prod a_k$ — this is exactly **linear attention with a decay mask**.
- **SSD algorithm**: split the sequence into chunks; within each chunk use the quadratic matrix-multiplication form (making full use of GPU **Tensor Cores**), and between chunks pass states in the recurrent form. This is much faster than Mamba-1's scan implementation and allows **larger state dimensions** $N$.
- Architectural changes: $B$, $C$, $\Delta$ and the input $x$ are produced **in parallel** at the start of the block (like $Q$, $K$, $V$ in attention), a multi-head structure is introduced, and an extra normalization layer is added before the output, making tensor parallelism easier.

Q11: ***What are the main weaknesses of SSMs such as Mamba?***

- **Weaker in-context recall and copying**: the fixed-size state must compress the entire history, so when a task requires **precise retrieval** of arbitrary information from the context (e.g., looking up a phone number in a long text, copying a long string, associative recall over many key-value pairs), SSMs are often worse than attention.
- Theoretical and empirical work (e.g., Jelassi et al., 2024, "Repeat After Me: Transformers are Better than State Space Models at Copying") shows that Transformers need only a few layers to copy strings of arbitrary length, while the copying ability of fixed-state models is limited by state size.
- **In-context learning**: on few-shot tasks requiring precise extraction of formats or examples from the prompt, pure SSM models may lag behind Transformers of the same size.
- **Ecosystem and tooling**: optimization, parallelism and inference frameworks for attention are more mature; SSMs need specialized kernels.
- Root cause: **attention = lossless retrieval from the KV cache; SSM = lossy fixed-size state compression** — a fundamental trade-off between efficiency and recall.

Q12: ***What are attention-Mamba hybrid models? Why do they work?***

- **Idea**: **insert a small number of attention layers** into a network dominated by Mamba layers, combining the strengths of both:
    - Mamba layers provide efficient sequence modeling with low compute and memory costs;
    - A few attention layers provide precise in-context retrieval and copying, compensating for the weaknesses of SSMs.
- Representative work:
    - **Jamba** (AI21, 2024): interleaves Transformer layers and Mamba layers and adds MoE to increase capacity;
    - **Zamba**: a Mamba backbone plus a shared attention block;
    - **Samba**: combines Mamba layers with sliding-window attention;
    - NVIDIA's systematic comparison of Mamba, Mamba-2 and hybrid models shows that hybrids make up for pure SSMs' shortcomings on recall-type tasks.
- Benefit: the KV cache exists only in a few attention layers, so memory and long-context inference costs are far lower than for pure Transformers, while strong recall and in-context learning abilities are retained.

Q13: ***How is Mamba applied to vision and other modalities?***

- Challenge: images are 2D and non-causal, whereas an SSM is a 1D causal sequence model.
- **Vision Mamba (Vim)**: splits the image into a sequence of patches and uses a **bidirectional SSM** (one forward scan and one backward scan) to capture global context, with position embeddings added.
- **VMamba**: proposes a **2D selective scan (SS2D / cross-scan)**, unrolling the patch sequence along four directions, scanning each and merging the results to better model 2D spatial relationships.
- Other modalities:
    - **Audio and speech**: modeling long waveform or spectrogram sequences;
    - **Genomics**: DNA sequences are extremely long, so the linear-complexity advantage is clear (e.g., Caduceus uses a bidirectional, reverse-complement-equivariant design);
    - **Video, point clouds, graphs, time series**, etc.
- Reflection: **MambaOut** (Yu & Wang, 2024) argues that short-sequence, non-causal tasks like image classification do not need SSMs; SSMs are better suited to **long-sequence, autoregressive** tasks.

Q14: ***How does Mamba compare with linear attention, RetNet and RWKV?***

- They all belong to the family of **linear RNNs / sub-quadratic sequence models** and can be written in a unified form: a recurrence with a matrix (or expanded) state $S_t=G_t\odot S_{t-1}+k_tv_t^\top$ and output $o_t=S_t^\top q_t$, with parallelizable training and constant-state inference. They differ mainly in the **form of the decay/gate $G_t$**:
    - **Linear attention** (Katharopoulos et al., 2020): replaces softmax with a kernel feature map, $S_t=S_{t-1}+\phi(k_t)v_t^\top$, with **no decay**; historical information keeps accumulating and is hard to forget;
    - **RetNet** (Sun et al., 2023): introduces a **fixed exponential decay** $\gamma$ (a constant per head) and supports parallel, recurrent and chunkwise computation forms; the decay is input-independent;
    - **RWKV**: implements linear attention with **learnable per-channel decay** via token shift and the WKV mechanism; in early versions the decay is input-independent, and from RWKV-6 onward **data-dependent** decay is introduced;
    - **Mamba / Mamba-2**: the decay is determined by the input-dependent $\Delta_t$, i.e., it is **data-dependent (selective)**; Mamba-2 can be viewed as linear attention with a scalar data-dependent gate, very close to GLA (Gated Linear Attention).
- Trend: **data-dependent gating/decay** is key to improving the expressiveness of linear models; further improvements such as DeltaNet and Gated DeltaNet update the state with the **delta rule**, improving associative recall.

Q15: ***What roles do Mamba's $\Delta$, $A$, $B$, $C$ play? How are they initialized?***

- **$\Delta$ (step size)**: controls the trade-off between the current input and the historical state, similar to forget/input gates; softplus ensures it is positive, and the bias is initialized so that $\Delta$ falls in a suitable range (e.g., on the order of $[0.001,0.1]$).
- **$A$ (state matrix)**: diagonal and negative to ensure stability; together with $\Delta$ it determines the decay rate $\exp(\Delta A)$. Different state dimensions correspond to different decay rates, amounting to memory at multiple time scales.
- **$B$ (input matrix)**: input-dependent; determines **what information is written into the state**;
- **$C$ (output matrix)**: input-dependent; determines **what information is read out of the state**;
- Analogy with attention: $C$ is like the query, $B$ like the key, $x$ like the value, and $\exp(\Delta A)$ like a position mask that decays over time.

Q16: ***In which scenarios does Mamba shine? How should one decide when choosing a model?***

- **Advantageous scenarios**:
    - **Very long sequences**: genomics, audio, long time series, long documents, etc., where linear complexity and constant inference memory are clear advantages;
    - **High-throughput inference**: no KV cache, suitable for large-batch, long-output generation;
    - **Streaming/online processing**: the natural recurrent form suits real-time and edge-device scenarios.
- **Selection considerations**:
    - If the task depends heavily on **precise retrieval and copying** (e.g., long-context QA, RAG, exact references in code completion), a pure SSM may underperform a Transformer; consider a hybrid architecture;
    - In short-sequence tasks, the quadratic cost of attention is not the bottleneck, so Mamba's advantage is limited;
    - Consider ecosystem maturity, specialized kernel support, and whether deployment frameworks support it.
