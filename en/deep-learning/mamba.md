# Mamba

Q1: ***What is a State Space Model (SSM)?***

- **An SSM comes from control theory and maps an input sequence to an output through a hidden state: at each step the matrix A decides how the old state evolves and forgets, B writes the input into the state, and C reads the output from the state. In deep learning it serves as a sequence-mixing layer that can replace an RNN or attention.**
- Usually an SSM is applied independently to each channel, and a linear layer then mixes the channels.
- D acts as a skip connection and is often omitted or treated as a residual.
- Key formula: $h'(t)=Ah(t)+Bx(t),\quad y(t)=Ch(t)$

Q2: ***How is an SSM discretized? What is Zero-Order Hold (ZOH)?***

- **Real sequences are discrete, so a step size Δ is used to turn the continuous parameters into discrete ones, giving a recurrence like $h_t=\bar{A}h_{t-1}+\bar{B}x_t$. Zero-order hold assumes the input stays constant within each step interval, and integrates the differential equation exactly under that assumption to get the discrete parameters.**
- Mamba uses ZOH for $\bar{A}$ and often the simpler approximation $\Delta B$ for $\bar{B}$; other methods include the bilinear transform.
- Δ can be read as time resolution: with a large Δ the state is refreshed more by the current input, with a small Δ it tends to stay as it was.
- Key formula: $\bar{A}=\exp(\Delta A)$

Q3: ***What are the recurrent and convolutional views of an SSM? What is each used for?***

- **The recurrent view updates the state step by step at constant cost per step, which suits autoregressive inference; the convolutional view applies when parameters do not change over time (LTI): unrolling the recurrence turns it into a convolution of the input with a fixed kernel, which can be trained in parallel with FFT. LTI SSMs can therefore "train as a convolution, infer as a recurrence".**
- The recurrent form cannot be parallelized along the sequence during training.
- The convolutional form costs $O(L\log L)$ with FFT.
- Mamba's selective SSM is no longer LTI, so it cannot use the convolutional view and relies on a parallel scan instead.

Q4: ***What are the main contributions of S4? What does HiPPO initialization do?***

- **S4 fixed two problems of naive SSMs: difficulty remembering long-range history, and the high cost of computing the convolution kernel. It initializes A with the HiPPO matrix to remember long histories, and parameterizes A as "normal plus low-rank" so the kernel can be computed efficiently and stably.**
- HiPPO derives a special A from "approximating the entire input history online with orthogonal polynomials", letting the hidden state compress the history optimally.
- S4 stood out on Long Range Arena, including Path-X, which no earlier model had solved.
- Later simplifications: S4D showed that a diagonal A with good initialization is enough; S5 uses a MIMO SSM with a parallel scan.

Q5: ***What is Mamba's selective SSM? How does it differ from S4?***

- **A selective SSM computes the parameters B, C and the step size Δ from the current input, so the model can decide based on content whether to remember or ignore information. S4's parameters are the same at every time step and treat every input the same way regardless of content.**
- A itself is input-independent, but the discretized $\exp(\Delta_t A)$ changes with Δ, so the state transition also depends on the input.
- Link to gated RNNs: under a particular simplification the selective SSM reduces to $h_t=(1-g_t)h_{t-1}+g_tx_t$, so Δ plays the role of an RNN gate.
- Key formula: $\Delta_t=\text{softplus}(\text{Linear}(x_t))$, with $B_t$ and $C_t$ likewise linear functions of $x_t$.

Q6: ***Why does selectivity matter? What do the selective copying and induction heads tasks show?***

- **An LTI model's convolution kernel is fixed, so it can process information only by relative position, not by content. Selectivity lets the model "remember important tokens and skip irrelevant ones", i.e. compress context based on content, which attention has and LTI SSMs lack.**
- Selective copying: the tokens to remember sit at random positions among noise tokens; a fixed kernel cannot solve it, a selective SSM can.
- Induction heads: output "B" after seeing "A B ... A", an important mechanism of in-context learning; Mamba solves it and extrapolates to sequences far longer than in training.
- Intuition: a large Δ resets the state and focuses on the current input, a small Δ ignores the current input; B and C control what is written and read.

Q7: ***How is Mamba's hardware-aware parallel scan implemented?***

- **A selective SSM cannot be parallelized with a convolution, and the unrolled hidden state is too large to write to GPU memory cheaply. Mamba uses a parallel scan to compute along the sequence in parallel, fuses discretization and the scan into one kernel that runs in fast SRAM, writes only the output back to memory, and recomputes the intermediate states during backward.**
- Parallel scan: the linear recurrence is associative, so a parallel prefix sum finishes it in logarithmic depth.
- Kernel fusion: the full hidden state is never materialized in HBM, greatly reducing memory reads and writes.
- Recomputation: trades compute for memory, bringing memory use on par with a Transformer using FlashAttention.
- Same idea as FlashAttention: the bottleneck is memory bandwidth, not compute.

Q8: ***How do Mamba and Transformers differ in complexity?***

- **In training, attention's compute grows quadratically with sequence length while Mamba's grows linearly; in inference, a Transformer's KV cache grows with the context and every step looks at all history, while Mamba keeps a fixed-size state, so each step has constant compute and memory.**
- Mamba therefore has higher inference throughput and uses less memory for long generations.
- Cost: the fixed state must compress history lossily, while attention can access all history losslessly (see Q11).

| | Training | Per inference step |
|---|---|---|
| Transformer | $O(L^2d)$ | $O(L)$, KV cache grows with L |
| Mamba | $O(LdN)$ | $O(1)$, fixed-size state |

Q9: ***What does the Mamba block look like?***

- **The Mamba block merges H3 with a gated MLP: the input is expanded and split into two branches; the main branch goes through a short causal convolution, SiLU and the selective SSM, the other goes through SiLU as a gate; the two are multiplied element-wise, projected back to the original size, and a residual is added. The whole network is a stack of these blocks, with no separate FFN.**
- The expansion factor is usually 2, with LayerNorm/RMSNorm in front.
- A is a diagonal negative real matrix (e.g. $A=-\exp(A_{\log})$), ensuring the state decays and training is stable.
- Division of labor: the short convolution mixes locally, the SSM handles long-range dependencies, and the gate adds expressiveness.

Q10: ***What are Mamba-2 and SSD (Structured State Space Duality)?***

- **Mamba-2 restricts the state matrix A to a scalar times the identity, under which the SSM can be written equivalently as a form of linear attention with a decay mask. This is "structured state space duality" (SSD): the same model can be computed linearly as a recurrence or with matrix multiplies as attention.**
- The SSD algorithm splits the sequence into chunks: within a chunk it uses the matrix-multiply form to exploit Tensor Cores, and between chunks it passes the state recurrently. It is much faster than Mamba-1 and allows a larger state dimension.
- Correspondence: C plays the role of Q, B of K, and the input x of V.
- Architecture changes: B, C and Δ are produced in parallel with x at the start of the block, heads are introduced, and a normalization is added before the output, making tensor parallelism easier.

Q11: ***What are the main weaknesses of SSMs such as Mamba?***

- **The main weakness is poor exact recall and copying: the fixed-size state must lossily compress all history, so when a task requires precisely retrieving arbitrary information from the context (finding a phone number, copying a long string, many key-value associations), SSMs usually fall behind attention.**
- Studies (e.g. "Repeat After Me") show Transformers can copy strings of arbitrary length, while fixed-state models are limited by their state size.
- On few-shot tasks that require extracting formats or examples exactly from the prompt, pure SSMs may lag Transformers of the same size.
- On the engineering side, the attention ecosystem is more mature, and SSMs need dedicated kernels.
- The fundamental trade-off: attention is lossless KV retrieval, an SSM is lossy fixed-size compression.

Q12: ***What are attention-Mamba hybrid models? Why do they work?***

- **Hybrid models are mostly Mamba layers with a few attention layers inserted: the Mamba layers do efficient sequence modeling, and the few attention layers provide exact retrieval and copying, covering the SSM's weak spot. The KV cache lives only in those few attention layers, so long-context inference costs far less than in a pure Transformer.**
- Representative work: Jamba (alternating Transformer and Mamba layers, plus MoE), Zamba (a shared attention block), Samba (combined with sliding-window attention).
- NVIDIA's systematic comparison shows hybrids close the gap of pure SSMs on recall tasks.

Q13: ***How is Mamba applied to vision and other modalities?***

- **The difficulty is that images are 2D and non-causal, while SSMs are 1D causal models. The usual approach splits the image into a patch sequence and scans it in several directions to get global context: Vision Mamba scans forward and backward, and VMamba scans along four directions and merges the results.**
- Other modalities: audio and speech, genomics (extremely long sequences, e.g. Caduceus), video, point clouds, time series.
- A counterpoint: MambaOut argues that short, non-causal tasks like image classification do not need SSMs, which are better suited to long, autoregressive tasks.

Q14: ***How does Mamba compare with linear attention, RetNet and RWKV?***

- **All are linear-complexity "linear RNNs" with parallel training and constant-size inference state, and they can be written in one form: "state = decay × old state + new key-value". The main difference is the decay: linear attention has none, RetNet uses a fixed decay, RWKV a learnable per-channel decay, and Mamba's decay is determined by the input.**
- Linear attention keeps accumulating history and struggles to forget.
- RWKV-6 onward also has data-dependent decay; Mamba-2 is very close to GLA (Gated Linear Attention).
- Trend: data-dependent gating is the key to expressiveness; DeltaNet and Gated DeltaNet update the state with the delta rule, further improving associative recall.
- Key formula: $S_t=G_t\odot S_{t-1}+k_tv_t^\top,\quad o_t=S_t^\top q_t$

Q15: ***What roles do Mamba's $\Delta$, $A$, $B$, $C$ play? How are they initialized?***

- **Δ controls the trade-off between the current input and the past state, like a forget gate; A together with Δ sets how fast the state decays; B decides what is written into the state; C decides what is read out of it.**
- Δ is kept positive by softplus, and its bias is initialized so it falls roughly in 0.001–0.1.
- A is diagonal and negative for stability; different dimensions decay at different rates, giving multi-timescale memory.
- Analogy to attention: C is like the query, B like the key, x like the value, and $\exp(\Delta A)$ like a positional mask that decays over time.

Q16: ***In which scenarios does Mamba shine? How should one decide when choosing a model?***

- **Mamba suits very long sequences, high-throughput generation and streaming, because its compute is linear, its inference memory is constant and it has no KV cache. If a task depends on exact retrieval and copying, though, a pure SSM may fall short of a Transformer, and a hybrid architecture is worth considering.**
- Strong scenarios: genomics, audio, long time series, long documents; large-batch long-output generation; real-time and edge devices.
- On short sequences attention's quadratic cost is not the bottleneck, so Mamba's advantage is limited.
- Also consider ecosystem maturity, dedicated kernel support and deployment-framework support.
