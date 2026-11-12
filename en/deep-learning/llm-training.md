# LLM Training

## Distributed Training

Q1: ***How much GPU memory does it take to train a model with $\Psi$ parameters? How do you estimate it?***

- Memory = **model states** (parameters, gradients, optimizer states) + **activations** + temporary buffers/fragmentation.
- With mixed precision + Adam, the model states take about **16 bytes per parameter**:
    - BF16/FP16 parameters: 2 bytes; BF16/FP16 gradients: 2 bytes
    - FP32 master weights: 4 bytes; Adam first moment $m$: 4 bytes; second moment $v$: 4 bytes
    - Total $2+2+12 = 16\Psi$ bytes (the ZeRO paper denotes the optimizer-related constant as $K=12$)
- Example: a 7B model needs about $7\times10^9 \times 16 \approx 112$ GB for model states alone, which does not fit on a single 80GB GPU, so sharding (ZeRO/FSDP) or offloading is required.
- **Activations** scale with batch size, sequence length and number of layers. Megatron's estimate: per layer per sample roughly $s\,h\,(34 + 5\,a\,s/h)$ bytes ($s$ sequence length, $h$ hidden size, $a$ number of heads, without recomputation/tensor parallelism), where the $5as^2$ term comes from the attention matrix; with FlashAttention this quadratic term essentially disappears.
- Inference only needs the parameters (+ KV cache): about $2\Psi$ bytes in BF16.
- Follow-up: training compute is commonly estimated as **$C \approx 6ND$** FLOPs (forward $2ND$, backward $4ND$), where $N$ is the parameter count and $D$ the number of training tokens.

Q2: ***How does data parallelism (DP) work? How does Ring All-Reduce work?***

- **Data parallelism**: each GPU holds a full model replica and processes a different data shard; after the backward pass, gradients are averaged with an **All-Reduce**, and each replica applies the same parameter update.
- PyTorch DDP groups gradients into buckets and communicates while the backward pass is still running, achieving **compute-communication overlap**.
- **Ring All-Reduce**: $N$ GPUs form a ring and the gradient is split into $N$ chunks:
    1. **Reduce-Scatter** ($N-1$ steps): at each step, a chunk is sent to the next GPU and accumulated; at the end each GPU holds the fully reduced result for one chunk;
    2. **All-Gather** ($N-1$ steps): the completed chunks are propagated around the ring; at the end every GPU has the full result.
- Each GPU sends $2\frac{N-1}{N}M$ data ($M$ is the gradient size), **essentially independent of the number of GPUs, i.e. bandwidth-optimal**; the downside is latency grows linearly with $N$, so at large scale tree/hierarchical algorithms are used (NVLink within a node, InfiniBand across nodes).
- Limitation: every GPU stores the full model states, which no longer fit once the model is large → this motivates ZeRO.

Q3: ***What do the three stages of ZeRO partition? How do memory and communication change?***

- ZeRO (Zero Redundancy Optimizer, DeepSpeed) eliminates the redundancy of model states in data parallelism. Let the DP degree be $N_d$:

| Stage | Partitioned | Model-state memory per GPU | Communication (vs. DP) |
|---|---|---|---|
| DP baseline | nothing | $16\Psi$ | $2\Psi$ |
| ZeRO-1 ($P_{os}$) | optimizer states | $4\Psi + 12\Psi/N_d$ | $1\times$ |
| ZeRO-2 ($P_{os+g}$) | + gradients | $2\Psi + 14\Psi/N_d$ | $1\times$ |
| ZeRO-3 ($P_{os+g+p}$) | + parameters | $16\Psi/N_d$ | $1.5\times$ |

- ZeRO-1/2: replace All-Reduce with Reduce-Scatter (each GPU only receives the gradient shard it owns) + All-Gather of parameters after the update; total communication is unchanged.
- ZeRO-3: before computing each layer in forward/backward, that layer's parameters must be All-Gathered and are freed after use, so communication rises to about $3\Psi$.
- Extensions: **ZeRO-Offload / ZeRO-Infinity** offload optimizer states/parameters to CPU memory or NVMe; **ZeRO++** reduces cross-node communication through quantized communication and hierarchical partitioning.
- Follow-up: ZeRO partitions model **states**, not **activations**; activations still need recomputation or sequence/tensor parallelism.

Q4: ***What is FSDP? How does it relate to DeepSpeed ZeRO?***

- **FSDP (Fully Sharded Data Parallel)** is PyTorch's native ZeRO-3-style implementation:
    - The model is wrapped into **FSDP units** (typically one Transformer block each);
    - Forward: All-Gather the unit's full parameters → compute → free the non-local shards;
    - Backward: All-Gather the parameters again → compute gradients → **Reduce-Scatter** the gradients, so each GPU keeps only its own shard of gradients and optimizer states.
- Key configuration:
    - `sharding_strategy`: `FULL_SHARD` (≈ZeRO-3), `SHARD_GRAD_OP` (≈ZeRO-2), `HYBRID_SHARD` (shard within a node, replicate across nodes to reduce cross-node traffic);
    - **prefetching** the next unit's parameters to overlap communication; mixed-precision policies; CPU offload.
- **FSDP2** (`fully_shard`) shards per parameter on top of DTensor, replacing FSDP1's FlatParameter, and composes more easily with TP/PP (e.g. TorchTitan).
- Comparison: same principle; the DeepSpeed ecosystem is more feature-rich (Offload, ZeRO++, config-driven), while FSDP integrates more tightly with native PyTorch features (`torch.compile`, DTensor).

Q5: ***How does tensor parallelism (Megatron-LM) split the MLP and attention?***

- Tensor parallelism splits **the weight matrices of a single layer** across GPUs, so each GPU computes only part of it.
- **MLP**: $Y = \text{GeLU}(XA)$, $Z = YB$
    - $A$ is split **by columns** $A=[A_1, A_2]$: each GPU computes $Y_i = \text{GeLU}(XA_i)$; since GeLU is element-wise, no communication is needed;
    - $B$ is split **by rows** $B=\begin{bmatrix}B_1\\B_2\end{bmatrix}$: each GPU gets a partial sum $Z_i = Y_iB_i$, and a final **All-Reduce** gives $Z=\sum_i Z_i$.
    - If $A$ were split by rows, synchronization would be needed before GeLU ($\text{GeLU}(X_1A_1+X_2A_2)\neq \text{GeLU}(X_1A_1)+\text{GeLU}(X_2A_2)$), hence "column first, then row".
- **Attention**: split by **attention heads**; $W_Q,W_K,W_V$ are column-split (each GPU owns some heads), the output projection $W_O$ is row-split, followed by an All-Reduce.
- Each Transformer layer needs 2 All-Reduces in forward and 2 in backward (the conjugate operators $f$/$g$ in Megatron).
- Communication is frequent and on the critical path, so TP is **usually confined to a single node** (NVLink, TP ≤ 8).
- The embedding is split along the vocabulary dimension (vocab parallel), and cross-entropy can also be computed in parallel to avoid gathering the full logits.

Q6: ***What is the "bubble" in pipeline parallelism? How do GPipe and 1F1B differ?***

- **Pipeline parallelism (PP)**: the model is split by layers into $p$ stages on different GPUs, and a batch is split into $m$ micro-batches that flow through in sequence.
- **Bubble**: some stages sit idle while the pipeline fills and drains. The bubble fraction is about
$$
\text{bubble} = \frac{p-1}{m+p-1}
$$
  so we need $m \gg p$.
- **GPipe**: run the forward pass of all micro-batches first, then all backward passes. Activations of all $m$ micro-batches must be kept at once, so memory is high.
- **1F1B (PipeDream-Flush)**: after warm-up, each stage alternates one forward and one backward. **The bubble size is the same as GPipe**, but at most $p$ micro-batches' activations are in flight, so **memory drops significantly**.
- **Interleaved 1F1B** (Megatron virtual pipeline): each GPU owns $v$ non-contiguous chunks of stages, reducing the bubble to $1/v$ of the original, at the cost of more communication.
- Further: **Zero Bubble** (split backward into input-gradient B and weight-gradient W, and use W to fill bubbles), and DeepSeek-V3's **DualPipe** (bidirectional pipeline + compute-communication overlap).
- PP only transfers activations at stage boundaries (point-to-point communication), so communication volume is small and it suits **cross-node** links.

Q7: ***What are Sequence Parallelism (SP) and Context Parallelism (CP)?***

- **Megatron Sequence Parallelism (SP)**: for operators outside the TP regions such as LayerNorm and Dropout, activations are otherwise replicated on every GPU; SP splits these activations along the **sequence dimension** and replaces TP's All-Reduce with **Reduce-Scatter + All-Gather** (same communication volume), cutting this part of activation memory to $1/t$. Usually enabled together with TP.
- **Context Parallelism (CP) / Ring Attention**: for long contexts (e.g. 128K+), the whole sequence is split across GPUs and every layer is sharded along the sequence:
    - Each GPU holds its local Q, while K/V blocks are passed around a **ring** and results are accumulated with online softmax (FlashAttention-style), overlapping communication and compute;
    - With a causal mask the load is imbalanced, so a "zigzag" partition is commonly used to balance it.
- **DeepSpeed-Ulysses**: uses **All-to-All** to convert "sharded by sequence" into "sharded by head" for computing attention, then converts back; the parallel degree is limited by the number of attention heads (or KV heads under GQA).
- Essence: addresses the memory problem of **activations growing with sequence length**, complementary to ZeRO (which addresses model states).

Q8: ***How does expert parallelism (EP) work in MoE models?***

- **Expert parallelism (EP)**: different experts (FFNs) are placed on different GPUs, while non-expert parts (attention, etc.) still use DP/TP.
- Flow: the router selects experts for each token → **All-to-All dispatch** sends tokens to the GPUs holding their experts → expert computation → **All-to-All combine** sends results back and takes the weighted sum.
- Difficulties:
    - All-to-All is communication-heavy and often crosses nodes, so it must be overlapped with compute (e.g. DeepSeek-V3 restricts each token to at most 4 nodes and uses custom communication kernels);
    - **Load imbalance**: GPUs holding popular experts become bottlenecks → requires a load-balancing loss or capacity limits (see the MoE question).
- Common combination: EP shares the same GPU group as DP (expert parameters sharded by EP, non-expert parameters replicated by DP).

Q9: ***How do you choose the combination of DP, TP and PP in 3D parallelism?***

- Total GPUs $= \text{DP}\times\text{TP}\times\text{PP}$ (adding CP and EP gives 4D/5D parallelism).
- General principle (map the most communication-intensive dimension onto the highest-bandwidth links):
    1. **TP within a node** (NVLink), usually ≤ 8, because every layer has All-Reduces;
    2. **PP across nodes**, only transferring stage-boundary activations with low volume; introduced when the model does not fit in one node;
    3. **DP at the outermost level** to scale throughput; combined with ZeRO-1 (ZeRO-2/3 combined with PP conflict with micro-batching on gradient/parameter sharding and add communication).
- Practical recipe: first find the smallest TP×PP that fits the model; add CP for very long sequences; add EP for MoE; give all remaining GPUs to DP; then tune the micro-batch size and $m$ to reduce bubbles, and measure **MFU** (Model FLOPs Utilization).
- Small to medium models (≲30B) often just use FSDP/ZeRO-3 + activation recomputation, with no need for 3D parallelism.

Q10: ***What is the principle and cost of activation checkpointing?***

- In the forward pass, only the activations of some layers (e.g. the input of each Transformer block) are kept and the rest discarded; in the backward pass, the **forward is re-run** to recover the needed activations.
- **Trade compute for memory**:
    - Full recomputation: activation memory drops from $O(L)$ layers to roughly one checkpoint per layer; the extra cost is about **one forward pass**, i.e. about 33% more total compute (forward:backward ≈ 1:2).
    - The classic $\sqrt{n}$ scheme: checkpoint every $\sqrt{n}$ layers, memory $O(\sqrt{n})$.
- **Selective recomputation** (Megatron): only recompute parts that are memory-heavy but compute-cheap (e.g. attention softmax, dropout), costing only a few percent.
- Other ways to save activations: FlashAttention (which itself recomputes attention in backward), sequence parallelism, offloading activations to CPU.

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for block in self.blocks:
        x = checkpoint(block, x, use_reentrant=False)
    return x
```

Q11: ***What is gradient accumulation? What pitfalls should you watch for?***

- Split a large batch into $k$ micro-batches, run forward/backward on each in turn, **accumulate gradients**, and update the parameters only once every $k$ steps.
- Effective global batch: $B = b_{\text{micro}} \times k \times N_{\text{DP}}$. Trades time for memory without increasing peak activations.
- Caveats:
    - In DDP, the first $k-1$ steps should use `no_sync()` to avoid an All-Reduce at every step;
    - **Loss normalization**: if micro-batches contain different numbers of valid tokens (variable-length sequences, padding, SFT that only counts the response), averaging within each micro-batch and then averaging across micro-batches is **not equivalent** to a token-level average over the large batch. The correct approach is to normalize by the global total of valid tokens (HF Transformers fixed a bug of this kind in 2024);
    - Layers that rely on batch statistics, like BatchNorm, are not equivalent (LLMs use LayerNorm/RMSNorm, so no issue);
    - The learning-rate schedule should count **optimizer update steps**, not micro-steps.

## Numerics and Stability

Q12: ***How do FP16, BF16 and FP8 differ? Why does LLM training prefer BF16?***

| Format | Sign/Exponent/Mantissa | Max value | Characteristics |
|---|---|---|---|
| FP32 | 1/8/23 | $\sim3.4\times10^{38}$ | baseline |
| FP16 | 1/5/10 | 65504 | higher precision, small range, prone to overflow/underflow |
| BF16 | 1/8/7 | same range as FP32 | large range, low precision, usually no loss scaling needed |
| FP8 E4M3 | 1/4/3 | 448 | relatively more precise, commonly used in forward (weights, activations) |
| FP8 E5M2 | 1/5/2 | 57344 | larger range, commonly used for backward gradients |

- BF16 has the same exponent bits as FP32, so gradients and activations rarely overflow, training is stable, and loss scaling is almost never needed; it has been the default since the A100.
- **FP8 training** (H100 and later): matrix multiplications run in FP8, while master weights, optimizer states and sensitive ops (LayerNorm, softmax, accumulation) stay in higher precision;
    - **Scaling factors** are needed: Transformer Engine uses per-tensor delayed scaling (based on the amax history);
    - DeepSeek-V3 uses **fine-grained quantization** (1×128 tiles for activations, 128×128 blocks for weights) and higher-precision accumulation;
    - Blackwell supports **MXFP8/MXFP4** microscaling formats (one shared scale per 32 elements).
- The basic mixed-precision flow (see the LLM card): low-precision forward/backward, FP32 master-weight update.

Q13: ***What is loss scaling? Why does FP16 need it while BF16 usually does not?***

- The smallest normal FP16 number is about $6\times10^{-5}$ (subnormals down to about $6\times10^{-8}$), so many small gradients **underflow to 0**.
- **Loss scaling**: multiply the loss by $S$ (e.g. $2^{16}$) before backward, so gradients are scaled by $S$ into the representable range; divide the gradients by $S$ (unscale) before the update.
- **Dynamic loss scaling**:
    - inf/NaN detected → skip this update, $S \leftarrow S/2$;
    - no overflow for a number of consecutive steps (e.g. 2000) → $S \leftarrow 2S$.
- Gradient clipping must happen **after unscaling**.
- BF16 has the same exponent range as FP32 and does not suffer from underflow, so it generally does not need loss scaling.

```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(x)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

Q14: ***What do you do about loss spikes during LLM pre-training? Common causes and remedies?***

- **Common causes**:
    - Learning rate too high, or anomalous/low-quality data in a batch;
    - Attention logits growing too large and saturating the softmax (attention entropy collapse);
    - Output-layer logit drift; Adam's second-moment estimate lagging behind rare large gradients (sudden jump in update size);
    - Overflow in low precision.
- **After the fact**: roll back to a checkpoint before the spike, **skip** the offending data batches and resume (PaLM's approach), temporarily lowering the learning rate if needed.
- **Prevention**:
    - Gradient clipping (global norm 1.0), a smaller peak learning rate, sufficient warmup;
    - Adam $\beta_2 = 0.95$ (instead of 0.999) so the second moment tracks gradient changes faster; an appropriate $\epsilon$;
    - **QK-Norm** (normalizing Q and K) to bound attention logits;
    - **z-loss**: $\mathcal{L}_z = 10^{-4}\cdot \log^2 Z$ ($Z$ is the softmax denominator) to prevent output-logit drift;
    - Pre-LN / extra normalization layers, embedding-layer gradient shrinking (GLM-130B);
    - BF16 instead of FP16; data cleaning.
- Monitoring: loss, gradient norm, per-layer activation/weight norms, max attention logits.

Q15: ***Why use a warmup + cosine learning-rate schedule? What does gradient clipping do?***

- **Warmup**: early in training the parameters are random and Adam's second-moment estimate is based on few samples and unreliable, so a large learning rate easily causes divergence; linear warmup (e.g. a few hundred to a few thousand steps) raises the learning rate from 0 to its peak. Post-LN architectures depend on warmup especially heavily.
- **Cosine decay**:
$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$
  $\eta_{\min}$ is typically 10% of the peak. Drawback: the total number of steps $T$ must be fixed in advance, making continued training awkward.
- **WSD (Warmup-Stable-Decay)**: after warmup, keep the learning rate constant, then decay quickly over a short final phase (often switching to high-quality data for annealing at the same time); this makes it easy to branch off intermediate checkpoints for continued training and scaling experiments, and many models now use it.
- **Gradient clipping** (by global norm):
$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|_2}\right)
$$
  typically $c=1.0$, preventing occasional large gradients from causing drastic parameter changes; in distributed training the global norm must first be computed across GPUs. The gradient norm itself is also an important health metric.

Q16: ***Briefly introduce μP (Maximal Update Parametrization).***

- Problem: under standard parametrization (SP), the optimal learning rate and initialization scale shift with model **width**, and grid search on large models is unaffordable.
- **μP** (Yang & Hu, the Tensor Programs series, "μTransfer"): adjusts initialization variance and **per-layer learning rates** with width so that each layer's feature updates stay $\Theta(1)$ as width goes to infinity.
- Effect: hyperparameters tuned on a small proxy model (learning rate, initialization, etc.) **transfer zero-shot** to a much wider model.
- Typical rules under Adam: hidden-layer learning rate $\propto 1/\text{width}$; output logits multiplied by a $1/\text{width}$ factor; attention scaled by $1/d$ instead of $1/\sqrt{d}$.
- Limitations: the original theory mainly addresses width; transfer across depth needs extra treatment (e.g. depth-μP), and hyperparameters like batch size and training duration do not transfer directly.

## Architecture Details

Q17: ***What is the difference between Pre-LN and Post-LN? Why is Pre-LN + RMSNorm the mainstream today?***

- **Post-LN** (original Transformer, BERT): $x_{l+1} = \text{LN}(x_l + F(x_l))$
    - Every layer output is normalized; gradients near the output layer are large, deep models train unstably and **depend heavily on warmup**; when training succeeds, final quality may be slightly better.
- **Pre-LN** (mainstream since GPT-2): $x_{l+1} = x_l + F(\text{LN}(x_l))$
    - The residual path is an identity mapping, so gradients flow back directly; training is stable and tolerates larger learning rates;
    - Downside: the residual-stream norm grows with depth, so deeper layers contribute relatively less ("deep-layer degradation"), and a final LN is needed.
- Variants: DeepNorm (up-weight the residual + down-scale initialization, enabling 1000-layer Post-LN), Sandwich-LN / normalization both before and after each sub-layer (e.g. Gemma 2/3).
- **RMSNorm**:
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g
$$
    - Drops LayerNorm's **mean subtraction** and bias $\beta$, only rescaling;
    - Cheaper to compute (about one fewer reduction) with quality comparable to LayerNorm; used by LLaMA, Qwen, DeepSeek, etc.

Q18: ***What is SwiGLU? Why is the FFN hidden size often $\frac{8}{3}d$?***

- Standard FFN: $\text{FFN}(x) = \text{ReLU}(xW_1)W_2$, hidden size $4d$.
- **GLU family** (gated linear units): $\text{SwiGLU}(x) = \left(\text{Swish}(xW) \odot xV\right)W_2$, where $\text{Swish}(z) = z\cdot\sigma(z)$ (i.e. SiLU).
- One branch acts as a gate and the other as the value, multiplied element-wise, which is more expressive; Shazeer (2020) showed GLU variants achieve lower perplexity at the same compute.
- SwiGLU has 3 matrices ($W, V, W_2$); to keep the parameter count equal to a $4d$ standard FFN, the hidden size is $\frac{2}{3}\times 4d = \frac{8}{3}d$ (rounded to a multiple of 64/256 in practice).
- PaLM, the LLaMA family and most modern LLMs use SwiGLU.

Q19: ***What is the core idea of RoPE? Why does it encode relative position?***

- **RoPE (Rotary Position Embedding, RoFormer)**: group the dimensions of $q$ and $k$ in pairs; the $i$-th pair at position $m$ is rotated by angle $m\theta_i$:
$$
\begin{pmatrix} q'_{2i} \\ q'_{2i+1}\end{pmatrix} =
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1}\end{pmatrix},\quad \theta_i = b^{-2i/d},\ b=10000
$$
- Relative-position property: $\langle R_m q, R_n k\rangle = q^\top R_m^\top R_n k = q^\top R_{n-m} k$, so the attention score depends only on the **relative distance** $n-m$. It implements relative position encoding in the form of an absolute position encoding.
- Advantages: no extra parameters; applied directly to Q/K, compatible with linear attention/FlashAttention; has a certain long-range decay property.
- Different dimensions have different frequencies: low dimensions (small $i$) are high-frequency and encode local position; high dimensions are low-frequency and encode long-range position.
- Follow-up: RoPE is applied only to Q/K, not V; combining it with MLA (DeepSeek) requires decoupling a subset of dimensions to carry RoPE separately.

Q20: ***How do you extend a RoPE model's context from $L$ to $L' = sL$? How do PI, NTK-aware and YaRN differ?***

- Naive extrapolation: beyond the training length, low-frequency dimensions encounter **rotation angles never seen during training**, and perplexity explodes.
- **Position Interpolation (PI)**: compress positions $m \to m/s$ so all angles fall within the trained range; requires light fine-tuning (about 1000 steps). Drawback: high-frequency dimensions are compressed too, making neighboring tokens hard to distinguish and hurting short-text performance.
- **NTK-aware interpolation**: change the base instead of positions, $b' = b\cdot s^{d/(d-2)}$, which effectively **leaves high-frequency dimensions almost unchanged (extrapolation) while interpolating low-frequency ones**; works reasonably well even without fine-tuning.
- **NTK-by-parts**: treat each dimension according to the ratio of its wavelength to the context length: no interpolation for wavelengths much shorter than $L$, full interpolation for wavelengths longer than $L$, linear blending in between.
- **YaRN**: NTK-by-parts + **attention temperature scaling** (logits multiplied by $1/t$, empirically $\sqrt{1/t} = 0.1\ln s + 1$) to compensate for the change in attention entropy on long sequences; extends to 64K–128K with a small amount of fine-tuning data, adopted by Qwen, DeepSeek, etc.
- Others: **ABF** (simply enlarge the base, e.g. LLaMA 3 uses 500000, together with long-text continued training), LongRoPE (searches non-uniform per-dimension scaling factors).
- In practice it is usually "larger base / YaRN + long-context continued pre-training", evaluated with needle-in-a-haystack, RULER, etc.

Q21: ***What are top-k routing, the load-balancing loss and the capacity factor in MoE?***

- **Structure**: replace the dense FFN with $E$ expert FFNs; the router picks $k$ experts per token:
$$
y = \sum_{i\in \text{TopK}(p)} g_i\,E_i(x),\quad p = \text{softmax}(xW_r)
$$
    - Switch Transformer: $k=1$; Mixtral 8x7B: top-2 of 8; DeepSeek-V3: top-8 of 256 routed experts + 1 shared expert (fine-grained experts + shared experts).
    - Many parameters but little compute activated per token (total parameters vs. active parameters).
- **Load-balancing loss** (Switch):
$$
\mathcal{L}_{\text{aux}} = \alpha\, E \sum_{i=1}^{E} f_i\, P_i
$$
  $f_i$ is the fraction of tokens dispatched to expert $i$, $P_i$ the router's average probability for expert $i$; minimized under a uniform distribution. Prevents the "rich get richer" effect (a few experts get selected and the rest never train).
- **Capacity factor** $C$: each expert processes at most $\text{capacity} = C\cdot\frac{k\cdot T}{E}$ tokens ($T$ is the number of tokens in the batch); overflow tokens are **dropped** (only pass through the residual connection). A larger $C$ drops fewer tokens but wastes compute/memory, typically 1.0–1.25; inference can be dropless (e.g. MegaBlocks).
- Other tricks: **router z-loss** to stabilize router logits; DeepSeek-V3's **auxiliary-loss-free load balancing** (adds a per-expert bias used only for selection, adjusted dynamically by load), avoiding the harm an auxiliary loss does to the main objective.
- Follow-up: MoE training challenges include routing instability, All-to-All communication, high memory (all parameters must be stored), and overfitting during fine-tuning.

## Parameter-Efficient Fine-Tuning

Q22: ***How does LoRA work? How are the rank, initialization and merging handled?***

- Assumes the fine-tuning weight update is **low-rank**. Freeze the pre-trained weight $W_0\in\mathbb{R}^{d\times k}$ and train only a low-rank decomposition:
$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll \min(d,k)
$$
- **Rank $r$**: usually 8–64; trainable parameters drop from $dk$ to $r(d+k)$. $\alpha/r$ is a scaling factor so that the learning rate need not be re-tuned when $r$ changes (rsLoRA suggests $\alpha/\sqrt{r}$).
- **Initialization**: $A$ is randomly initialized (Gaussian/Kaiming), $B$ is **initialized to 0**, so $\Delta W = 0$ at the start and the model is exactly the pre-trained model.
- **Where to apply**: the original paper only adds it to $W_q, W_v$; in practice adding it to all linear layers (Q/K/V/O + MLP) works better.
- **Merging**: before inference compute $W = W_0 + \frac{\alpha}{r}BA$, giving **zero extra inference latency**; alternatively keep them unmerged so many LoRAs share one base model in serving (e.g. S-LoRA, multi-tenant).
- Advantages: low memory (no full gradients or optimizer states), small checkpoints, hot-swappable.
- Variants: **DoRA** (decomposes into magnitude and direction), **LoRA+** (different learning rates for $A$ and $B$), **PiSSA** (initialized from SVD principal components).
- Limitations (see the LLM card): when the target distribution differs greatly from pre-training (e.g. a new language, lots of new knowledge), a low-rank update may be insufficient.

Q23: ***What does QLoRA improve? What are NF4, double quantization and paged optimizers?***

- QLoRA = **4-bit quantized frozen base model** + LoRA adapters (BF16), with backpropagation passing through the quantized weights into the LoRA; it makes fine-tuning a 65B model possible on a single 48GB GPU.
- **NF4 (4-bit NormalFloat)**: pre-trained weights are roughly zero-mean normally distributed; NF4's 16 quantization levels are the **quantiles** of the standard normal distribution (normalized to $[-1,1]$), so each bin is expected to receive the same number of weights, which is information-theoretically optimal for normally distributed data. Absmax scaling is applied per block (64 elements).
- **Double Quantization**: one FP32 scale constant per 64 parameters costs $32/64=0.5$ bits/parameter; quantizing these constants again to 8-bit (block size 256) reduces the overhead to about 0.127 bits/parameter, saving about 0.37 bits per parameter on average.
- **Paged Optimizers**: use NVIDIA unified memory to automatically page optimizer states to CPU memory when GPU memory runs short, avoiding OOM from memory spikes such as long sequences.
- At compute time NF4 weights are **dequantized to BF16** for the matmul, so it is somewhat slower than plain LoRA; what it saves is memory.
- Follow-up: the LoRA weights are trained against a quantized base, so merging back into a 4-bit base introduces quantization error (it is usually merged into a BF16 base).

Q24: ***How do Adapters, Prefix Tuning, Prompt Tuning and LoRA differ?***

| Method | How it works | Trainable params | Inference overhead | Notes |
|---|---|---|---|---|
| **Adapter** (Houlsby 2019) | Insert a bottleneck MLP (down-project, nonlinearity, up-project) + residual in each layer | ~0.5–5% | Extra sequential layers, **adds latency** | Stable, good quality |
| **Prefix Tuning** | Prepend trainable prefix vectors to K/V in every layer | ~0.1% | Consumes context length | Unstable training, often reparametrized with an MLP |
| **Prompt Tuning** | Add soft prompts only before the input embeddings | Very few | Consumes context length | Approaches full fine-tuning as models grow; weak on small models |
| **P-tuning v2** | Like Prefix, continuous prompts in every layer | ~0.1–3% | Same as above | More robust for small/medium models and NLU tasks |
| **LoRA** | Add a low-rank increment to the weights | ~0.1–1% | Mergeable, **no extra latency** | Currently the most popular |
| **(IA)³** | Learn vectors that rescale K, V and FFN activations | Very few | Mergeable | Fewest parameters |

- Unified view (He et al. 2022): all of these can be seen as adding a correction term $\Delta h$ to hidden states; they differ in where they modify (attention/FFN), sequential vs. parallel insertion, and the composition function.
- Selection: LoRA/QLoRA is the default choice; consider Prompt Tuning for many tasks sharing a base with minimal parameters; consider full fine-tuning when data is abundant and the domain gap is large.

## Data and Scaling

Q25: ***What are the key design points of an LLM tokenizer? What is byte-level BPE?***

- **BPE**: start from characters and repeatedly merge the most frequent adjacent symbol pair in the corpus until the target vocabulary size is reached (WordPiece differs by selecting merges by likelihood gain; see the LLM/Transformer cards).
- **Byte-level BPE** (since GPT-2): the base vocabulary is 256 bytes, so any UTF-8 text can be represented and **there is no OOV/UNK**; the downside is that multi-byte characters (e.g. Chinese) may be split into incomplete byte fragments.
- **SentencePiece**: trains directly on raw text (treating spaces as an ordinary symbol `▁`), supports BPE and Unigram, language-agnostic; used by LLaMA 1/2 with byte fallback.
- Design points:
    - **Vocabulary size**: a large vocabulary compresses better (shorter sequences, faster inference) but has more embedding/output-layer parameters and under-trained rare tokens; the trend is upward (LLaMA 3 about 128K, multilingual models often 150K+);
    - **Pre-tokenization rules**: regex splitting on whitespace and punctuation; **splitting numbers into individual digits** helps arithmetic;
    - Balancing coverage and compression across languages/code; special tokens (chat templates, tool calls).
- Follow-up: problems caused by tokenizers — spelling/counting tasks are hard (the model cannot see characters), "glitch tokens" (tokens that barely appeared in training), unequal cost across languages.

Q26: ***How do you deduplicate, quality-filter and mix pre-training data?***

- **Deduplication**:
    - Exact dedup: hash documents/lines (URL, content hash);
    - **Fuzzy dedup**: **MinHash + LSH**, estimating Jaccard similarity over n-gram sets; documents above a threshold (e.g. 0.8) are clustered and only one is kept;
    - Substring dedup: suffix arrays to find long repeated spans; semantic dedup: embedding clustering (SemDeDup).
    - Benefits: less memorization and privacy leakage, better training efficiency, less test-set contamination.
- **Quality filtering**:
    - Heuristic rules: length, symbol/letter ratios, duplicated-line ratio, stop words (Gopher/C4 rules);
    - Language identification (fastText); perplexity filtering (CCNet uses KenLM);
    - **Quality classifiers**: train a classifier with high-quality corpora (Wikipedia, textbooks) as positives; FineWeb-Edu has an LLM score "educational value" and distills it into a small classifier;
    - Safety: toxicity, PII scrubbing; **benchmark decontamination** (n-gram overlap checks against eval sets).
- **Data mixture**: mix web, code, math, books, papers, multilingual data, etc. in proportions;
    - Methods: small-model ablations, **DoReMi** (a proxy model learns domain weights), RegMix (regression to predict the optimal mixture);
    - Upsample high-quality data (repeating up to ~4 epochs costs little); **anneal** on high-quality data at the end of training;
    - Code and math data help reasoning.

Q27: ***How do the Kaplan and Chinchilla scaling laws differ? What does "about 20 tokens per parameter" mean?***

- **Scaling laws**: loss follows a power law in parameter count $N$, data size $D$ and compute $C$, e.g.
$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$
  where $E$ is the irreducible loss. Given a compute budget $C\approx 6ND$, find the optimal split between $N$ and $D$.
- **Kaplan et al. (OpenAI, 2020)**: concluded model size matters more than data, $N_{\text{opt}}\propto C^{0.73}$, so extra compute should mostly go into a bigger model → led to "big model, little data" models like GPT-3 (175B / 300B tokens).
- **Chinchilla (Hoffmann et al., DeepMind, 2022)**: $N_{\text{opt}}\propto C^{0.5}$, $D_{\text{opt}}\propto C^{0.5}$; **parameters and data should scale in equal proportion**, with the optimum around **20 tokens per parameter**. The 70B Chinchilla trained on 1.4T tokens outperformed the 280B Gopher.
- Reasons for the discrepancy: Kaplan used a fixed learning-rate schedule length for all models (not matching the cosine cycle to the training duration), mainly smaller models, and counted only non-embedding parameters.
- **"Over-training" in practice**: the Chinchilla optimum is **training-compute optimal** and ignores inference cost. To make small models cheaper to serve, modern models go far beyond 20 tokens/parameter (e.g. LLaMA 3 8B was trained on about 15T tokens).
- Follow-up: scaling under data constraints (diminishing returns from repeated data), downstream task performance not necessarily changing smoothly with loss, and scaling laws for inference-time compute (test-time compute).

Q28: ***Are "emergent abilities" of LLMs real?***

- **Emergent abilities** (Wei et al., 2022): some abilities (multi-step arithmetic, chain-of-thought reasoning, etc.) are near random on small models and suddenly improve sharply beyond a certain scale, so they cannot be predicted by extrapolating from small models.
- **Critique** (Schaeffer et al., 2023, "Are Emergent Abilities a Mirage?"): emergence is largely **an artifact of the evaluation metric**:
    - Exact match, multiple-choice accuracy, etc. are **nonlinear/discontinuous** metrics; per-token accuracy improves smoothly, but requiring the whole sequence to be correct makes it look like a "jump";
    - With continuous metrics (e.g. token-level edit distance, log-likelihood of the correct answer), performance grows smoothly with scale;
    - Few evaluation samples for small models also give insufficient resolution and can create "jumps".
- **Middle ground**:
    - For users, task success (discrete) is what matters, so the "jump in usability" is real;
    - Some work views emergence through **pre-training loss**: abilities appear once the loss falls below a threshold, which is more predictable when loss rather than parameter count is on the x-axis;
    - Related to grokking and phase transitions, but the mechanism is still unsettled.
- Interview takeaway: be able to explain that "the choice of metric affects the conclusion", and point out why predictability matters for safety and planning.
