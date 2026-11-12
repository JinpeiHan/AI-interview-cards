# LLM Training

## Distributed Training

Q1: ***How much GPU memory does it take to train a model with $\Psi$ parameters? How do you estimate it?***

- **Training memory is mainly model states plus activations. With mixed precision + Adam, model states take about 16 bytes per parameter, so a 7B model needs about 112 GB for model states alone, which does not fit on a single 80GB GPU.**
- The 16 bytes: BF16 parameters 2 + BF16 gradients 2 + FP32 master weights 4 + Adam $m$ and $v$ at 4 each.
- Activations grow linearly with batch size, sequence length and layer count; the attention matrix adds a part that grows quadratically with sequence length, which FlashAttention essentially removes.
- Inference only needs the parameters (plus the KV cache): about $2\Psi$ bytes in BF16.
- Follow-up: training compute is commonly estimated as $C \approx 6ND$ FLOPs (2 for forward, 4 for backward), where $N$ is the parameter count and $D$ the number of tokens.

Q2: ***How does data parallelism (DP) work? How does Ring All-Reduce work?***

- **In data parallelism every GPU holds the full model and processes different data; after the backward pass, an All-Reduce averages the gradients and every replica applies the same update. Ring All-Reduce connects the GPUs in a ring and splits the gradient into N chunks, then runs a Reduce-Scatter followed by an All-Gather, each taking N−1 steps.**
- Each GPU sends about twice the gradient size, almost independent of the number of GPUs, so it is bandwidth-optimal; the downside is that latency grows linearly with the GPU count, so hierarchical/tree algorithms are used at large scale.
- PyTorch DDP groups gradients into buckets and communicates while the backward pass is still running, overlapping compute and communication.
- Limitation: every GPU stores the full model states, which stop fitting once the model is large; this motivates ZeRO.

Q3: ***What do the three stages of ZeRO partition? How do memory and communication change?***

- **ZeRO shards the redundant model states of data parallelism across GPUs: ZeRO-1 shards optimizer states, ZeRO-2 also shards gradients, and ZeRO-3 also shards parameters. The first two stages keep DP's communication volume; ZeRO-3 adds about 50%.**

| Stage | Partitioned | Model-state memory per GPU | Communication (relative to DP) |
|---|---|---|---|
| DP baseline | nothing | $16\Psi$ | $1\times$ |
| ZeRO-1 | optimizer states | $4\Psi + 12\Psi/N_d$ | $1\times$ |
| ZeRO-2 | + gradients | $2\Psi + 14\Psi/N_d$ | $1\times$ |
| ZeRO-3 | + parameters | $16\Psi/N_d$ | $1.5\times$ |

- ZeRO-1/2 replace All-Reduce with a Reduce-Scatter plus an All-Gather of parameters after the update, so total communication is unchanged.
- ZeRO-3 must All-Gather each layer's parameters before computing it and free them afterwards, hence more communication.
- Extensions: ZeRO-Offload/Infinity offload to CPU memory or NVMe; ZeRO++ uses quantization and hierarchical sharding to cut cross-node communication.

Q4: ***What is FSDP? How does it relate to DeepSpeed ZeRO?***

- **FSDP is PyTorch's native ZeRO-3-style implementation: it shards parameters per unit (usually one Transformer block), All-Gathers the full parameters before computing and frees them afterwards, and after the backward pass uses Reduce-Scatter so each GPU keeps only its own share of gradients and optimizer states.**
- Sharding strategies: `FULL_SHARD` ≈ ZeRO-3, `SHARD_GRAD_OP` ≈ ZeRO-2, `HYBRID_SHARD` shards within a node and replicates across nodes to reduce cross-node traffic.
- It supports prefetching the next unit's parameters to overlap communication, plus mixed precision and CPU offload.
- FSDP2 shards per parameter on top of DTensor and composes more easily with TP/PP.
- Same principle as DeepSpeed: DeepSpeed has more features (Offload, ZeRO++), while FSDP integrates better with native PyTorch features (`torch.compile`).

Q5: ***How does tensor parallelism (Megatron-LM) split the MLP and attention?***

- **Tensor parallelism splits a single layer's weight matrices across GPUs. In the MLP the first matrix is split by columns and the second by rows, so the GeLU in between can be computed independently on each GPU and only one All-Reduce is needed at the end; attention is split by heads, also with one All-Reduce at the end.**
- Why "columns first, then rows": if the first matrix were split by rows, a sum would have to be synchronized before the GeLU, because a nonlinearity cannot be applied to partial sums separately.
- Each layer needs 2 All-Reduces forward and 2 backward; this communication is frequent and on the critical path, so TP is usually kept within a node (NVLink, ≤8).
- The embedding is split along the vocabulary, and the cross-entropy can be computed in parallel to avoid gathering the full logits.

Q6: ***What is the "bubble" in pipeline parallelism? How do GPipe and 1F1B differ?***

- **Pipeline parallelism splits the model by layers into p stages and the batch into m micro-batches that flow through in turn; during ramp-up and drain some stages sit idle, and that idle time is the bubble. GPipe and 1F1B have the same bubble size, but 1F1B uses much less memory.**
- GPipe runs all forward passes before all backward passes and must keep m sets of activations.
- 1F1B, after a warm-up, alternates one forward and one backward per stage, so at most p sets of activations are in flight.
- Interleaved 1F1B gives each GPU several non-contiguous stages, shrinking the bubble at the cost of more communication; Zero Bubble and DualPipe fill the bubble further.
- PP only sends activations point-to-point at stage boundaries, so communication is small and it suits cross-node links.
- Key formula: bubble fraction $\approx \frac{p-1}{m+p-1}$, so you want $m \gg p$.

Q7: ***What are Sequence Parallelism (SP) and Context Parallelism (CP)?***

- **Both split along the sequence dimension to tackle activation memory that grows with sequence length. Megatron sequence parallelism only splits the parts outside TP, such as LayerNorm and Dropout; context parallelism splits the whole long sequence across GPUs, with every layer partitioned by sequence.**
- Sequence parallelism replaces TP's All-Reduce with Reduce-Scatter + All-Gather at the same communication volume, and is usually enabled together with TP.
- Context parallelism (Ring Attention): each GPU holds a local Q, K/V blocks are passed around a ring and accumulated with online softmax; under a causal mask a zigzag split balances the load.
- DeepSpeed-Ulysses uses All-to-All to turn "split by sequence" into "split by heads" for attention; its degree is limited by the number of heads.
- They address activations and complement ZeRO, which addresses model states.

Q8: ***How does expert parallelism (EP) work in MoE models?***

- **Expert parallelism places different experts on different GPUs: the router picks experts for each token, an All-to-All sends tokens to the GPUs hosting their experts for computation, and another All-to-All sends the results back for the weighted sum. The non-expert parts still use DP/TP.**
- Difficulty one: All-to-All traffic is heavy and often crosses nodes, so it must be overlapped with compute (e.g. DeepSeek-V3 routes each token to at most 4 nodes).
- Difficulty two: load imbalance, where GPUs hosting popular experts become bottlenecks; this needs a load-balancing loss or capacity limits.
- A common setup is for EP and DP to share the same group of GPUs.

Q9: ***How do you choose the combination of DP, TP and PP in 3D parallelism?***

- **The principle is to put the most communication-heavy parallelism on the highest-bandwidth links: TP within a node, PP across nodes, and DP on the outermost level to scale throughput.**
- TP has an All-Reduce in every layer, so it stays on NVLink and is usually ≤8; PP only passes boundary activations and suits cross-node links.
- Rule of thumb: find the smallest TP×PP that fits the model, add CP for long sequences and EP for MoE, give the remaining GPUs to DP, then tune the micro-batches to shrink the bubble and check MFU.
- DP is usually paired with ZeRO-1, because ZeRO-2/3 conflict with PP's micro-batches and add communication.
- For small and medium models (≲30B), FSDP/ZeRO-3 + activation checkpointing is often enough.

Q10: ***What is the principle and cost of activation checkpointing?***

- **During the forward pass only a few checkpoints (e.g. the input of each block) are kept and the other activations are discarded; during backward the forward pass is rerun to recover them, trading compute for memory. Full recomputation costs about one extra forward pass, roughly 33% more total compute.**
- The classic $\sqrt{n}$ scheme places a checkpoint every $\sqrt{n}$ layers, cutting memory to $O(\sqrt{n})$.
- Selective recomputation only recomputes parts that use a lot of memory but are cheap to compute (e.g. softmax, dropout), costing only a few percent.
- Other ways to save activations: FlashAttention, sequence parallelism, offloading activations to CPU.

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for block in self.blocks:
        x = checkpoint(block, x, use_reentrant=False)
    return x
```

Q11: ***What is gradient accumulation? What pitfalls should you watch for?***

- **Gradient accumulation splits a large batch into k micro-batches that run forward and backward in turn, summing gradients and updating only after k steps. It trades time for memory, and the effective global batch is micro-batch × k × DP size.**
- In DDP, use `no_sync()` for the first k−1 steps to avoid an All-Reduce every step.
- Loss normalization: with variable-length sequences, "average per micro-batch, then average" is not the same as per-token averaging over the big batch; normalize by the global count of valid tokens.
- Layers that depend on batch statistics, like BatchNorm, are not equivalent (LLMs use LayerNorm/RMSNorm, so no issue).
- The learning-rate schedule should count optimizer update steps, not micro-steps.

## Numerics and Stability

Q12: ***How do FP16, BF16 and FP8 differ? Why does LLM training prefer BF16?***

- **They differ in how bits are split between exponent and mantissa: FP16 has more precision but a small range and overflows easily; BF16 has the same exponent width as FP32, giving a large range with lower precision. LLM training prefers BF16 because it almost never overflows and needs no loss scaling, so training is more stable.**

| Format | Sign/exponent/mantissa | Notes |
|---|---|---|
| FP16 | 1/5/10 | Small range (max 65504), prone to overflow/underflow |
| BF16 | 1/8/7 | Same range as FP32, usually no loss scaling |
| FP8 E4M3 | 1/4/3 | Relatively more precision, used for forward weights and activations |
| FP8 E5M2 | 1/5/2 | Larger range, used for backward gradients |

- FP8 training (H100 onward) uses FP8 only for matrix multiplies; master weights, optimizer states and sensitive ops like LayerNorm and softmax stay in higher precision.
- FP8 needs scaling factors: Transformer Engine uses per-tensor delayed scaling, DeepSeek-V3 uses fine-grained block quantization, and Blackwell supports MXFP8/MXFP4 microscaling formats.

Q13: ***What is loss scaling? Why does FP16 need it while BF16 usually does not?***

- **FP16 has a small range, so many small gradients underflow to 0. Loss scaling multiplies the loss by a large number S before backward so the gradients land in the representable range, then divides them back before the update. BF16 has the same range as FP32 and does not underflow, so it usually does not need this.**
- Dynamic loss scaling: on inf/NaN, skip the step and halve S; after many steps (e.g. 2000) without overflow, double S.
- Gradient clipping must happen after unscaling.

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

- **After a spike, the usual fix is to roll back to a checkpoint before it, skip the offending data batches and retrain, temporarily lowering the learning rate if needed; prevention comes from the learning rate, the optimizer, normalization and precision.**
- Common causes: too high a learning rate or bad data; attention logits growing so large that softmax saturates; output-logit drift; Adam's second moment reacting too slowly to sudden large gradients; low-precision overflow.
- Prevention: gradient clipping (norm 1.0), enough warmup, Adam $\beta_2 = 0.95$ so the second moment tracks gradient changes faster.
- Architecturally: QK-Norm bounds attention logits, z-loss prevents output-logit drift, Pre-LN, BF16 instead of FP16, plus data cleaning.
- Monitor the loss, gradient norm, per-layer activation/weight norms and the maximum attention logit.

Q15: ***Why use a warmup + cosine learning-rate schedule? What does gradient clipping do?***

- **Early in training the parameters are random and Adam's second-moment estimates are unreliable, so a large learning rate easily diverges; warmup therefore ramps the rate linearly up to its peak. Cosine then decays it smoothly to about 10% of the peak, which helps the model converge well late in training. Gradient clipping caps the global gradient norm so that a few huge gradients cannot cause drastic parameter changes.**
- The downside of cosine is that the total number of steps must be fixed in advance, which makes continued training awkward.
- WSD (Warmup-Stable-Decay) holds a constant rate after warmup and decays quickly at the very end, making it easy to continue from intermediate checkpoints or run scaling experiments.
- The clipping threshold is usually 1.0; in distributed training the global norm must first be computed across GPUs. The gradient norm itself is also an important health signal.
- Key formula: $g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|_2}\right)$

Q16: ***Briefly introduce μP (Maximal Update Parametrization).***

- **μP is a parametrization that scales the initialization and per-layer learning rates with model width so that each layer's feature updates stay the same size regardless of width; hyperparameters tuned on a small model can then be transferred directly to a large one, avoiding hyperparameter search at scale.**
- Background: under standard parametrization, the optimal learning rate and initialization drift with width.
- Typical rules under Adam: hidden-layer learning rate inversely proportional to width, output logits multiplied by a 1/width factor, attention scaled by $1/d$ rather than $1/\sqrt{d}$.
- Limitations: it mainly targets width; depth needs extra handling (e.g. depth-μP), and batch size or training length do not transfer directly.

## Architecture Details

Q17: ***What is the difference between Pre-LN and Post-LN? Why is Pre-LN + RMSNorm the mainstream today?***

- **Post-LN normalizes after the residual addition, while Pre-LN normalizes before entering the sublayer. In Pre-LN the residual path is an identity map, so gradients flow straight back and training is more stable and less dependent on warmup, which is why it became mainstream; RMSNorm drops the mean subtraction and the bias, saving compute with comparable quality.**
- Post-LN is unstable for deep models and heavily depends on warmup, but may end up slightly better when training succeeds.
- Pre-LN's drawback is that the residual-stream norm grows with depth, so deeper layers contribute relatively less, and a final LN is needed.
- Variants: DeepNorm lets Post-LN train to a thousand layers; Sandwich-LN normalizes both before and after the sublayer (e.g. Gemma 2/3).
- Key formula: Pre-LN is $x_{l+1} = x_l + F(\text{LN}(x_l))$, Post-LN is $x_{l+1} = \text{LN}(x_l + F(x_l))$.

Q18: ***What is SwiGLU? Why is the FFN hidden size often $\frac{8}{3}d$?***

- **SwiGLU is a gated FFN: one branch goes through a Swish activation as the gate, the other is a linear value, and the two are multiplied element-wise and projected back; at equal compute it works better than a ReLU FFN. Because it has 3 matrices instead of 2, the hidden size is set to 2/3 of 4d, i.e. 8d/3, to keep the parameter count equal to the standard 4d FFN.**
- In practice the size is rounded to a multiple of 64 or 256.
- PaLM, LLaMA and most modern LLMs use SwiGLU.
- Key formula: $\text{SwiGLU}(x) = \left(\text{Swish}(xW) \odot xV\right)W_2$

Q19: ***What is the core idea of RoPE? Why does it encode relative position?***

- **RoPE groups the dimensions of q and k into pairs and rotates each pair by an angle that depends on the position. When two rotated vectors are dotted, the result depends only on the difference of their rotation angles, i.e. their relative distance, so relative positional encoding is achieved in an absolute-position way.**
- Different dimensions rotate at different frequencies: high-frequency dimensions encode local position, low-frequency ones encode long-range position.
- Advantages: no extra parameters, applied directly to Q/K, compatible with FlashAttention, with some long-range decay.
- RoPE applies only to Q/K, not V; combining it with DeepSeek's MLA requires a separate slice of dimensions for RoPE.
- Key formula: $\langle R_m q, R_n k\rangle = q^\top R_{n-m} k$

Q20: ***How do you extend a RoPE model's context from $L$ to $L' = sL$? How do PI, NTK-aware and YaRN differ?***

- **Naive extrapolation makes the low-frequency dimensions see rotation angles never seen in training, and quality collapses. PI compresses all positions proportionally; NTK-aware enlarges the RoPE base so high-frequency dimensions stay nearly unchanged and only low-frequency ones are interpolated; YaRN interpolates per dimension band and adds attention temperature scaling, working best.**
- PI is simple but also compresses high frequencies, making neighboring tokens hard to tell apart; it needs a little fine-tuning.
- NTK-aware works reasonably even without fine-tuning; NTK-by-parts decides per dimension, by its wavelength, whether to interpolate.
- YaRN builds on NTK-by-parts and compensates the change in attention entropy on long sequences; a small amount of data extends it to 64K–128K.
- In practice it is "raise the base (e.g. 500000 in LLaMA 3) or YaRN + continued pre-training on long text", evaluated with needle-in-a-haystack and RULER.

Q21: ***What are top-k routing, the load-balancing loss and the capacity factor in MoE?***

- **Top-k routing means the router picks the k highest-scoring experts for each token and takes a weighted sum; the load-balancing loss encourages tokens to spread evenly across experts so a few experts are not overloaded while others never learn; the capacity factor caps how many tokens each expert can process, and overflow tokens are dropped and only take the residual path.**
- Examples: Switch uses k=1, Mixtral picks 2 of 8, DeepSeek-V3 picks 8 of 256 fine-grained experts plus 1 shared expert.
- A larger capacity factor drops fewer tokens but wastes compute; 1.0–1.25 is common, and inference can be dropless.
- Other tricks: router z-loss stabilizes routing; DeepSeek-V3 does auxiliary-loss-free load balancing with a dynamic bias that only affects selection.
- Key formula: $\mathcal{L}_{\text{aux}} = \alpha\, E \sum_{i} f_i\, P_i$ ($f_i$ is the fraction of tokens sent to expert $i$, $P_i$ its mean routing probability)

## Parameter-Efficient Fine-Tuning

Q22: ***How does LoRA work? How are the rank, initialization and merging handled?***

- **LoRA assumes the fine-tuning weight update is low-rank: it freezes the original weights and trains only two small matrices B and A whose product is added to the weights as the update. The rank r is usually 8–64; B is initialized to 0 so the model starts out identical to the original; before inference the update is merged into the weights, adding no latency.**
- Trainable parameters drop from $dk$ to $r(d+k)$, so memory and checkpoints are small, and many LoRAs can share one base model for serving.
- The scaling factor $\alpha/r$ means changing r does not require retuning the learning rate (rsLoRA suggests $\alpha/\sqrt{r}$).
- In practice, applying it to all linear layers works better than only Q and V.
- Variants: DoRA, LoRA+, PiSSA; when the target is far from the pre-training distribution, a low-rank update may not be enough.
- Key formula: $W = W_0 + \frac{\alpha}{r} BA$

Q23: ***What does QLoRA improve? What are NF4, double quantization and paged optimizers?***

- **QLoRA quantizes the frozen base model to 4 bits and trains only BF16 LoRA adapters on top, letting a 65B model be fine-tuned on a single 48GB GPU. Its three key techniques are the NF4 format, double quantization that re-quantizes the scaling constants, and paged optimizers that prevent memory spikes.**
- NF4: weights are roughly normally distributed, so the 16 quantization levels are quantiles of the normal distribution, putting an equal number of weights in each bin; this is optimal for normal data.
- Double quantization: the FP32 scaling constant shared by every 64 parameters is itself quantized to 8 bits, saving about 0.37 bits per parameter on average.
- Paged optimizers: when GPU memory runs short, optimizer states are paged to CPU memory automatically, avoiding OOM.
- NF4 weights must be dequantized to BF16 for computation, so it is slower than plain LoRA; the saving is memory. Merging is usually done into a BF16 base.

Q24: ***How do Adapters, Prefix Tuning, Prompt Tuning and LoRA differ?***

- **All of them train only a few parameters; they differ in where they intervene: Adapters insert small MLPs inside layers, Prefix/Prompt Tuning prepend trainable vectors to attention or the input, and LoRA adds a low-rank update to the weights. LoRA can be merged into the weights with no inference overhead and is the current default.**

| Method | What it does | Inference overhead |
|---|---|---|
| Adapter | Inserts a bottleneck MLP + residual in each layer | Extra serial layer, adds latency |
| Prefix Tuning | Prepends trainable prefixes to each layer's K/V | Uses context length, somewhat unstable training |
| Prompt Tuning | Adds soft prompts only before the input | Fewest parameters, near full fine-tuning only for large models |
| LoRA | Adds a low-rank update to weights | Mergeable, no extra latency |

- Unified view: all of these can be seen as adding a correction term to the hidden states; they differ in where it is added and whether it is serial or parallel.
- Choice: start with LoRA/QLoRA; consider full fine-tuning when data is plentiful and the domain gap is large.

## Data and Scaling

Q25: ***What are the key design points of an LLM tokenizer? What is byte-level BPE?***

- **BPE starts from basic symbols and repeatedly merges the most frequent adjacent pair until the vocabulary reaches its target size. Byte-level BPE uses the 256 bytes as the base vocabulary, so any text can be represented and there is no OOV; the cost is that multi-byte characters such as Chinese may be split into incomplete byte fragments.**
- Vocabulary size: a large vocabulary compresses better and gives shorter sequences, but has more embedding parameters and undertrained rare tokens; the trend is upward (about 128K in LLaMA 3).
- Pre-tokenization rules: split whitespace and punctuation with a regex; splitting numbers into single digits helps arithmetic.
- You also balance coverage of multiple languages and code, and design special tokens for chat templates and tool calls.
- Common problems: spelling/counting tasks are hard, "glitch tokens", unequal cost across languages.

Q26: ***How do you deduplicate, quality-filter and mix pre-training data?***

- **Deduplication uses exact hashing plus MinHash + LSH fuzzy dedup; quality filtering combines heuristic rules, language identification and quality classifiers; mixing blends domains like web, code, math and books, setting the ratios via small-model experiments or learned methods and upsampling high-quality data.**
- Dedup reduces memorization and privacy leaks, improves training efficiency and lowers test-set contamination.
- Quality classifiers use high-quality corpora as positives; FineWeb-Edu has an LLM score "educational value" and distills it into a small classifier. Also filter toxicity, scrub PII and decontaminate against benchmarks.
- Mixing methods include small-model ablations, DoReMi and RegMix; repeating high-quality data up to about 4 epochs costs little.
- Anneal on high-quality data at the end of training; code and math data help reasoning.

Q27: ***How do the Kaplan and Chinchilla scaling laws differ? What does "about 20 tokens per parameter" mean?***

- **Both find that loss falls as a power law in parameters and data, but Kaplan concluded that extra compute should go mainly into a bigger model, whereas Chinchilla found that parameters and data should scale in equal proportion, with the compute-optimal point at about 20 training tokens per parameter.**
- Example: the 70B Chinchilla trained on 1.4T tokens beat the 280B Gopher.
- Kaplan's bias came mainly from using a fixed-length learning-rate schedule for all models and from experimenting mostly with small models.
- Chinchilla-optimal only counts training compute, not inference cost; modern models "over-train" small models to make inference cheap, e.g. LLaMA 3 8B was trained on about 15T tokens.
- Key formula: $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$, optimized under the constraint $C\approx 6ND$.

Q28: ***Are "emergent abilities" of LLMs real?***

- **It is debated. Wei et al. observed some abilities appearing suddenly once models pass a certain scale; but Schaeffer et al. argued this is largely an artifact of the metric: with discontinuous metrics like exact match it looks like a jump, while with continuous metrics performance actually improves smoothly with scale.**
- Mechanism: per-token accuracy improves smoothly, but requiring the whole sequence to be correct turns it into a jump; few evaluation samples for small models can also create apparent jumps.
- Middle ground: users care whether a task succeeds, so the "jump in usability" is real.
- Plotting against pre-training loss instead of parameter count makes the appearance of abilities more predictable.
- Interview point: explain that "metric choice affects the conclusion", and why predictability matters for safety and planning.
