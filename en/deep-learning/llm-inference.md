# LLM Inference Optimization

## Inference Phases and Performance Bottlenecks

Q1: ***What are the prefill and decode phases of LLM inference, and what characterizes each?***

- **Prefill**: processes the whole prompt in one pass; all input tokens go through the forward pass **in parallel**, the KV cache for every layer is computed and written, and the first token is produced.
    - Essentially large matrix multiplications (GEMM); FLOPs ≈ $2 \cdot N_{params} \cdot n_{prompt}$ (excluding the $O(n^2)$ attention term). Usually **compute-bound**.
    - Determines **time to first token (TTFT)**; for long prompts the quadratic attention cost becomes significant.
- **Decode**: autoregressively generates **1 token** per step, reading all weights and the historical KV cache and appending new K/V.
    - Essentially matrix-vector products (GEMV, or skinny GEMMs at small batch); very little computation per weight read. Usually **memory-bandwidth-bound**.
    - Determines **per-token latency (TPOT/ITL)**; the longer the output, the larger the share of total time spent in decode.
- Follow-up: because the two phases behave so differently, scheduling optimizations such as **chunked prefill** and **disaggregated prefill/decode** exist (see below).

Q2: ***What do compute-bound and memory-bandwidth-bound mean? How do you use arithmetic intensity to tell them apart?***

- **Arithmetic intensity** $I = \frac{\text{FLOPs}}{\text{bytes moved}}$.
- **Roofline model**: attainable performance $= \min(\text{peak compute},\ I \times \text{memory bandwidth})$. Ridge point $I^* = \frac{\text{peak compute}}{\text{bandwidth}}$.
    - $I < I^*$: memory-bound; $I > I^*$: compute-bound.
    - For an H100 SXM, dense BF16 is about 989 TFLOPS and HBM bandwidth about 3.35 TB/s, so $I^* \approx 300$ FLOPs/byte.
- **Linear layers in decode**: weight $W \in \mathbb{R}^{d \times d}$ (FP16, $2d^2$ bytes); with batch $B$ the FLOPs are $2Bd^2$, so $I \approx B$. $B$ must reach a few hundred to saturate compute → this is why **increasing batch size raises decode throughput almost "for free"**.
- **Attention in decode**: each request's KV cache is used only by its own single query, so $I \approx 1$ (about the group size $g$ under GQA); batching cannot amortize it, so it stays memory-bound.
- **Lower bound on single-stream decode latency**: $t_{token} \gtrsim \frac{\text{weight bytes} + \text{KV bytes}}{\text{bandwidth}}$. Example: a 7B FP16 model is about 14 GB; at 3.35 TB/s that is about 4 ms/token, i.e. a single-stream ceiling of roughly 240 tokens/s.

Q3: ***What latency/throughput metrics are common for LLM serving, and how do they trade off?***

- **TTFT (Time To First Token)**: from request arrival to the first returned token, including queueing + prefill. Affects interactive experience.
- **TPOT (Time Per Output Token) / ITL (Inter-Token Latency)**: the interval between consecutive output tokens; determines "typing speed". TPOT usually refers to the mean, ITL focuses on the distribution (P99 jitter).
- **End-to-end latency** $\approx \text{TTFT} + \text{TPOT} \times (n_{out}-1)$.
- **Throughput**: tokens generated per second by the system (or requests per second).
- **Goodput**: throughput of requests that meet the SLO (e.g. TTFT < 1 s, TPOT < 50 ms); closer to real production goals than raw throughput.
- Trade-offs:
    - Larger batch → higher throughput, but per-step compute time, queueing and interference grow, so TPOT increases.
    - Inserting prefills into decode batches → lower TTFT for new requests, but requests already decoding see stalls (ITL spikes).
    - Quantization / speculative decoding → lower latency, but may affect quality or lose benefit under high load.
- Follow-up: offline batch jobs optimize throughput/cost, online chat optimizes P99 TTFT and TPOT, and multi-turn agent workloads care about prefix-cache hit rate.

## KV Cache and Memory Optimization

Q4: ***Why is a KV cache needed? How do you compute its memory footprint?***

- During autoregressive generation, the K and V of past tokens do not change in later steps. Caching them reduces per-step attention from recomputing the whole prefix, $O(n^2)$, to computing only the new token, $O(n)$.
- Memory formula:
$$\text{KV bytes} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{seq} \cdot \text{bytes} \quad (\times\ \text{batch})$$
    - 2 accounts for K and V; $L$ is the number of layers; $n_{kv}$ the number of KV heads; $d_{head}$ the per-head dimension; bytes is bytes per element (2 for FP16).
- Examples:
    - Llama-2-7B (MHA, $L=32, n_{kv}=32, d_{head}=128$): per token $2\cdot32\cdot32\cdot128\cdot2 = 512$ KB; a single 4K-context sequence is about 2 GB.
    - Llama-3-8B (GQA, $n_{kv}=8$): 128 KB per token, 1/4 of the above.
    - Llama-3-70B ($L=80, n_{kv}=8$): 320 KB per token; a single 128K-context sequence is about 40 GB.
- Conclusion: with long contexts and large batches, **the KV cache often exceeds the weight memory** and becomes the main limit on concurrency.
- Optimization directions: fewer heads (MQA/GQA/MLA), lower precision (KV quantization), shorter length (sliding window/sparsity/eviction), less fragmentation (PagedAttention), reuse (prefix caching).

Q5: ***How do MQA, GQA and MLA shrink the KV cache? What does each cost?***

- **MHA**: $n_{kv} = n_h$; every query head has its own K/V.
- **MQA (Multi-Query Attention, 2019)**: all query heads share **a single** K/V, $n_{kv}=1$, shrinking the KV cache by $n_h\times$; the cost is some quality loss and possible training instability; under TP the KV head must be replicated.
- **GQA (Grouped-Query Attention, 2023)**: query heads are split into $G$ groups, each sharing one K/V, $n_{kv}=G$ (e.g. $G=8$ in Llama-3). Quality close to MHA, memory close to MQA; the current mainstream choice. It can be obtained from an MHA checkpoint by **mean-pooling the KV heads + a small amount of continued training (uptraining)**.
- **MLA (Multi-head Latent Attention, DeepSeek-V2, 2024)**:
    - Compresses K/V low-rank into a latent vector $c_t^{KV} = W^{DKV} h_t$ (dimension $d_c$, e.g. 512); only $c_t^{KV}$ plus a decoupled RoPE key (e.g. 64-dim) are cached.
    - At inference the up-projection $W^{UK}$ is absorbed into $W^Q$ and $W^{UV}$ into $W^O$, so K/V never need to be explicitly reconstructed.
    - Because RoPE is position-dependent and cannot be absorbed, a separate shared small-dimension RoPE key is kept (decoupled RoPE).
    - About $d_c + d_h^R = 576$ elements are cached per token per layer, versus $2 n_h d_h$ for MHA (32768 in DeepSeek-V2), a compression of tens of times, and the paper reports quality no worse than MHA.
- Follow-up: GQA/MQA also raise the arithmetic intensity of decode attention (one copy of KV is reused by $g$ query heads).

Q6: ***What problem does PagedAttention solve? How does vLLM manage the KV cache?***

- Problem: early systems **pre-allocated contiguous memory for the maximum length** of each request, leading to
    - **Internal fragmentation**: the actual generated length is far below the maximum;
    - **External fragmentation**: holes of varying sizes;
    - No sharing: common prefixes in parallel sampling and beam search are stored repeatedly.
    - The paper measured actual memory utilization in such systems as low as 20%–40%.
- **PagedAttention (vLLM, SOSP 2023)**: borrows **paged virtual memory** from operating systems.
    - The KV cache is split into fixed-size **blocks** (e.g. 16 tokens per block) that need not be physically contiguous;
    - Each request maintains a **block table** (logical block → physical block mapping); blocks are allocated on demand, so waste only occurs in the last block;
    - The attention kernel reads KV via indirect addressing through the block table;
    - Blocks carry **reference counts**, enabling multiple sequences to share a prefix, with **copy-on-write** on modification.
- Effect: near-zero memory waste, room for larger batches, significantly higher throughput.
- **Preemption** when memory runs out: swap some requests' blocks out to CPU, or drop them and recompute later.

Q7: ***What is continuous batching? How does it differ from static batching?***

- **Static batching**: a batch of requests starts together and ends together; slots of short requests sit idle after they finish, waiting for the longest one; new requests must wait for the whole batch to complete.
- **Continuous batching (iteration-level scheduling, Orca, OSDI 2022)**:
    - Schedules at the granularity of **each decode step**: a finished request leaves immediately and a new one joins immediately;
    - Sequences of different lengths share a batch: for linear layers all tokens are concatenated into one big matrix, while attention is computed per request (selective batching).
- Effect: much higher GPU utilization and throughput, lower TTFT; standard in vLLM, SGLang and TensorRT-LLM (in-flight batching).
- Related issues:
    - New requests' prefills compete with decode for compute, causing ITL spikes → **chunked prefill** (e.g. Sarathi-Serve) splits long prompts into chunks and forms mixed batches with decode tokens under a fixed token budget;
    - Requires KV-memory-aware admission control and preemption policies.

Q8: ***What is prefix caching (prompt caching)? How is it implemented, and where does it help?***

- Idea: many requests share the same prefix (system prompt, few-shot examples, multi-turn history, agent tool descriptions, multiple questions about one document); its KV cache can be reused, skipping that part of prefill.
- Implementation:
    - **vLLM Automatic Prefix Caching**: hashes "all prefix tokens up to this block" per block; on a hit, the existing physical block is referenced directly;
    - **SGLang RadixAttention**: organizes all cached token sequences in a **radix tree**, supports arbitrary prefix matching, with LRU eviction;
    - Caches can also be tiered into CPU memory/SSD/a distributed KV store (as in Mooncake's design).
- Benefits: large drops in TTFT and prefill compute; commercial APIs often bill cache-hit input tokens at a discount.
- Caveats:
    - Only **exactly identical prefixes** match; change one token in the middle and everything after it is invalidated → design prompts with **stable content first, variable content last** (don't put a timestamp at the start of the system prompt);
    - Scheduling can do cache-aware routing, sending requests with the same prefix to the same instance.

## Attention Computation Optimization

Q9: ***What is the core idea of FlashAttention? Why is it faster and more memory-efficient?***

- Bottleneck: standard attention writes/reads the $N\times N$ matrices $S=QK^T$ and $P=\text{softmax}(S)$ to/from HBM, with $\Theta(N^2)$ memory traffic; on GPUs it is **memory-bound**, not compute-bound.
- **IO-awareness**: GPU SRAM is fast but small, HBM is large but slow. FlashAttention aims to reduce HBM reads/writes, not FLOPs.
- **Tiling**: load blocks of $Q, K, V$ into SRAM and compute $QK^T$, softmax and multiplication by $V$ on-chip, never writing the $N\times N$ matrix back to HBM (kernel fusion).
- **Online softmax**: with tiling, the row max and sum aren't available at once, so running statistics are kept and corrected block by block:
$$m^{new} = \max(m, \tilde m),\quad \ell^{new} = e^{m-m^{new}}\ell + e^{\tilde m - m^{new}}\tilde \ell$$
$$O^{new} = e^{m-m^{new}} O + e^{\tilde m - m^{new}} \tilde P V_j$$
  The final output is $O/\ell$, **numerically exact** relative to standard attention (not an approximation).
- **Recomputation**: the backward pass does not store $P$; only the output $O$ and the per-row logsumexp are kept, and $S, P$ are recomputed in SRAM during backward. FLOPs rise slightly, but memory drops from $O(N^2)$ to $O(N)$, and it is faster overall thanks to less memory traffic.
- IO complexity: standard attention $\Theta(Nd + N^2)$; FlashAttention $\Theta(N^2 d^2 / M)$ ($M$ = SRAM size).

Q10: ***What did each FlashAttention version improve? Which attention kernels are used in decode?***

- **FlashAttention-1 (2022)**: tiling + online softmax + recomputation; parallel over batch and heads.
- **FlashAttention-2 (2023)**:
    - Fewer non-matmul FLOPs (rescaling deferred to a single final step);
    - Additional parallelism over the **sequence-length dimension**, so SMs are saturated even for long sequences with small batches;
    - Better work partitioning across warps (splitting Q across warps), less shared-memory communication; about 2x faster than FA1.
- **FlashAttention-3 (2024, targeting Hopper H100)**: uses TMA asynchronous copies and WGMMA, **warp specialization** (producer/consumer warps), ping-pong overlap of softmax and GEMM, and supports **FP8** (with block quantization and incoherent processing to reduce error).
- Later versions target newer architectures such as Blackwell, with the same philosophy: match hardware features to reduce memory traffic and overlap compute with data movement.
- **Decode phase**: query length is 1, so parallelizing over batch×heads is not enough to fill the GPU → **Flash-Decoding** **splits KV along the sequence dimension (split-K)**, computes partial results in parallel, and merges them using logsumexp.
- In practice, libraries such as **FlashInfer** provide efficient kernels for paged KV, prefix sharing and various attention variants.

## Quantization

Q11: ***What forms of quantization are used for LLM inference? How do weight-only quantization and W8A8 differ, and when is each appropriate?***

- Basic form: $x_q = \text{round}(x / s) + z$, with scale $s$ and zero point $z$; granularity can be per-tensor, per-channel or per-group (e.g. groups of 128 elements).
- **Weight-only (e.g. W4A16, W8A16)**:
    - Only weights are stored quantized; at compute time they are dequantized to FP16 and multiplied with activations;
    - Memory and **weight-read bandwidth** drop by 2–4x directly → large speedups for **memory-bound decode and small batches**;
    - FLOPs are not reduced; at large batch/prefill the dequantization overhead can cancel the gain.
    - Representatives: GPTQ, AWQ, INT4 group-wise.
- **Weight + activation (W8A8 INT8 / FP8)**:
    - Both weights and activations are quantized and INT8/FP8 Tensor Cores are used, **doubling compute** → effective for **compute-bound prefill and large-batch high-throughput serving**;
    - The difficulty is **outliers** in activations (a few channels in LLMs have huge magnitudes), making naive quantization error large.
    - Representatives: SmoothQuant, LLM.int8() (mixed decomposition with outlier channels in FP16), FP8.
- More aggressive: W4A4 (e.g. QuaRot, SpinQuant, which use rotations to remove outliers), and 1–2 bit schemes, with greater quality risk.
- Follow-up: evaluate on target tasks after quantization (perplexity, downstream accuracy, long context, reasoning/code tasks); low bit widths hurt math/code tasks more.

Q12: ***How does GPTQ work?***

- Post-training quantization (PTQ), solved layer by layer:
$$\min_{\hat W} \| W X - \hat W X \|_2^2$$
  where $X$ is the input to that layer on a small calibration set.
- Based on OBS/OBQ: after quantizing each weight, second-order information (Hessian $H = 2XX^T$) is used to **update the remaining not-yet-quantized weights** to compensate for the error:
$$\delta_F = -\frac{w_q - \text{quant}(w_q)}{[H_F^{-1}]_{qq}} (H_F^{-1})_{:,q}$$
- GPTQ's engineering improvements:
    - All rows are quantized **in the same column order**, sharing $H^{-1}$, which drastically reduces complexity;
    - **Lazy batch updates** (processing a block of columns at a time) improve GPU utilization;
    - **Cholesky decomposition** for numerically stable $H^{-1}$.
- Result: models with tens to hundreds of billions of parameters can be quantized to 3–4 bits in a few GPU hours with small accuracy loss; commonly combined with group-wise scales (e.g. group size 128) and act-order.

Q13: ***How do AWQ and SmoothQuant each deal with outliers?***

- Shared idea: use the equivalence $Y = XW = (X \operatorname{diag}(s)^{-1})(\operatorname{diag}(s) W)$ to "migrate" quantization difficulty between activations and weights.
- **SmoothQuant (W8A8)**:
    - Activation outliers concentrate in a few channels while weights are relatively smooth → "smooth" the large activation magnitudes into the weights;
    - $s_j = \frac{\max |X_j|^{\alpha}}{\max |W_j|^{1-\alpha}}$, with $\alpha$ typically around 0.5;
    - $\operatorname{diag}(s)^{-1}$ can be fused offline into the preceding LayerNorm/linear layer, adding no inference overhead.
- **AWQ (Activation-aware Weight Quantization, weight-only)**:
    - Observation: about 1% of **salient weight channels** are crucial for accuracy, and saliency is determined by **the magnitude of the corresponding activations**, not the weights themselves;
    - Rather than keeping them in mixed precision, it **scales up those channels' weights per channel** by $s$ (shrinking the activations correspondingly), reducing their relative quantization error;
    - $s = s_X^{\alpha}$, with $\alpha$ found by a small grid search minimizing output error; no backpropagation or reconstruction, so less risk of overfitting the calibration set than GPTQ.
- Comparison: SmoothQuant makes **activations** quantizable; AWQ protects the important channels of **weights**.

Q14: ***What are the advantages of FP8 inference? How is KV-cache quantization done?***

- **FP8**: two formats, **E4M3** (more precision, range about ±448, commonly used for weights/activations in the forward pass) and **E5M2** (larger range, commonly used for gradients).
    - Hopper/Ada and later GPUs natively support FP8 Tensor Cores, roughly doubling compute and halving memory versus BF16;
    - A floating-point format tolerates outliers better than INT8; per-tensor or per-channel/per-block scales usually reach near-BF16 accuracy, making it **almost the default choice for W8A8**;
    - Fine-grained (block-wise) scales reduce error further; e.g. DeepSeek-V3 used fine-grained FP8 in training.
- **KV-cache quantization**:
    - With long contexts/large batches the KV cache dominates memory and bandwidth; quantizing it to FP8/INT8 can double concurrency and halve decode attention memory traffic;
    - Needs scales (per-tensor/per-head/per-token); the attention kernel dequantizes after reading;
    - More aggressive 2–4 bit schemes (e.g. KIVI) found that **quantizing Keys per channel and Values per token** works better (Keys have fixed outlier channels);
    - Risk: long-context retrieval tasks are sensitive to KV precision and need dedicated evaluation.

## Decoding Acceleration

Q15: ***How does speculative decoding work? Why does it preserve the output distribution?***

- Motivation: decode is memory-bound; one forward pass of the large model verifying $\gamma+1$ tokens costs almost the same as generating 1 token.
- Procedure:
    1. A small, fast **draft model** $q$ autoregressively guesses $\gamma$ tokens $x_1,\dots,x_\gamma$;
    2. The **target model** $p$ computes the distributions at these positions in one parallel forward pass;
    3. Verify left to right: accept each with probability $\min\left(1, \frac{p(x)}{q(x)}\right)$;
    4. On the first rejection, sample a replacement token from the **residual distribution** $p'(x) = \frac{\max(0,\ p(x) - q(x))}{\sum_{x'} \max(0,\ p(x') - q(x'))}$ and stop; if all are accepted, sample one extra token from $p$.
- **Proof of correctness**: let $\beta = \sum_x \min(p(x), q(x))$ be the acceptance probability; then
$$P(\text{output}=x) = q(x)\min\left(1,\tfrac{p(x)}{q(x)}\right) + (1-\beta)\frac{\max(0,p(x)-q(x))}{1-\beta} = \min(p,q) + \max(0,p-q) = p(x)$$
  So the speedup is **lossless**: identical in distribution to sampling directly from the target model (under greedy decoding it reduces to "accept while the prefix matches").
- Speedup: with per-token acceptance rate $\alpha$, the expected number of tokens per round is $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$; the actual speedup also depends on the draft model's cost $c$ and the choice of $\gamma$.
- Caveats: at large batch the system is already near compute-bound, so the extra verification compute is no longer "free" and the gain shrinks; draft and target must share a tokenizer.

Q16: ***How do Medusa, EAGLE and similar methods differ from classic speculative decoding?***

- The classic method needs a separate draft model that must be kept aligned and takes memory. The improvement is to **let the target model draft for itself**:
- **Medusa (2024)**: adds several **extra decoding heads** on the target model's final hidden state; the $k$-th head directly predicts token $t+k$; **tree attention** verifies multiple candidate combinations in one pass; the backbone can be frozen and only the heads trained. It often uses "typical acceptance" to relax verification, in which case the distribution is no longer strictly preserved.
- **EAGLE (2024) family**: drafts autoregressively at the **feature level (second-to-last hidden state)**, also feeding in the token sampled at the previous step to remove uncertainty; the draft head is just one lightweight Transformer layer, with higher acceptance rates than Medusa. EAGLE-2 uses confidence-based **dynamic draft trees**; EAGLE-3 further fuses multi-layer features and drops the feature-prediction loss to scale better with data.
- Others: **Lookahead decoding** (Jacobi iteration + n-gram pool, no draft model), **Prompt lookup** (retrieves n-grams from the input; suits summarization/code editing where output overlaps heavily with input), **MTP (multi-token prediction)** heads (e.g. DeepSeek-V3's MTP module from training can serve as a drafter at inference).
- Common ground: all rely on the "parallel verification by the target model" step; schemes with strict rejection sampling preserve the distribution.

## Decoding Strategies and Constrained Generation

Q17: ***What are the common decoding strategies? How do temperature, top-k, top-p and repetition penalties work?***

- **Greedy**: take the argmax each step; highly deterministic but prone to repetition and dullness.
- **Beam search**: keeps the $k$ sequences with the highest cumulative log probability; suited to tasks with a "reference answer" such as translation/ASR; in open-ended generation it tends to produce generic, repetitive text, and needs a length penalty $\frac{\log P}{|y|^\alpha}$.
- **Temperature**: $p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$. $T<1$ is sharper/more conservative, $T>1$ flatter/more diverse, $T\to0$ is equivalent to greedy.
- **Top-k**: keep only the $k$ most probable tokens and renormalize; drawback: fixed $k$ doesn't adapt to how peaked the distribution is.
- **Top-p (nucleus)**: keep the smallest set of tokens whose cumulative probability reaches $p$; the candidate set size adapts.
- **Min-p**: keep tokens with probability $\ge p_{min} \cdot p_{max}$; more stable than top-p at high temperature.
- **Repetition penalties**:
    - CTRL-style repetition penalty: for tokens already seen, divide positive logits by $\theta$ and multiply negative logits by $\theta$ ($\theta>1$);
    - Frequency penalty: subtract from the logit linearly in the occurrence count; presence penalty: subtract a constant once a token has appeared;
    - No-repeat n-gram: forbid repeated n-grams (mostly with beam search).
- Common processing order: logits → penalties → temperature → top-k/top-p/min-p filtering → sampling.
- Follow-up: reasoning models/code tasks often use lower temperatures or vendor-recommended parameters; for reproducibility fix the seed, though factors like lack of batch invariance can still make results not fully deterministic.

Q18: ***How is structured/constrained decoding implemented? What are the pitfalls?***

- Goal: guarantee output strictly conforms to a JSON Schema, regex or context-free grammar (CFG), e.g. function-call arguments.
- Implementation: at each step compute the **set of legal tokens** from the current grammar state, set illegal tokens' logits to $-\infty$ (logit mask), then sample.
    - **Regex/JSON Schema → finite-state machine (FSM)**: precompute the legal tokens for each state (e.g. Outlines);
    - **CFG → pushdown automaton (PDA)**: supports nested structures; XGrammar and others precompute most masks by separating "context-independent tokens", and overlap with GPU compute, so overhead is small;
    - **Jump-forward** (SGLang): when the following characters are uniquely determined by the grammar (e.g. `{"name": "`), skip token-by-token generation and emit them directly.
- Pitfalls:
    - **Token boundaries don't align with the grammar**: one token may span several grammar symbols, so matching must happen at the byte/character level;
    - **Distribution distortion**: forced masking changes the model's distribution; if the model "wants" to output something else, the result may be syntactically valid but semantically poor; mitigate by giving the schema/examples in the prompt;
    - Reasoning models can "think" freely first and only then enter a constrained final-answer segment;
    - If mask computation runs synchronously on the CPU, it slows large-batch decode.

## Parallelism, MoE and System Architecture

Q19: ***How does tensor parallelism (TP) work for inference? How do you choose between it and pipeline parallelism (PP)?***

- **TP (Megatron-style)**:
    - MLP: the first weight is **split by columns** (each GPU computes part of the intermediate activations independently; GeLU/SwiGLU is elementwise and done locally), the second is **split by rows**, followed by one **all-reduce** to sum;
    - Attention: split by **heads**, each GPU handles some Q heads and the corresponding KV heads; the output projection is split by rows, followed by another all-reduce;
    - About 2 all-reduces per layer, so communication is frequent → suited to **intra-node NVLink**.
- Benefit: weights and KV cache are spread across GPUs, **each GPU reads fewer bytes → lower single-request decode latency**, which is TP's core value in inference compared with other parallelism.
- Limits: $n_{kv}$ should be divisible by the TP degree, otherwise KV heads are replicated (e.g. 8 GQA KV heads at TP=16 need replication); communication share grows with very large TP degrees.
- **PP**: splits layers across devices; a single request passes through all stages in sequence, so it **does not reduce single-request latency**, but communication is small and suits cross-node setups; multiple micro-batches are needed to fill the pipeline.
- Others: **data parallelism (replicas)** scales throughput; **expert parallelism (EP)** for MoE; **context/sequence parallelism** for prefill of very long prompts.

Q20: ***What are the characteristics and challenges of MoE model inference?***

- Characteristics: each token is processed only by the top-k experts chosen by the router, so **total parameters are large while activated parameters are small** (e.g. DeepSeek-V3 has 671B total and 37B activated per token).
    - **Compute** scales with activated parameters; per-token FLOPs are low;
    - **Memory** scales with total parameters; all experts must be resident.
- Challenges:
    - **Decode memory traffic**: at small batch each expert gets very few tokens, so arithmetic intensity is even lower; as batch grows, more distinct experts are activated and more weights are read → MoE needs larger batches to realize its advantage;
    - **Load imbalance**: hot experts become bottlenecks. Training uses auxiliary load-balancing losses (or DeepSeek-V3's auxiliary-loss-free strategy that adjusts routing via bias terms); at inference, hot experts can be **replicated (redundant experts)** and dynamically rearranged;
    - **Expert parallelism (EP)**: experts live on different GPUs; tokens are dispatched and combined via **all-to-all**, which is communication-heavy and needs efficient communication libraries and compute-communication overlap (e.g. interleaving two micro-batches);
    - Kernels: tokens routed to the same expert are gathered and computed in batches with **grouped GEMM**.
- Common deployment: TP/DP for attention, EP for MoE layers; large deployments combine this with prefill/decode disaggregation.

Q21: ***What is disaggregated prefill/decode serving, and why do it?***

- Problem: running prefill and decode on the same GPU causes **mutual interference** — long prefills block decode and cause ITL spikes; the two phases also have different optimal parallelism strategies, batch sizes and hardware configurations.
- **Disaggregation** (e.g. DistServe, Splitwise, Mooncake, proposed around 2024):
    - **Prefill instances**: handle prompts only, optimize TTFT, compute-heavy, can use high-compute configurations;
    - **Decode instances**: handle generation only, optimize TPOT and large-batch throughput, memory-heavy, can use different parallelism degrees;
    - After prefill, the **KV cache is transferred over high-speed interconnect (NVLink/RDMA)** to the decode instance.
- Benefits: the two phases scale and meet SLOs independently, improving goodput; can be combined with a distributed KV cache pool for prefix reuse.
- Costs: KV transfer bandwidth and latency, higher system complexity; limited gain for short-prompt workloads.
- Alternative without disaggregation: **chunked prefill** mixed batches that balance TTFT and ITL within a single instance.

Q22: ***What techniques optimize long-context inference?***

- Problems: prefill attention costs $O(n^2)$ compute; each decode step reads $O(n)$ KV, and KV memory grows linearly with length.
- **Sliding window attention (SWA)**: each token attends only to the last $w$ tokens (e.g. $w=4096$ in Mistral 7B); the KV cache uses a fixed-size **rolling buffer**; stacked layers give a receptive field of about $L \times w$. Often interleaved with global attention layers (some layers global, some local) to retain long-range ability.
- **Attention sinks (StreamingLLM)**: the first few tokens absorb a large share of attention; keeping these "sink" tokens + a recent window enables stable streaming generation (though it doesn't truly remember the middle content).
- **Sparse attention**:
    - Fixed patterns: local + global + random (Longformer, BigBird);
    - Trainable/dynamic sparsity: e.g. DeepSeek's NSA (three branches: compression, selection, sliding window) and DeepSeek Sparse Attention (a lightweight indexer selects the top-k past tokens for each query).
- **KV eviction/compression**: H2O (keeps "heavy hitter" tokens with high cumulative attention), SnapKV (selects important KV based on an observation window at the end of the prompt), KV quantization, MLA, etc.
- **System level**: context parallelism/Ring Attention to split very long prefills; chunked prefill; offloading KV to CPU.
- Also watch position-encoding extrapolation (e.g. RoPE interpolation, YaRN) and long-context evaluations like "needle in a haystack" — don't look only at speed and ignore quality.

Q23: ***What are the mainstream LLM serving frameworks and their characteristics?***

- **vLLM**: originated with PagedAttention and continuous batching; broad model and hardware coverage, active community; supports prefix caching, speculative decoding, quantization, TP/PP/EP, structured outputs and an OpenAI-compatible API; a common default for general deployment.
- **SGLang**: RadixAttention prefix caching, efficient structured outputs and constrained decoding, a frontend programming language (good for multi-call and agent-style programs); significant investment in large-scale MoE deployment (EP, PD disaggregation); very strong performance.
- **TensorRT-LLM**: NVIDIA's official framework; builds/compiles optimized engines ahead of time, deep kernel fusion, good FP8/FP4 low-precision support, in-flight batching; top performance on NVIDIA hardware but less flexible and harder to get started with; often paired with the Triton inference server.
- Others: **llama.cpp/Ollama** (GGUF quantization, CPU/consumer GPU/on-device), **LMDeploy**, cloud vendors' in-house engines, etc.
- Selection criteria: hardware platform, model support (especially new architectures), workload characteristics (long shared prefixes, structured output, high concurrency), quantization needs, operational cost; **benchmark with your own traffic distribution** rather than relying only on public benchmarks.

Q24: ***How do you estimate the resources and cost of serving an LLM?***

- **Memory estimate**:
    - Weights: $N_{params} \times \text{bytes}$ (e.g. 70B is about 140 GB in BF16, 70 GB in FP8, 35 GB in INT4);
    - KV cache: the Q4 formula times concurrency × average context length;
    - Reserve extra for activations, CUDA graphs, communication buffers, etc. (about 10%–20%).
    - Example: Llama-3-70B with FP8 weights is about 70 GB with 320 KB of KV per token; to support 64 concurrent requests × 8K context, KV is about $64 \times 8192 \times 320\text{KB} \approx 168$ GB, requiring multi-GPU TP.
- **Latency estimate**:
    - Decode: $\text{TPOT} \approx \frac{\text{weight bytes}/TP + \text{KV bytes in batch}/TP}{\text{bandwidth} \times \text{utilization}}$ + communication overhead;
    - Prefill: $\text{TTFT} \approx \frac{2 N_{params} n_{prompt}}{\text{compute} \times \text{MFU}}$ + queueing.
- **Cost estimate**:
$$\text{cost per million tokens} = \frac{\text{GPU hourly price} \times \text{number of GPUs}}{\text{throughput (tokens/s)} \times 3600} \times 10^6$$
    - Example (assumed numbers): one GPU at 2 USD/hour with sustained throughput of 2000 tokens/s gives about $\frac{2}{2000 \times 3600}\times10^6 \approx 0.28$ USD per million tokens.
    - Count input and output tokens separately: prefill tokens are cheap (parallel, compute-bound), decode tokens are expensive (memory-bound), which is why APIs usually price output far higher than input; cache-hit input is cheaper still.
- Cost levers: higher utilization (batching, load balancing, autoscaling), quantization, prefix caching, speculative decoding, MoE/small-model routing (send easy requests to small models), batch APIs for offline jobs.

Q25: ***Your production LLM service has high latency. How would you systematically diagnose and optimize it?***

- **First break down the metrics**: is TTFT high or TPOT high? Is P50 high or only P99?
    - High TTFT: queueing (insufficient capacity/unfair scheduling), overly long prompts, prefill being blocked → scale out, prefix caching, chunked prefill, PD disaggregation, cap maximum input length;
    - High TPOT: batch too large, KV memory shortage causing preemption/recomputation, large attention share from long contexts → tune max batch/token budget, KV quantization, GQA models, higher TP degree, speculative decoding;
    - P99 jitter: long prefills cutting in, preemption, GC/CPU bottlenecks (tokenizer, sampling, structured-decoding masks on CPU).
- **Then look at resources**: GPU utilization, memory (KV cache occupancy, preemption count), bandwidth utilization, communication share (profiling, e.g. Nsight).
- **Model side**: smaller/distilled models, quantization (weight-only for low concurrency, FP8 for high throughput), MoE.
- **System side**: CUDA graphs to cut kernel-launch overhead, kernel fusion, FlashAttention/FlashInfer, cache-aware routing.
- **Application side**: streaming output to reduce perceived latency, cap output length, trim prompts and put stable prefixes first, parallelize independent calls.
- Validate every optimization with **quality evaluation + real traffic replay**, so speed isn't bought at the expense of quality.
