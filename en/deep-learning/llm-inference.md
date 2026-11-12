# LLM Inference Optimization

## Inference Phases and Performance Bottlenecks

Q1: ***What are the prefill and decode phases of LLM inference, and what characterizes each?***

- **Prefill processes the whole prompt in parallel in one pass, writes the KV cache and emits the first token, so it is compute-bound. Decode generates only one token per step but must read all the weights and the KV cache each time, so it is memory-bandwidth-bound.**
- Prefill is essentially a large matrix multiply (GEMM) and determines time to first token (TTFT); with long prompts the quadratic attention cost becomes significant.
- Decode is essentially a matrix-vector multiply that does very little compute per weight read; it determines time per output token (TPOT), and the longer the output, the larger decode's share of total time.
- Because the two phases behave so differently, schedulers use tricks such as chunked prefill and disaggregated prefill/decode serving.

Q2: ***What do compute-bound and memory-bandwidth-bound mean? How do you use arithmetic intensity to tell them apart?***

- **Arithmetic intensity is "how many operations you do per byte moved". If it is below the hardware's compute-to-bandwidth ratio, memory bandwidth is the bottleneck; if it is above, compute is the bottleneck.**
- Roofline model: achievable performance is the smaller of peak compute and intensity times bandwidth. On an H100 the ridge point is about 300 FLOPs/Byte.
- The arithmetic intensity of decode's linear layers is roughly the batch size, so increasing the batch raises throughput almost "for free" until it reaches a few hundred.
- In decode attention each request's KV is used only by its own query, so intensity is about 1; batching cannot amortize it and it stays memory-bound.
- A lower bound on single-stream decode latency is roughly (weight + KV bytes) / bandwidth; for example a 7B FP16 model is about 14 GB, about 4 ms/token on an H100.
- Key formula: $I = \frac{\text{FLOPs}}{\text{bytes moved}}$

Q3: ***What latency/throughput metrics are common for LLM serving, and how do they trade off?***

- **The core metrics are TTFT (time to first token, including queueing + prefill), TPOT/ITL (the gap between consecutive tokens, i.e. "typing speed"), throughput (tokens generated per second), and goodput, which counts only requests that meet the SLO.**
- End-to-end latency is roughly TTFT plus TPOT times the output length.
- A larger batch gives higher throughput but increases each request's TPOT and queueing time.
- Inserting new requests' prefill into decode batches lowers TTFT but causes stalls (ITL spikes) for requests that are already generating.
- Priorities depend on the scenario: offline batch jobs care about throughput and cost, online chat about P99 TTFT/TPOT, and multi-turn agents about prefix-cache hit rate.

## KV Cache and Memory Optimization

Q4: ***Why is a KV cache needed? How do you compute its memory footprint?***

- **During generation the K and V of past tokens never change, so caching them means each step only computes the new token instead of recomputing the whole prefix; the price is memory that grows linearly with layers, KV heads, sequence length and batch.**
- Example: Llama-2-7B (MHA) takes about 512 KB per token, about 2 GB for one 4K-context sequence; Llama-3-8B with GQA needs only 128 KB per token.
- With long contexts and large batches the KV cache is often larger than the weights and becomes the main limit on concurrency.
- Optimization directions: fewer KV heads (MQA/GQA/MLA), lower precision (KV quantization), shorter length (sliding window/eviction), less fragmentation (PagedAttention), and reuse (prefix caching).
- Key formula: $\text{KV bytes} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{seq} \cdot \text{bytes} \cdot \text{batch}$ (the 2 is for K and V)

Q5: ***How do MQA, GQA and MLA shrink the KV cache? What does each cost?***

- **All three reduce the amount of K/V that must be cached: MQA makes all query heads share one set of K/V, GQA makes each group of query heads share one set, and MLA compresses K/V into a low-dimensional latent vector and caches only that.**
- MQA: shrinks the KV cache to $1/n_h$ of its size, but quality drops somewhat and training can be unstable.
- GQA: quality close to MHA with memory close to MQA, the current mainstream (e.g. Llama-3 uses 8 groups); it can be obtained from an MHA checkpoint by mean-pooling KV heads and briefly continuing training.
- MLA (DeepSeek-V2): caches only the low-rank latent plus a small decoupled RoPE key, and at inference absorbs the up-projection matrices into the Q and output projections; it compresses by tens of times without being worse than MHA, at the cost of a more complex implementation.
- Side benefit: one copy of KV is reused by several query heads, which also raises the arithmetic intensity of decode attention.

Q6: ***What problem does PagedAttention solve? How does vLLM manage the KV cache?***

- **It solves the heavy fragmentation and waste caused by preallocating contiguous memory for the KV cache at maximum length; borrowing paging from operating systems, it splits the KV cache into fixed-size blocks that are allocated on demand and stored non-contiguously.**
- With the old approach actual memory utilization could be only 20%–40% (internal fragmentation, external fragmentation, duplicated shared prefixes).
- Each request has a block table mapping logical blocks to physical blocks, so waste only happens in the last block, and the attention kernel reads through the table indirectly.
- Blocks carry reference counts, so multiple sequences can share a prefix, with copy-on-write on modification.
- When memory runs out, requests can be preempted: their blocks are swapped to CPU or dropped and recomputed. The result is a much larger batch and significantly higher throughput.

Q7: ***What is continuous batching? How does it differ from static batching?***

- **Static batching waits for the whole batch to finish before admitting new requests, so slots of short requests sit idle; continuous batching schedules at the granularity of every decode step, removing a request as soon as it finishes and admitting new ones immediately.**
- Sequences of different lengths can share a batch: linear layers concatenate all tokens into one big matrix, while attention is computed per request.
- GPU utilization and throughput rise sharply and TTFT drops; it is standard in vLLM, SGLang and TensorRT-LLM.
- Side issues: new requests' prefill competes with decode and causes stalls, commonly handled with chunked prefill that splits long prompts into chunks mixed into decode batches; it also needs KV-memory-aware admission and preemption.

Q8: ***What is prefix caching (prompt caching)? How is it implemented, and where does it help?***

- **Many requests share the same prefix (system prompt, few-shot examples, multi-turn history, tool descriptions); storing that prefix's KV cache and reusing it skips that part of prefill, sharply reducing TTFT and compute.**
- Implementation: vLLM hashes each block's prefix for matching; SGLang's RadixAttention organizes the cache as a radix tree with LRU eviction; caches can also be tiered to CPU/SSD/distributed storage.
- Only exactly identical prefixes match, and changing one token in the middle invalidates everything after it, so prompts should put stable content first and changing content (such as timestamps) last.
- Schedulers can do cache-aware routing; commercial APIs often discount cache-hit input tokens.

## Attention Computation Optimization

Q9: ***What is the core idea of FlashAttention? Why is it faster and more memory-efficient?***

- **The bottleneck of standard attention is repeatedly writing and reading the N×N attention matrix to and from slow GPU memory (HBM); FlashAttention loads Q, K and V in tiles into fast on-chip SRAM and computes the whole attention on chip, never writing the N×N matrix back to memory.**
- After tiling you cannot get a full row's softmax at once, so online softmax keeps a running max and sum and corrects block by block; the result is exactly equal to standard attention, not an approximation.
- In the backward pass it does not store the attention matrix, only the output and each row's logsumexp, and recomputes on chip when needed; FLOPs rise slightly but memory drops from $O(N^2)$ to $O(N)$.
- It reduces memory reads and writes, not compute, because attention itself is memory-bound.

Q10: ***What did each FlashAttention version improve? Which attention kernels are used in decode?***

- **Each version fits the hardware more closely: FA2 improves parallelism and work partitioning and is about 2x faster than FA1; FA3 adds asynchrony and low-precision optimizations for the H100; in decode, Flash-Decoding splits the KV along the sequence dimension and processes it in parallel.**
- FA1: tiling + online softmax + recomputation, parallel over batch and heads.
- FA2: less non-matmul compute, extra parallelism along the sequence dimension, better work split between warps.
- FA3: uses Hopper's TMA asynchronous copies and WGMMA, producer/consumer warp specialization, overlap of softmax and matmul, and supports FP8.
- In decode the query length is 1, so parallelizing over batch×heads does not fill the GPU; Flash-Decoding splits the KV into segments, computes partial results in parallel and merges them. Libraries such as FlashInfer also provide efficient kernels for paged KV and prefix sharing.

## Quantization

Q11: ***What forms of quantization are used for LLM inference? How do weight-only quantization and W8A8 differ, and when is each appropriate?***

- **Weight-only (e.g. W4A16) compresses only the weights and dequantizes them to FP16 for compute, mainly saving memory and read bandwidth, which suits memory-bound decode and small batches; W8A8 quantizes both weights and activations and uses INT8/FP8 Tensor Cores to double compute, which suits compute-bound prefill and large-batch high-throughput serving.**
- Granularity can be per-tensor, per-channel or per-group (e.g. one group per 128 elements).
- Weight-only examples are GPTQ and AWQ; it does not reduce FLOPs, and at large batch the dequantization overhead can cancel the gains.
- The difficulty of W8A8 is the huge outliers in a few activation channels; representative solutions are SmoothQuant, LLM.int8() and FP8.
- More aggressive W4A4 or 1–2 bit schemes carry more quality risk; after quantization, evaluate specifically on sensitive tasks such as math, code and long context.

Q12: ***How does GPTQ work?***

- **GPTQ is layer-wise post-training quantization: using a little calibration data, it quantizes weights column by column, and after each column it uses second-order information (the Hessian) to adjust the not-yet-quantized weights to compensate for the error, keeping the layer's output as unchanged as possible.**
- The idea comes from OBS/OBQ; GPTQ's engineering improvements are quantizing all rows in the same column order so they share the inverse Hessian, lazy batched updates, and a Cholesky decomposition for numerical stability.
- It can quantize models with tens to hundreds of billions of parameters to 3–4 bits in a few GPU hours with small accuracy loss; it is often combined with group size 128 and act-order.
- Key formula (per-layer objective): $\min_{\hat W} \| W X - \hat W X \|_2^2$

Q13: ***How do AWQ and SmoothQuant each deal with outliers?***

- **Both use the equivalence "divide the activation by s, multiply the weight by s, and the result is unchanged" to move quantization difficulty around: SmoothQuant smooths activation outliers into the weights so that activations can be quantized too (W8A8); AWQ scales up the weights of important channels to protect them (weight-only).**
- SmoothQuant: activation outliers concentrate in a few channels while weights are smoother; a hyperparameter $\alpha$ (about 0.5) balances the difficulty between the two, and the scaling can be fused offline into the previous layer with no inference overhead.
- AWQ: about 1% of salient weight channels matter most, and "salient" is determined by the magnitude of the corresponding activations; scaling those weights up per channel reduces their relative error, the scale is found by a small grid search with no backpropagation, and it overfits the calibration set less than GPTQ.
- In one line: SmoothQuant is about making activations quantizable, AWQ is about protecting weights.

Q14: ***What are the advantages of FP8 inference? How is KV-cache quantization done?***

- **FP8 has native Tensor Core support on GPUs such as the H100, giving about twice the compute of BF16 with half the memory, and the floating-point format handles outliers better than INT8, with accuracy close to BF16, making it almost the default choice for W8A8. KV-cache quantization stores the KV in low precision such as FP8/INT8, doubling concurrency and halving decode attention's memory traffic.**
- FP8 comes in E4M3 (more precision, common for forward) and E5M2 (more range, common for gradients); fine-grained block-wise scales reduce error further.
- KV quantization needs per-tensor/per-head/per-token scales, and the kernel dequantizes after reading.
- More aggressive 2–4 bit schemes (e.g. KIVI) found that quantizing Keys per channel and Values per token works better, because Keys have fixed outlier channels.
- Risk: long-context retrieval tasks are sensitive to KV precision and need dedicated evaluation.

## Decoding Acceleration

Q15: ***How does speculative decoding work? Why does it preserve the output distribution?***

- **A small draft model quickly guesses several tokens, then the large model verifies them in parallel in a single forward pass; because decode is memory-bound, verifying several tokens takes almost the same time as generating one, so each large-model forward pass can yield multiple tokens.**
- Draft tokens are accepted one by one with probability $\min(1, p/q)$; at the first rejection a token is resampled from the residual distribution "target minus draft" and the round stops.
- This rejection-sampling rule can be proven to give exactly the same output distribution as sampling directly from the large model, so the speedup is lossless.
- The speedup depends on the acceptance rate, draft length and draft-model cost; at large batch the system is already close to compute-bound and the gain shrinks; draft and target must share a tokenizer.

Q16: ***How do Medusa, EAGLE and similar methods differ from classic speculative decoding?***

- **The classic method needs a separate draft model to maintain; Medusa, EAGLE and similar methods let the target model draft for itself with only very lightweight prediction heads, removing the separate model and achieving higher acceptance rates.**
- Medusa: adds several decoding heads on the last hidden state, where head k directly predicts token t+k, and verifies multiple candidates at once with tree attention; it is often paired with relaxed "typical acceptance", in which case the distribution is no longer strictly preserved.
- EAGLE: drafts autoregressively at the feature level and also feeds in the token sampled at the previous step, with higher acceptance than Medusa; EAGLE-2 uses a dynamic draft tree and EAGLE-3 fuses multi-layer features.
- Others: Lookahead decoding (no draft model), Prompt lookup (finds n-grams in the input, good for summarization and code editing), and MTP heads (e.g. DeepSeek-V3).
- What they share is "parallel verification by the target model"; schemes with strict rejection sampling preserve the distribution.

## Decoding Strategies and Constrained Generation

Q17: ***What are the common decoding strategies? How do temperature, top-k, top-p and repetition penalties work?***

- **Greedy takes the most probable token each step and beam search keeps several high-probability sequences; sampling methods first use temperature to adjust how sharp the distribution is, then use top-k/top-p to cut off the low-probability tail before sampling, while repetition penalties lower the probability of tokens that have already appeared.**
- Greedy is deterministic but prone to repetition; beam search suits tasks with a reference answer such as translation, but is bland in open-ended generation and needs a length penalty.
- Temperature below 1 is more conservative, above 1 more diverse, and approaching 0 is equivalent to greedy.
- Top-k keeps a fixed k tokens; top-p keeps the smallest set whose cumulative probability reaches p, so its size adapts; min-p keeps tokens whose probability is at least a fraction of the top probability, which is more stable at high temperature.
- Repetition penalties include the repetition penalty (rescaling logits of seen tokens), frequency/presence penalties (subtracting by count or by occurrence), and no-repeat n-gram; a common order is penalties → temperature → truncation → sampling.

Q18: ***How is structured/constrained decoding implemented? What are the pitfalls?***

- **At each step, compute the set of legal tokens from the current grammar state, set the logits of illegal tokens to negative infinity, and then sample, which guarantees the output strictly follows the JSON Schema, regex or grammar.**
- Regex/JSON Schema is compiled into a finite-state machine with the legal tokens of each state precomputed (e.g. Outlines); nested structures use a pushdown automaton (e.g. XGrammar, where most masks can be precomputed).
- SGLang's jump-forward: when the next characters are uniquely determined by the grammar, skip them instead of generating token by token.
- Pitfalls: token boundaries do not align with grammar symbols, so matching must happen at the character level; forced masking distorts the model's distribution and can produce output that is syntactically valid but semantically poor, which putting a schema example in the prompt helps; computing masks synchronously on the CPU slows down large-batch decode.

## Parallelism, MoE and System Architecture

Q19: ***How does tensor parallelism (TP) work for inference? How do you choose between it and pipeline parallelism (PP)?***

- **TP splits each layer's weights across several GPUs that compute together (the MLP is split first by columns then by rows, attention by heads), with two all-reduces per layer; because each GPU reads fewer weights and less KV, it directly lowers single-request latency. PP splits layers across devices, which does not lower single-request latency but communicates little and suits cross-node setups.**
- TP communicates frequently, so it suits intra-node NVLink; with too high a TP degree communication takes a larger share.
- The TP degree must divide the number of KV heads, otherwise KV heads must be replicated.
- PP needs several micro-batches to fill the pipeline.
- Others: data parallelism (replicas) scales throughput, expert parallelism is for MoE, and context parallelism is for prefill of very long prompts.

Q20: ***What are the characteristics and challenges of MoE model inference?***

- **Each MoE token activates only a few experts, so compute scales with the activated parameters and is low, but memory must hold all experts; the main challenges are memory efficiency, expert load imbalance, and cross-GPU all-to-all communication.**
- Example: DeepSeek-V3 has 671B total parameters but activates only 37B per token.
- At small batch each expert receives very few tokens, so arithmetic intensity is even lower; larger batches are needed to realize the benefit.
- Hot experts become bottlenecks: training uses load-balancing losses or bias-adjusted routing, and inference can replicate hot experts.
- Expert parallelism dispatches tokens via all-to-all with high communication cost, requiring compute-communication overlap; tokens for the same expert are computed together with grouped GEMM. A common deployment uses TP/DP for attention and EP for MoE.

Q21: ***What is disaggregated prefill/decode serving, and why do it?***

- **Prefill and decode run on separate GPU instances, because running them together causes interference (long prefills block decode), and their optimal parallelism, batch size and hardware configuration all differ.**
- Prefill instances target TTFT and are compute-bound; decode instances target TPOT and large-batch throughput and are memory-bound; after prefill, the KV cache is transferred to a decode instance over NVLink/RDMA.
- The benefit is that the two phases can be scaled and tuned for their SLOs independently, improving goodput (e.g. DistServe, Splitwise, Mooncake).
- The cost is KV transfer overhead and system complexity, with limited gains for short prompts; without disaggregation, chunked prefill offers a compromise within a single instance.

Q22: ***What techniques optimize long-context inference?***

- **The problems with long context are that prefill attention grows quadratically with length and the KV cache grows linearly; the optimizations make each token look at less history (sliding window/sparse), store less KV (eviction/compression/quantization), and split or offload work at the system level.**
- Sliding-window attention: attend only to the last w tokens with a fixed-size KV, often interleaved with global attention layers to keep long-range ability.
- StreamingLLM: keeping the first few "attention sink" tokens plus a recent window enables stable streaming generation, but it does not remember the middle content.
- Sparse attention: fixed patterns (Longformer, BigBird) or dynamic selection (DeepSeek's NSA, and DSA, which uses a lightweight indexer to pick top-k past tokens for each query).
- KV eviction and compression: H2O, SnapKV, KV quantization, MLA; at the system level, context parallelism/Ring Attention, chunked prefill and offloading KV to CPU. Also watch position-encoding extrapolation (YaRN) and "needle in a haystack" evaluation rather than looking only at speed.

Q23: ***What are the mainstream LLM serving frameworks and their characteristics?***

- **The mainstream options are vLLM (general-purpose with the broadest ecosystem), SGLang (strong prefix caching and structured output, heavy investment in large-scale MoE deployment), and TensorRT-LLM (official NVIDIA, top performance but a steeper learning curve); llama.cpp/Ollama are common for edge devices and CPUs.**
- vLLM: started with PagedAttention and continuous batching, broad model and hardware coverage, OpenAI-compatible API, the default for general deployment.
- SGLang: RadixAttention, efficient constrained decoding, suited to multi-call and agent programs, strong at EP and PD disaggregation.
- TensorRT-LLM: precompiled engines, deep kernel fusion, good FP8/FP4 support, often paired with the Triton server.
- Choose based on hardware, model support, workload characteristics and operating cost, and benchmark with your own real traffic rather than only public benchmarks.

Q24: ***How do you estimate the resources and cost of serving an LLM?***

- **Memory = weights (parameter count × bytes per parameter) + KV cache (per-token size × concurrency × context length) + 10%–20% extra overhead; for latency, decode is "bytes read / bandwidth" and prefill is "compute / FLOPS"; cost = GPU cost per hour / tokens generated per hour.**
- Example: Llama-3-70B in FP8 has about 70 GB of weights and about 320 KB of KV per token; 64 concurrent requests × 8K context need about 168 GB of KV, requiring multi-GPU TP.
- Decode TPOT is about (weight + KV bytes) / TP degree / effective bandwidth, and prefill TTFT is about $2 N_{params} n_{prompt}$ divided by effective compute.
- Input tokens are cheap (parallel, compute-bound) and output tokens expensive (memory-bound), so APIs usually price output far higher than input, and cache-hit input is cheaper still.
- Cost levers: higher utilization, quantization, prefix caching, speculative decoding, routing by difficulty to smaller models, and batch APIs for offline jobs.
- Key formula: $\text{cost per million tokens} = \frac{\text{GPU hourly price} \times \text{number of GPUs}}{\text{throughput (tokens/s)} \times 3600} \times 10^6$

Q25: ***Your production LLM service has high latency. How would you systematically diagnose and optimize it?***

- **First break down the metrics to find whether TTFT or TPOT is high, and whether it is high overall or only P99 jitter; then look at GPU resources and profiling to find the bottleneck; then apply targeted fixes at the model, system and application layers, validating every change with quality evaluation and real-traffic replay.**
- High TTFT: queueing, overly long prompts, blocked prefill → scale out, prefix caching, chunked prefill, PD disaggregation, limit input length.
- High TPOT: batch too large, insufficient KV causing preemption and recomputation, long context → tune the batch/token budget, KV quantization, GQA models, higher TP, speculative decoding.
- P99 jitter: long prefills cutting in, preemption, CPU bottlenecks (tokenizer, sampling, constrained-decoding masks).
- Other measures: quantization, distilled smaller models, CUDA graphs, kernel fusion, cache-aware routing; on the application side, streaming output, capping output length, putting stable prefixes first, and parallelizing independent calls.
