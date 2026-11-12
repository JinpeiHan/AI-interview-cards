# LLM推理优化

## 推理阶段与性能瓶颈

Q1：***LLM推理的prefill和decode阶段分别是什么？各自有什么特点？***

- **Prefill（预填充）**：一次性处理整个prompt，所有输入token**并行**前向，计算并写入每层的KV cache，输出第一个token。
    - 本质是大矩阵乘（GEMM），FLOPs ≈ $2 \cdot N_{params} \cdot n_{prompt}$（不含注意力的 $O(n^2)$ 项），通常是**计算密集（compute-bound）**。
    - 决定**首token延迟（TTFT）**；长prompt时注意力的二次方开销显著。
- **Decode（解码）**：自回归地每步只生成**1个token**，读取全部权重和历史KV cache，追加新的K/V。
    - 本质是矩阵-向量乘（GEMV，或小batch的瘦GEMM），每读一次权重只做很少计算，通常是**访存密集（memory-bandwidth-bound）**。
    - 决定**每token延迟（TPOT/ITL）**；输出越长，decode占总时间比例越大。
- 追问：两阶段特性不同，所以才有 **chunked prefill**、**prefill/decode分离部署**等调度优化（见后文）。

Q2：***什么是compute-bound和memory-bandwidth-bound？如何用算术强度（arithmetic intensity）判断？***

- **算术强度** $I = \frac{\text{FLOPs}}{\text{访存字节数}}$。
- **Roofline模型**：可达性能 $= \min(\text{峰值算力},\ I \times \text{显存带宽})$。拐点 $I^* = \frac{\text{峰值算力}}{\text{带宽}}$。
    - $I < I^*$：访存瓶颈；$I > I^*$：计算瓶颈。
    - 以H100 SXM为例，BF16稠密约 989 TFLOPS、HBM带宽约 3.35 TB/s，$I^* \approx 300$ FLOPs/Byte。
- **decode的线性层**：权重 $W \in \mathbb{R}^{d \times d}$（FP16，$2d^2$ 字节），batch为 $B$ 时FLOPs为 $2Bd^2$，故 $I \approx B$。要 $B$ 达到几百才能吃满算力 → 这就是为什么**增大batch能几乎“免费”提升decode吞吐**。
- **decode的注意力**：每个请求的KV cache只被自己的1个query使用，$I \approx 1$（GQA下约为组大小 $g$），batch无法摊薄，始终访存受限。
- **单流decode延迟下界**：$t_{token} \gtrsim \frac{\text{权重字节} + \text{KV字节}}{\text{带宽}}$。例：7B FP16 约14 GB，在3.35 TB/s上约 4 ms/token，即单流上限约 240 token/s。

Q3：***LLM服务常见的延迟/吞吐指标有哪些？它们之间如何权衡？***

- **TTFT（Time To First Token）**：从请求到达到第一个token返回，含排队 + prefill。影响交互体验。
- **TPOT（Time Per Output Token）/ ITL（Inter-Token Latency）**：相邻输出token的间隔，决定“打字速度”。TPOT通常指平均值，ITL关注分布（P99抖动）。
- **端到端延迟** $\approx \text{TTFT} + \text{TPOT} \times (n_{out}-1)$。
- **吞吐（Throughput）**：系统每秒生成的token数（或请求数/秒）。
- **Goodput**：满足SLO（如TTFT<1s、TPOT<50ms）的请求吞吐，比纯吞吐更贴近线上目标。
- 权衡：
    - batch越大 → 吞吐越高，但单token计算时间、排队与相互干扰增加，TPOT变大。
    - prefill插入decode批次 → 新请求TTFT降低，但正在decode的请求出现卡顿（ITL尖刺）。
    - 量化/投机解码 → 降延迟，但可能影响质量或在高负载下收益下降。
- 追问：离线批处理优化吞吐/成本，在线聊天优化P99 TTFT与TPOT，Agent多轮场景重视前缀缓存命中率。

## KV Cache及其显存优化

Q4：***为什么需要KV cache？KV cache的显存如何计算？***

- 自回归生成时，历史token的K、V在后续每步都不变。缓存它们可把每步注意力从重算全部前缀 $O(n^2)$ 降为只算新token $O(n)$。
- 显存公式：
$$\text{KV bytes} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{seq} \cdot \text{bytes} \quad (\times\ \text{batch})$$
    - 2 表示K和V；$L$ 为层数；$n_{kv}$ 为KV头数；$d_{head}$ 为每头维度；bytes为每元素字节数（FP16为2）。
- 例子：
    - Llama-2-7B（MHA，$L=32, n_{kv}=32, d_{head}=128$）：每token $2\cdot32\cdot32\cdot128\cdot2 = 512$ KB，4K上下文单序列约 2 GB。
    - Llama-3-8B（GQA，$n_{kv}=8$）：每token 128 KB，是前者的 1/4。
    - Llama-3-70B（$L=80, n_{kv}=8$）：每token 320 KB，128K上下文单序列约 40 GB。
- 结论：长上下文、大batch下**KV cache往往超过权重显存**，成为并发数的主要限制。
- 优化方向：减少头数（MQA/GQA/MLA）、减少精度（KV量化）、减少长度（滑窗/稀疏/驱逐）、减少碎片（PagedAttention）、复用（前缀缓存）。

Q5：***MQA、GQA、MLA如何压缩KV cache？各有什么代价？***

- **MHA**：$n_{kv} = n_h$，每个query头有独立K/V。
- **MQA（Multi-Query Attention, 2019）**：所有query头共享**1组**K/V，$n_{kv}=1$，KV cache缩小 $n_h$ 倍；代价是质量有所下降、训练可能不稳定；TP时KV头需复制。
- **GQA（Grouped-Query Attention, 2023）**：query头分 $G$ 组，每组共享一组K/V，$n_{kv}=G$（如Llama-3的 $G=8$）。质量接近MHA，显存接近MQA，是当前主流。可通过对MHA检查点的KV头做**均值池化 + 少量继续训练（uptraining）**得到。
- **MLA（Multi-head Latent Attention, DeepSeek-V2, 2024）**：
    - 将K/V低秩压缩为潜向量 $c_t^{KV} = W^{DKV} h_t$（维度 $d_c$，如512），只缓存 $c_t^{KV}$ 及一个解耦的RoPE key（如64维）。
    - 推理时把上投影 $W^{UK}$ 吸收进 $W^Q$、$W^{UV}$ 吸收进 $W^O$，无需显式还原K/V。
    - 由于RoPE与位置相关、无法被吸收，故单独保留一个共享的小维度RoPE key（decoupled RoPE）。
    - 每token每层缓存约 $d_c + d_h^R = 576$ 个元素，对比MHA的 $2 n_h d_h$（DeepSeek-V2中为32768），压缩数十倍，且论文报告效果不弱于MHA。
- 追问：GQA/MQA也提升了decode注意力的算术强度（一份KV被 $g$ 个query头复用）。

Q6：***PagedAttention解决了什么问题？vLLM是如何管理KV cache的？***

- 问题：早期系统为每个请求按**最大长度预分配连续显存**，导致
    - **内部碎片**：实际生成长度远小于上限；
    - **外部碎片**：不同大小的空洞；
    - 无法共享：并行采样、beam search的公共前缀被重复存储。
    - 论文统计显存实际利用率可能只有20%–40%。
- **PagedAttention（vLLM, SOSP 2023）**：借鉴操作系统的**虚拟内存分页**。
    - KV cache切成固定大小的**block**（如每块16个token），物理上不连续；
    - 每个请求维护**block table**（逻辑块 → 物理块映射），按需分配，浪费只发生在最后一个块；
    - 注意力kernel按block table间接寻址读取KV；
    - 块带**引用计数**，支持多序列共享前缀，修改时**写时复制（copy-on-write）**。
- 效果：显存浪费接近0，可容纳更大batch，吞吐显著提升。
- 显存不足时的**抢占**：把某些请求的块换出到CPU（swap）或直接丢弃后重算（recompute）。

Q7：***什么是continuous batching？与static batching有什么区别？***

- **Static batching**：一批请求一起开始、一起结束，短请求结束后其槽位空转，要等最长请求生成完；新请求需排队等整批完成。
- **Continuous batching（iteration-level scheduling, Orca, OSDI 2022）**：
    - 以**每个decode步**为调度粒度，请求一结束立刻移出，新请求立刻加入；
    - 不同长度的序列可在同一批次：对线性层把所有token拼接成一个大矩阵计算，对注意力按请求分别计算（selective batching）。
- 效果：GPU利用率和吞吐大幅提升，TTFT降低，是vLLM、SGLang、TensorRT-LLM（in-flight batching）的标配。
- 配套问题：
    - 新请求的prefill会与decode争抢算力，导致ITL尖刺 → **chunked prefill**（如Sarathi-Serve）把长prompt切块，与decode token组成固定token预算的混合批次；
    - 需要KV显存感知的准入控制与抢占策略。

Q8：***什么是前缀缓存（prefix caching / prompt caching）？如何实现？适用于哪些场景？***

- 思想：很多请求共享相同前缀（system prompt、few-shot示例、多轮对话历史、Agent工具描述、同一文档的多次提问），其KV cache可复用，跳过这部分的prefill。
- 实现：
    - **vLLM Automatic Prefix Caching**：按block对“前缀所有token”做哈希，命中则直接引用已有物理块；
    - **SGLang RadixAttention**：用**基数树（radix tree）**组织所有缓存的token序列，支持任意前缀匹配，LRU驱逐；
    - 也可分层存储到CPU内存/SSD/分布式KV存储（如Mooncake的思路）。
- 收益：TTFT与prefill算力大幅下降；商业API常对缓存命中的输入token打折计费。
- 注意：
    - 只能匹配**完全相同的前缀**，中间改一个token，后续全部失效 → prompt设计应把**稳定内容放前、变化内容放后**（不要把时间戳放在system prompt开头）；
    - 调度时可做缓存感知路由，把相同前缀的请求发往同一实例。

## 注意力计算优化

Q9：***FlashAttention的核心原理是什么？为什么能加速且省显存？***

- 瓶颈判断：标准注意力需在HBM中写/读 $N\times N$ 的 $S=QK^T$ 和 $P=\text{softmax}(S)$，访存 $\Theta(N^2)$，在GPU上是**访存受限**而非计算受限。
- **IO-awareness**：GPU的SRAM很快但小，HBM大但慢。FlashAttention的目标是减少HBM读写次数，而不是减少FLOPs。
- **Tiling（分块）**：把 $Q, K, V$ 分块加载到SRAM，在片上完成 $QK^T$、softmax、乘 $V$，不把 $N\times N$ 矩阵写回HBM（kernel融合）。
- **Online softmax**：分块时无法一次得到整行的max和求和，维护运行统计量逐块修正：
$$m^{new} = \max(m, \tilde m),\quad \ell^{new} = e^{m-m^{new}}\ell + e^{\tilde m - m^{new}}\tilde \ell$$
$$O^{new} = e^{m-m^{new}} O + e^{\tilde m - m^{new}} \tilde P V_j$$
  最后输出 $O/\ell$，结果与标准注意力**数值精确一致**（不是近似）。
- **Recomputation（重计算）**：反向时不存 $P$，只存输出 $O$ 和每行的logsumexp，反向时在SRAM中重算 $S, P$。FLOPs略增，但显存从 $O(N^2)$ 降到 $O(N)$，且因访存少整体更快。
- IO复杂度：标准注意力 $\Theta(Nd + N^2)$；FlashAttention $\Theta(N^2 d^2 / M)$（$M$ 为SRAM大小）。

Q10：***FlashAttention各版本有什么改进？decode阶段用什么注意力kernel？***

- **FlashAttention-1（2022）**：tiling + online softmax + 重计算，按batch和head并行。
- **FlashAttention-2（2023）**：
    - 减少非矩阵乘FLOPs（rescale延后到最后统一做）；
    - 额外在**序列长度维度**并行，长序列小batch时也能占满SM；
    - 改进warp间工作划分（Q在warp间切分），减少共享内存通信；速度约为FA1的2倍。
- **FlashAttention-3（2024，面向Hopper H100）**：利用TMA异步拷贝与WGMMA，**warp specialization**（生产者/消费者warp）、softmax与GEMM的乒乓重叠，支持**FP8**（配合块量化与非相干处理降低误差）。
- 之后还有面向Blackwell等新架构的版本，思路一致：贴合硬件特性减少访存、重叠计算与数据搬运。
- **Decode阶段**：query长度为1，按batch×head并行不足以占满GPU → **Flash-Decoding**把KV沿序列维**切分（split-K）**并行计算部分结果，再用logsumexp合并。
- 工程上还有**FlashInfer**等库，为分页KV、前缀共享、不同注意力变体提供高效kernel。

## 量化

Q11：***LLM推理量化有哪些形式？weight-only量化和W8A8有什么区别，分别适合什么场景？***

- 基本形式：$x_q = \text{round}(x / s) + z$，$s$ 为scale，$z$ 为零点；粒度可以是per-tensor、per-channel、per-group（如每128个元素一组）。
- **Weight-only（如W4A16、W8A16）**：
    - 仅权重量化存储，计算时反量化回FP16与激活相乘；
    - 显存和**权重读带宽**直接减少2–4倍 → 对**访存受限的decode、小batch**加速显著；
    - 不减少FLOPs，大batch/prefill下反量化开销可能抵消收益。
    - 代表：GPTQ、AWQ，INT4 group-wise。
- **Weight + Activation（W8A8 INT8 / FP8）**：
    - 权重和激活都量化，使用INT8/FP8 Tensor Core，**算力翻倍** → 对**计算受限的prefill、大batch高吞吐**有效；
    - 难点是激活存在**离群值（outlier）**（LLM中少数通道幅值极大），直接量化误差大。
    - 代表：SmoothQuant、LLM.int8()（离群通道走FP16的混合分解）、FP8。
- 更激进：W4A4（如需旋转消除离群值的QuaRot、SpinQuant等），以及1–2 bit方案，质量风险更大。
- 追问：量化后要在目标任务上评估（困惑度、下游准确率、长上下文、推理/代码任务），低bit对数学/代码等任务更敏感。

Q12：***GPTQ的原理是什么？***

- 训练后量化（PTQ），逐层求解：
$$\min_{\hat W} \| W X - \hat W X \|_2^2$$
  $X$ 为少量校准数据在该层的输入。
- 基于OBS/OBQ思想：每量化一个权重，利用二阶信息（Hessian $H = 2XX^T$）**更新其余尚未量化的权重**来补偿误差：
$$\delta_F = -\frac{w_q - \text{quant}(w_q)}{[H_F^{-1}]_{qq}} (H_F^{-1})_{:,q}$$
- GPTQ的工程改进：
    - 所有行**按相同列顺序**量化，共享 $H^{-1}$，复杂度大幅下降；
    - **惰性批量更新**（一次处理一组列），提高GPU利用率；
    - 用**Cholesky分解**稳定求 $H^{-1}$。
- 效果：几个GPU小时即可将百亿级甚至千亿级模型量化到3–4 bit，精度损失小；常配合group-wise scale（如group size 128）与act-order。

Q13：***AWQ和SmoothQuant分别是怎么处理离群值的？***

- 共同思路：利用等价变换 $Y = XW = (X \operatorname{diag}(s)^{-1})(\operatorname{diag}(s) W)$，在激活和权重之间“搬运”量化难度。
- **SmoothQuant（W8A8）**：
    - 激活的离群值集中在少数通道，权重较平滑 → 把激活的大幅值“平滑”到权重上；
    - $s_j = \frac{\max |X_j|^{\alpha}}{\max |W_j|^{1-\alpha}}$，$\alpha$ 通常取0.5左右；
    - $\operatorname{diag}(s)^{-1}$ 可离线融合进前一层LayerNorm/线性层，推理无额外开销。
- **AWQ（Activation-aware Weight Quantization，weight-only）**：
    - 观察：约1%的**显著权重通道**对精度至关重要，显著性由**对应激活的幅值**决定，而非权重本身大小；
    - 不用混合精度保留，而是对显著通道的权重**按通道放大** $s$（激活相应缩小），降低其相对量化误差；
    - $s = s_X^{\alpha}$，在小网格上搜索 $\alpha$ 最小化输出误差；无需反向传播与重建，对校准集过拟合风险小于GPTQ。
- 对比：SmoothQuant是为了让**激活**可量化；AWQ是为了保护**权重**中的重要通道。

Q14：***FP8推理有什么优势？KV cache量化怎么做？***

- **FP8**：两种格式，**E4M3**（精度更高，范围约±448，常用于权重/激活前向）和**E5M2**（范围更大，常用于梯度）。
    - Hopper/Ada及之后GPU原生支持FP8 Tensor Core，相比BF16算力约翻倍、显存减半；
    - 浮点格式对离群值比INT8更友好，通常per-tensor或per-channel/per-block scale就能达到接近BF16的精度，**几乎是W8A8的首选**；
    - 细粒度（block-wise）scale进一步降低误差，例如DeepSeek-V3在训练中就采用了细粒度FP8。
- **KV cache量化**：
    - KV cache在长上下文/大batch时占主要显存与带宽，量化到FP8/INT8可使并发翻倍、decode注意力访存减半；
    - 需要scale（per-tensor/per-head/per-token），注意力kernel读取后反量化；
    - 更激进的2–4 bit方案（如KIVI）发现：**Key按通道量化、Value按token量化**效果更好（Key存在固定的离群通道）；
    - 风险：长上下文检索类任务对KV精度敏感，需要专门评估。

## 解码加速

Q15：***投机解码（speculative decoding）的原理是什么？为什么能保证输出分布不变？***

- 动机：decode是访存受限，大模型一次前向验证 $\gamma+1$ 个token与生成1个token的耗时几乎相同。
- 流程：
    1. 小而快的**草稿模型** $q$ 自回归猜 $\gamma$ 个token $x_1,\dots,x_\gamma$；
    2. **目标模型** $p$ 一次前向并行计算这些位置的分布；
    3. 从左到右逐个验证：以概率 $\min\left(1, \frac{p(x)}{q(x)}\right)$ 接受；
    4. 第一次拒绝时，从**残差分布** $p'(x) = \frac{\max(0,\ p(x) - q(x))}{\sum_{x'} \max(0,\ p(x') - q(x'))}$ 采样一个token替代并停止；若全部接受，再从 $p$ 额外采样一个token。
- **正确性证明**：设 $\beta = \sum_x \min(p(x), q(x))$ 为接受概率，
$$P(\text{输出}=x) = q(x)\min\left(1,\tfrac{p(x)}{q(x)}\right) + (1-\beta)\frac{\max(0,p(x)-q(x))}{1-\beta} = \min(p,q) + \max(0,p-q) = p(x)$$
  因此是**无损**加速，与直接从目标模型采样同分布（贪心时退化为“前缀匹配则接受”）。
- 加速比：设每个token接受率为 $\alpha$，每轮期望生成token数为 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$；实际加速还取决于草稿模型开销 $c$ 和 $\gamma$ 的选择。
- 注意：大batch时系统已接近计算受限，额外验证token的计算不再“免费”，收益下降；草稿与目标需共享tokenizer。

Q16：***Medusa、EAGLE等方法与经典投机解码有什么不同？***

- 经典方法需要单独的草稿模型，需维护对齐且占显存。改进方向是**让目标模型自己出草稿**：
- **Medusa（2024）**：在目标模型最后的hidden state上加多个**额外解码头**，第 $k$ 个头直接预测第 $t+k$ 个token；用**树形注意力（tree attention）**一次验证多个候选组合；可冻结主干只训练头。常用“典型接受”（typical acceptance）放宽验证，此时不再严格保分布。
- **EAGLE（2024）系列**：在**特征层（倒数第二层hidden state）**做自回归草稿，并输入上一步采样的token以消除不确定性；草稿头仅为一个轻量Transformer层，接受率高于Medusa。EAGLE-2使用基于置信度的**动态草稿树**，EAGLE-3进一步融合多层特征、去掉特征预测损失以更好地随数据扩展。
- 其他：**Lookahead decoding**（Jacobi迭代 + n-gram池，无需草稿模型）、**Prompt lookup**（从输入中检索n-gram，适合摘要/代码编辑等输出与输入高度重叠的场景）、**MTP（multi-token prediction）**头（如DeepSeek-V3训练时的MTP模块可在推理时作草稿）。
- 共同点：都依赖“目标模型并行验证”这一步，带有严格拒绝采样的方案可保证分布不变。

## 解码策略与约束生成

Q17：***常见的解码策略有哪些？temperature、top-k、top-p、重复惩罚分别如何作用？***

- **Greedy**：每步取argmax，确定性强，但容易重复、单调。
- **Beam search**：保留 $k$ 条累积对数概率最高的序列，适合翻译/ASR等有“标准答案”的任务；开放式生成中易产生通用、重复的文本，且需要长度惩罚 $\frac{\log P}{|y|^\alpha}$。
- **Temperature**：$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$。$T<1$ 更尖锐/保守，$T>1$ 更平坦/多样，$T\to0$ 等价于greedy。
- **Top-k**：只保留概率最高的 $k$ 个token后重新归一化；缺点是 $k$ 固定，不适应分布的尖锐程度。
- **Top-p（nucleus）**：保留累积概率达到 $p$ 的最小token集合，候选集合大小自适应。
- **Min-p**：保留概率 $\ge p_{min} \cdot p_{max}$ 的token，高温时比top-p更稳。
- **重复惩罚**：
    - CTRL式repetition penalty：对已出现token，正logit除以 $\theta$、负logit乘以 $\theta$（$\theta>1$）；
    - frequency penalty：按出现次数线性扣减logit；presence penalty：出现过即扣一个常数；
    - no-repeat n-gram：禁止重复的n-gram（多用于beam search）。
- 常见处理顺序：logits → 惩罚 → temperature → top-k/top-p/min-p过滤 → 采样。
- 追问：推理模型/代码任务常用较低温度或厂商推荐的特定参数；需要可复现时固定seed，但batch不变性等因素仍可能导致结果不完全确定。

Q18：***结构化/约束解码（structured decoding）如何实现？有什么坑？***

- 目标：保证输出严格满足JSON Schema、正则或上下文无关文法（CFG），例如函数调用参数。
- 实现：每步根据当前语法状态计算**合法token集合**，把非法token的logit置为 $-\infty$（logit mask），再采样。
    - **正则/JSON Schema → 有限状态机（FSM）**：预先计算每个状态下的合法token（如Outlines）；
    - **CFG → 下推自动机（PDA）**：支持嵌套结构；XGrammar等通过区分“与上下文无关的token”预计算大部分mask，并与GPU计算重叠，使开销很小；
    - **Jump-forward**（SGLang）：当后续字符被语法唯一确定（如 `{"name": "`），直接跳过逐token生成。
- 坑：
    - **token边界与语法不对齐**：一个token可能横跨多个语法符号，需要在字节/字符层面做匹配；
    - **分布扭曲**：强制mask改变了模型分布，模型若“想”输出别的内容，可能生成语法正确但语义很差的结果；可在prompt中给出schema/示例缓解；
    - 推理模型可先自由“思考”再进入受约束的最终输出段；
    - mask计算若在CPU上同步进行，会拖慢大batch的decode。

## 并行、MoE与系统架构

Q19：***推理中的张量并行（TP）是如何工作的？与流水线并行（PP）如何选择？***

- **TP（Megatron式）**：
    - MLP：第一层权重**按列切分**（各卡独立算部分中间激活，GeLU/SwiGLU可逐元素在本地做），第二层**按行切分**，最后一次**all-reduce**求和；
    - Attention：按**头**切分，每卡负责若干Q头及对应KV头，输出投影按行切分，再一次all-reduce；
    - 每层约2次all-reduce，通信频繁 → 适合**节点内NVLink**互联。
- 收益：权重与KV cache分摊到多卡，**每卡读取的字节数减少 → 单请求decode延迟下降**，这是TP相对其他并行在推理中的核心价值。
- 限制：$n_{kv}$ 需能被TP度整除，否则需复制KV头（如GQA的8个KV头在TP=16时需复制）；TP度过大时通信占比上升。
- **PP**：按层切分到不同设备，单请求需依次经过所有阶段，**不降低单请求延迟**，但通信量小、适合跨节点；需多个micro-batch填满流水线。
- 其他：**数据并行（多副本）**扩展吞吐；**专家并行（EP）**用于MoE；**上下文/序列并行**用于超长prompt的prefill。

Q20：***MoE模型的推理有哪些特点和挑战？***

- 特点：每个token只由router选出的top-k个专家处理，**总参数量大、激活参数量小**（如DeepSeek-V3总参数671B、每token激活37B）。
    - **计算**按激活参数算，单token FLOPs低；
    - **显存**按总参数算，所有专家都必须驻留。
- 挑战：
    - **decode访存**：小batch时每个专家只分到很少token，算术强度更低；batch增大后会激活更多不同专家，读取权重量增多 → MoE需要更大batch才能发挥优势；
    - **负载不均**：热门专家成为瓶颈。训练时用辅助负载均衡损失（或DeepSeek-V3的无辅助损失、用偏置项调节路由的策略）；推理时可**复制热门专家（冗余专家）**并动态重排；
    - **专家并行（EP）**：专家分布在不同GPU，token通过**all-to-all**分发与汇聚，通信开销大，需要高效通信库及计算-通信重叠（如双micro-batch交错）；
    - kernel：同一专家的token聚合后用**grouped GEMM**批量计算。
- 部署常见组合：Attention部分用TP/DP，MoE部分用EP；大规模部署配合prefill/decode分离。

Q21：***什么是prefill/decode分离（disaggregated serving）？为什么要这样做？***

- 问题：同一GPU同时跑prefill和decode会**相互干扰**——长prefill会阻塞decode导致ITL尖刺；两阶段最优的并行策略、batch大小和硬件配置也不同。
- **分离部署**（如DistServe、Splitwise、Mooncake，2024年前后提出）：
    - **Prefill实例**：专门处理prompt，追求TTFT，计算密集，可用高算力配置；
    - **Decode实例**：专门生成，追求TPOT与大batch吞吐，访存密集，可用不同并行度；
    - prefill完成后把**KV cache通过高速互联（NVLink/RDMA）传输**到decode实例。
- 收益：两阶段独立扩缩容、独立优化SLO，提升goodput；可配合分布式KV缓存池做前缀复用。
- 代价：KV传输带宽与延迟、系统复杂度更高；短prompt场景收益有限。
- 不分离时的替代：**chunked prefill**混合批次，在单实例内平衡TTFT和ITL。

Q22：***长上下文推理有哪些优化手段？***

- 问题：prefill注意力 $O(n^2)$ 计算；decode每步读取 $O(n)$ 的KV，KV显存随长度线性增长。
- **滑动窗口注意力（SWA）**：每个token只关注最近 $w$ 个token（如Mistral 7B的 $w=4096$），KV cache用**滚动缓冲区**固定大小；多层堆叠后感受野约为 $L \times w$。常与全局注意力层交替（如部分层全局、部分层局部）以保留长程能力。
- **Attention sink（StreamingLLM）**：发现开头少数token吸收大量注意力，保留这些“sink” token + 最近窗口即可稳定地进行流式生成（但并不能真正记住中间内容）。
- **稀疏注意力**：
    - 固定模式：局部 + 全局 + 随机（Longformer、BigBird）；
    - 可训练/动态稀疏：如DeepSeek的NSA（压缩、选择、滑窗三路分支）以及DeepSeek Sparse Attention（用轻量indexer为每个query选top-k个历史token）。
- **KV驱逐/压缩**：H2O（保留累计注意力高的“重击者”token）、SnapKV（根据prompt末尾的观察窗口挑选重要KV）、KV量化、MLA等。
- **系统层面**：上下文并行/Ring Attention切分超长prefill；分块prefill；把KV卸载到CPU。
- 另外需关注位置编码外推（如RoPE插值、YaRN）以及“大海捞针”等长上下文能力评测，避免只看速度不看效果。

Q23：***主流的LLM推理框架有哪些？各自有什么特点？***

- **vLLM**：PagedAttention、continuous batching起家，模型与硬件覆盖广，社区活跃，支持前缀缓存、投机解码、量化、TP/PP/EP、结构化输出，OpenAI兼容API，是通用部署的常见默认选择。
- **SGLang**：RadixAttention前缀缓存、高效的结构化输出与约束解码、前端编程语言（适合多调用、Agent类程序），在大规模MoE部署（EP、PD分离）上投入较多，性能很强。
- **TensorRT-LLM**：NVIDIA官方，预先构建/编译优化引擎，深度kernel融合，FP8/FP4等低精度支持好，in-flight batching；在NVIDIA硬件上性能极致，但灵活性与上手成本相对高，常与Triton推理服务器配合。
- 其他：**llama.cpp/Ollama**（GGUF量化，CPU/消费级GPU/端侧）、**LMDeploy**、各云厂商自研引擎等。
- 选型思路：硬件平台、模型是否被支持（尤其新架构）、负载特征（长前缀复用、结构化输出、高并发）、量化需求、运维成本；**用自己的流量分布实测**，而不是只看公开benchmark。

Q24：***如何估算LLM服务的部署资源和成本？***

- **显存估算**：
    - 权重：$N_{params} \times \text{bytes}$（如70B BF16约140 GB，FP8约70 GB，INT4约35 GB）；
    - KV cache：用Q4公式乘以并发数 × 平均上下文长度；
    - 再预留激活、CUDA graph、通信缓冲等开销（约10%–20%）。
    - 例：Llama-3-70B FP8权重约70 GB，每token KV 320 KB，若要支持64个并发×8K上下文，KV约 $64 \times 8192 \times 320\text{KB} \approx 168$ GB，需多卡TP。
- **延迟估算**：
    - decode：$\text{TPOT} \approx \frac{\text{权重字节}/TP + \text{batch内KV字节}/TP}{\text{带宽} \times \text{利用率}}$ + 通信开销；
    - prefill：$\text{TTFT} \approx \frac{2 N_{params} n_{prompt}}{\text{算力} \times \text{MFU}}$ + 排队。
- **成本估算**：
$$\text{每百万token成本} = \frac{\text{GPU小时单价} \times \text{GPU数}}{\text{吞吐(token/s)} \times 3600} \times 10^6$$
    - 例（假设数字）：单卡每小时2美元、稳定吞吐2000 token/s，则约 $\frac{2}{2000 \times 3600}\times10^6 \approx 0.28$ 美元/百万token。
    - 输入与输出token分开算：prefill token便宜（并行、计算密集），decode token贵（访存密集），所以API通常输出单价远高于输入；缓存命中的输入更便宜。
- 降本杠杆：提高利用率（批处理、负载均衡、自动扩缩容）、量化、前缀缓存、投机解码、MoE/小模型路由（简单请求用小模型）、离线任务用批处理接口。

Q25：***线上LLM服务延迟过高，你会如何系统地排查和优化？***

- **先拆指标**：是TTFT高还是TPOT高？是P50高还是只有P99高？
    - TTFT高：排队（容量不足/调度不公平）、prompt过长、prefill被阻塞 → 扩容、前缀缓存、chunked prefill、PD分离、限制最大输入长度；
    - TPOT高：batch过大、KV显存不足导致抢占/重算、长上下文注意力占比高 → 调整最大batch/token预算、KV量化、GQA模型、提高TP度、投机解码；
    - P99抖动：长prefill插队、抢占、GC/CPU瓶颈（tokenizer、采样、结构化解码mask在CPU上）。
- **再看资源**：GPU利用率、显存（KV cache占用率、抢占次数）、带宽利用率、通信占比（profiling，如Nsight）。
- **模型侧**：更小模型/蒸馏、量化（weight-only给低并发，FP8给高吞吐）、MoE。
- **系统侧**：CUDA graph减少kernel launch开销、kernel融合、FlashAttention/FlashInfer、缓存感知路由。
- **应用侧**：流式输出降低感知延迟、限制输出长度、prompt精简并把稳定前缀放前面、并行化多次独立调用。
- 每一项优化都要在**质量评估 + 真实流量回放**下验证，避免以牺牲效果为代价换速度。
