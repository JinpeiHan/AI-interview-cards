# LLM推理优化

## 推理阶段与性能瓶颈

Q1：***LLM推理的prefill和decode阶段分别是什么？各自有什么特点？***

- **Prefill一次性并行处理整个prompt、写入KV cache并输出第一个token，是计算密集的；decode每步只生成一个token，却要读一遍全部权重和KV cache，是访存密集的。**
- Prefill本质是大矩阵乘（GEMM），决定首token延迟（TTFT），长prompt时注意力的二次方开销明显。
- Decode本质是矩阵-向量乘，每读一次权重只做很少计算，决定每token延迟（TPOT）；输出越长，decode占总时间越多。
- 两阶段特性不同，所以才有chunked prefill、prefill/decode分离部署等调度优化。

Q2：***什么是compute-bound和memory-bandwidth-bound？如何用算术强度（arithmetic intensity）判断？***

- **算术强度是"每搬运1字节数据做多少次计算"。它低于硬件的"算力/带宽"比值时，瓶颈在显存带宽；高于这个比值时，瓶颈在算力。**
- Roofline模型：可达性能取"峰值算力"和"算术强度×带宽"中的较小者。H100的拐点约为300 FLOPs/Byte。
- Decode的线性层算术强度约等于batch大小，所以增大batch几乎能"免费"提升吞吐，直到几百才吃满算力。
- Decode的注意力每份KV只被自己的query使用，强度约为1，batch摊不薄，始终受访存限制。
- 单流decode的延迟下界约为"权重+KV字节数÷带宽"，例如7B FP16模型约14GB，在H100上约4ms/token。
- 核心公式：$I = \frac{\text{FLOPs}}{\text{访存字节数}}$

Q3：***LLM服务常见的延迟/吞吐指标有哪些？它们之间如何权衡？***

- **核心指标是TTFT（首token延迟，含排队+prefill）、TPOT/ITL（相邻token间隔，即"打字速度"）、吞吐（每秒生成token数），以及只统计满足SLO请求的goodput。**
- 端到端延迟约等于TTFT加上TPOT乘以输出长度。
- batch越大吞吐越高，但每个请求的TPOT和排队时间变大。
- 把新请求的prefill插进decode批次能降低TTFT，但会让正在生成的请求出现卡顿（ITL尖刺）。
- 场景不同侧重不同：离线批处理看吞吐和成本，在线聊天看P99 TTFT/TPOT，Agent多轮看前缀缓存命中率。

## KV Cache及其显存优化

Q4：***为什么需要KV cache？KV cache的显存如何计算？***

- **生成时历史token的K、V不会变，缓存起来后每步只需为新token计算，避免每步重算整个前缀；代价是显存随层数、KV头数、序列长度和batch线性增长。**
- 例：Llama-2-7B（MHA）每token约512KB，4K上下文单序列约2GB；Llama-3-8B用GQA后每token只有128KB。
- 长上下文、大batch时KV cache往往比权重还大，成为并发数的主要限制。
- 优化方向：减少KV头数（MQA/GQA/MLA）、降精度（KV量化）、缩短长度（滑窗/驱逐）、减少碎片（PagedAttention）、复用（前缀缓存）。
- 核心公式：$\text{KV bytes} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{seq} \cdot \text{bytes} \cdot \text{batch}$（2代表K和V）

Q5：***MQA、GQA、MLA如何压缩KV cache？各有什么代价？***

- **三者都是减少需要缓存的K/V量：MQA让所有query头共享一组K/V，GQA让每组query头共享一组K/V，MLA把K/V压缩成一个低维潜向量只缓存它。**
- MQA：KV cache缩小为原来的 $1/n_h$，但质量有所下降、训练可能不稳。
- GQA：质量接近MHA、显存接近MQA，是当前主流（如Llama-3用8组），可由MHA检查点均值池化后少量继续训练得到。
- MLA（DeepSeek-V2）：只缓存低秩潜向量和一个小的解耦RoPE key，推理时把上投影矩阵吸收进Q和输出投影；压缩数十倍且效果不弱于MHA，代价是实现更复杂。
- 附带好处：一份KV被多个query头复用，也提高了decode注意力的算术强度。

Q6：***PagedAttention解决了什么问题？vLLM是如何管理KV cache的？***

- **它解决了KV cache按最大长度预分配连续显存带来的大量碎片和浪费；做法是借鉴操作系统的分页，把KV cache切成固定大小的块，按需分配、非连续存放。**
- 旧做法的显存实际利用率可能只有20%–40%（内部碎片、外部碎片、公共前缀重复存储）。
- 每个请求有一张block table记录逻辑块到物理块的映射，浪费只发生在最后一个块，注意力kernel按表间接寻址。
- 块带引用计数，多个序列可共享前缀，修改时写时复制。
- 显存不足时可抢占：把块换出到CPU或丢弃后重算。效果是能放下更大的batch，吞吐显著提升。

Q7：***什么是continuous batching？与static batching有什么区别？***

- **Static batching要等整批请求都生成完才放新请求进来，短请求结束后槽位空转；continuous batching以每个decode步为粒度调度，请求一结束就移出、新请求立刻加入。**
- 不同长度的序列可以同批：线性层把所有token拼成大矩阵一起算，注意力按请求分别算。
- GPU利用率和吞吐大幅提升、TTFT降低，是vLLM、SGLang、TensorRT-LLM的标配。
- 配套问题：新请求的prefill会抢占decode算力导致卡顿，常用chunked prefill把长prompt切块混入decode批次；还需要KV显存感知的准入和抢占策略。

Q8：***什么是前缀缓存（prefix caching / prompt caching）？如何实现？适用于哪些场景？***

- **很多请求共享相同的前缀（system prompt、few-shot示例、多轮历史、工具描述），把这段前缀的KV cache存下来复用，就能跳过这部分的prefill，显著降低TTFT和算力。**
- 实现：vLLM对每个block的前缀做哈希匹配；SGLang的RadixAttention用基数树组织缓存，LRU驱逐；也可分层存到CPU/SSD/分布式存储。
- 只能匹配完全相同的前缀，中间改一个token后面全部失效，所以prompt要把稳定内容放前、变化内容（如时间戳）放后。
- 调度时可做缓存感知路由；商业API常对命中缓存的输入token打折。

## 注意力计算优化

Q9：***FlashAttention的核心原理是什么？为什么能加速且省显存？***

- **标准注意力的瓶颈是把N×N的注意力矩阵反复写入和读出慢速显存（HBM）；FlashAttention把Q、K、V分块放进快速的片上SRAM，在片上算完整个注意力，从不把N×N矩阵写回显存。**
- 分块后无法一次拿到整行的softmax，就用online softmax维护运行中的最大值和求和，逐块修正，结果与标准注意力精确一致，不是近似。
- 反向传播时不存注意力矩阵，只存输出和每行的logsumexp，需要时在片上重算；FLOPs略增，但显存从 $O(N^2)$ 降到 $O(N)$。
- 它减少的是显存读写次数而不是计算量，因为注意力本身是访存受限的。

Q10：***FlashAttention各版本有什么改进？decode阶段用什么注意力kernel？***

- **每个版本都在更贴合硬件：FA2改进并行和工作划分，速度约为FA1的2倍；FA3针对H100做异步和低精度优化；decode阶段则用Flash-Decoding沿KV序列维切分并行。**
- FA1：分块 + online softmax + 重计算，按batch和head并行。
- FA2：减少非矩阵乘计算，额外在序列维并行，改进warp间分工。
- FA3：利用Hopper的TMA异步拷贝和WGMMA，生产者/消费者warp分工、softmax与矩阵乘重叠，支持FP8。
- Decode时query长度为1，按batch×head并行占不满GPU，Flash-Decoding把KV切段并行算部分结果再合并；FlashInfer等库还为分页KV、前缀共享提供高效kernel。

## 量化

Q11：***LLM推理量化有哪些形式？weight-only量化和W8A8有什么区别，分别适合什么场景？***

- **Weight-only（如W4A16）只压缩权重、计算时反量化回FP16，主要省显存和读带宽，适合访存受限的decode和小batch；W8A8把权重和激活都量化，用INT8/FP8 Tensor Core让算力翻倍，适合计算受限的prefill和大batch高吞吐。**
- 量化粒度可以是per-tensor、per-channel或per-group（如每128个元素一组）。
- Weight-only代表是GPTQ、AWQ；它不减少FLOPs，大batch时反量化开销可能抵消收益。
- W8A8的难点是激活中少数通道有极大的离群值，代表方案有SmoothQuant、LLM.int8()、FP8。
- 更激进的W4A4或1–2 bit质量风险更大；量化后要在数学、代码、长上下文等敏感任务上专门评估。

Q12：***GPTQ的原理是什么？***

- **GPTQ是逐层的训练后量化：用少量校准数据，逐列量化权重，每量化一列就利用二阶信息（Hessian）调整还没量化的权重来补偿误差，使该层输出尽量不变。**
- 思想来自OBS/OBQ，GPTQ的工程改进是所有行按同一列顺序量化以共享Hessian逆、批量惰性更新、用Cholesky分解保证数值稳定。
- 几个GPU小时就能把百亿到千亿级模型量化到3–4 bit，精度损失小；常配合group size 128和act-order。
- 核心公式（每层的目标）：$\min_{\hat W} \| W X - \hat W X \|_2^2$

Q13：***AWQ和SmoothQuant分别是怎么处理离群值的？***

- **两者都利用"激活除以s、权重乘以s，结果不变"的等价变换来搬运量化难度：SmoothQuant把激活的离群值平滑到权重上，让激活也能量化（W8A8）；AWQ放大重要通道的权重来保护它们（weight-only）。**
- SmoothQuant：激活离群值集中在少数通道、权重较平滑，用超参 $\alpha$（约0.5）平衡两边难度，缩放可离线融合进前一层，推理无额外开销。
- AWQ：约1%的显著权重通道最重要，而"显著"由对应激活的幅值决定；按通道放大这些权重降低相对误差，通过小网格搜索缩放系数，无需反向传播，比GPTQ更不易过拟合校准集。
- 一句话区分：SmoothQuant是为了量化激活，AWQ是为了保护权重。

Q14：***FP8推理有什么优势？KV cache量化怎么做？***

- **FP8在H100等新GPU上有原生Tensor Core支持，算力约为BF16的两倍、显存减半，而且浮点格式对离群值比INT8友好，精度接近BF16，几乎是W8A8的首选。KV cache量化则把KV存成FP8/INT8等低精度，让并发翻倍、decode注意力访存减半。**
- FP8有E4M3（精度高，常用于前向）和E5M2（范围大，常用于梯度）两种；细粒度的block-wise scale能进一步降误差。
- KV量化需要per-tensor/per-head/per-token的scale，kernel读取后反量化。
- 更激进的2–4 bit方案（如KIVI）发现Key按通道量化、Value按token量化效果更好，因为Key有固定的离群通道。
- 风险：长上下文检索类任务对KV精度敏感，需要专门评估。

## 解码加速

Q15：***投机解码（speculative decoding）的原理是什么？为什么能保证输出分布不变？***

- **用一个小草稿模型快速猜出几个token，再让大模型一次前向并行验证；因为decode是访存受限的，验证几个token和生成一个token耗时几乎一样，所以每次大模型前向能产出多个token。**
- 验证时按概率 $\min(1, p/q)$ 逐个接受草稿token；第一次拒绝时从"目标分布减去草稿分布"的残差分布重新采样并停止。
- 这套拒绝采样规则可以证明最终输出与直接从大模型采样完全同分布，是无损加速。
- 加速比取决于接受率、草稿长度和草稿模型的开销；大batch时系统已接近计算受限，收益下降；草稿和目标需共享tokenizer。

Q16：***Medusa、EAGLE等方法与经典投机解码有什么不同？***

- **经典方法要额外维护一个草稿模型；Medusa、EAGLE等让目标模型自己出草稿，只加很轻量的预测头，省去单独的模型且接受率更高。**
- Medusa：在最后的hidden state上加多个解码头，第k个头直接预测第t+k个token，用树形注意力一次验证多个候选；常配合放宽的"典型接受"，此时不再严格保分布。
- EAGLE：在特征层做自回归草稿，并输入上一步采样的token，接受率高于Medusa；EAGLE-2用动态草稿树，EAGLE-3融合多层特征。
- 其他：Lookahead decoding（无需草稿模型）、Prompt lookup（从输入里找n-gram，适合摘要和代码编辑）、MTP头（如DeepSeek-V3）。
- 共同点都是"目标模型并行验证"，用严格拒绝采样的方案可保证分布不变。

## 解码策略与约束生成

Q17：***常见的解码策略有哪些？temperature、top-k、top-p、重复惩罚分别如何作用？***

- **Greedy每步取概率最大的token，beam search保留多条高概率序列，采样类方法则先用temperature调整分布尖锐程度，再用top-k/top-p截掉低概率尾部后采样，重复惩罚用来压低已出现token的概率。**
- Greedy确定但易重复；beam search适合翻译等有标准答案的任务，开放生成时易单调，需要长度惩罚。
- Temperature小于1更保守，大于1更多样，趋于0等价于greedy。
- Top-k固定保留k个；top-p保留累积概率达到p的最小集合，大小自适应；min-p保留概率不低于最大概率一定比例的token，高温时更稳。
- 重复惩罚有repetition penalty（缩放已出现token的logit）、frequency/presence penalty（按次数或是否出现扣分）、no-repeat n-gram；常见处理顺序是惩罚 → temperature → 截断 → 采样。

Q18：***结构化/约束解码（structured decoding）如何实现？有什么坑？***

- **每一步根据当前语法状态算出合法token集合，把非法token的logit设为负无穷再采样，从而保证输出严格符合JSON Schema、正则或文法。**
- 正则/JSON Schema编译成有限状态机预先算好每个状态的合法token（如Outlines）；嵌套结构用下推自动机（如XGrammar，大部分mask可预计算）。
- SGLang的jump-forward：后续字符被语法唯一确定时直接跳过，不逐token生成。
- 坑：token边界和语法符号不对齐，要在字符层面匹配；强制mask会扭曲模型分布，可能语法正确但语义很差，可在prompt中给schema示例缓解；mask若在CPU上同步计算会拖慢大batch。

## 并行、MoE与系统架构

Q19：***推理中的张量并行（TP）是如何工作的？与流水线并行（PP）如何选择？***

- **TP把每一层的权重切到多张卡上一起算（MLP先按列再按行切，Attention按头切），每层做两次all-reduce；它让每卡读取的权重和KV变少，能直接降低单请求延迟。PP按层切到不同设备，不降低单请求延迟，但通信少、适合跨节点。**
- TP通信频繁，适合节点内NVLink；TP度过大时通信占比上升。
- TP度要能整除KV头数，否则需要复制KV头。
- PP需要多个micro-batch填满流水线。
- 其他：数据并行（多副本）扩展吞吐，专家并行用于MoE，上下文并行用于超长prompt的prefill。

Q20：***MoE模型的推理有哪些特点和挑战？***

- **MoE每个token只激活少数几个专家，所以计算量按激活参数算、很低，但显存要装下全部专家；主要挑战是访存效率、专家负载不均和跨卡all-to-all通信。**
- 例：DeepSeek-V3总参数671B，每token只激活37B。
- 小batch时每个专家分到的token很少，算术强度更低；需要更大batch才能发挥优势。
- 热门专家会成为瓶颈：训练时用负载均衡损失或偏置调节路由，推理时可复制热门专家。
- 专家并行通过all-to-all分发token，通信开销大，需要计算-通信重叠；同一专家的token用grouped GEMM批量计算。常见部署是Attention用TP/DP、MoE用EP。

Q21：***什么是prefill/decode分离（disaggregated serving）？为什么要这样做？***

- **把prefill和decode放到不同的GPU实例上，因为两者混跑会互相干扰（长prefill阻塞decode），而且它们的最优并行策略、batch大小和硬件配置都不同。**
- Prefill实例追求TTFT、计算密集；decode实例追求TPOT和大batch吞吐、访存密集；prefill完成后通过NVLink/RDMA把KV cache传给decode实例。
- 好处是两阶段可独立扩缩容、独立优化SLO，提升goodput（如DistServe、Splitwise、Mooncake）。
- 代价是KV传输开销和系统复杂度，短prompt场景收益有限；不分离时可用chunked prefill在单实例内折中。

Q22：***长上下文推理有哪些优化手段？***

- **长上下文的问题是prefill注意力随长度平方增长、KV cache随长度线性增长；优化思路是让每个token少看一些历史（滑窗/稀疏）、少存一些KV（驱逐/压缩/量化），以及在系统上切分和卸载。**
- 滑动窗口注意力：只看最近w个token，KV固定大小，常与全局注意力层交替以保留长程能力。
- StreamingLLM：保留开头几个"attention sink" token加最近窗口，就能稳定流式生成，但记不住中间内容。
- 稀疏注意力：固定模式（Longformer、BigBird）或动态选择（DeepSeek的NSA、DSA用轻量indexer为每个query选top-k历史token）。
- KV驱逐与压缩：H2O、SnapKV、KV量化、MLA；系统上用上下文并行/Ring Attention、分块prefill、KV卸载到CPU。也要关注位置编码外推（YaRN）和"大海捞针"评测，不能只看速度。

Q23：***主流的LLM推理框架有哪些？各自有什么特点？***

- **主流是vLLM（通用、生态最广）、SGLang（前缀缓存和结构化输出强、大规模MoE部署投入多）、TensorRT-LLM（NVIDIA官方、性能极致但上手成本高），端侧和CPU常用llama.cpp/Ollama。**
- vLLM：PagedAttention和continuous batching起家，模型与硬件覆盖广，OpenAI兼容API，是通用部署的默认选择。
- SGLang：RadixAttention、高效约束解码、适合多调用和Agent程序，在EP和PD分离上很强。
- TensorRT-LLM：预编译引擎、深度kernel融合、FP8/FP4支持好，常配合Triton服务器。
- 选型看硬件、模型支持、负载特征和运维成本，并用自己的真实流量实测，而不是只看公开benchmark。

Q24：***如何估算LLM服务的部署资源和成本？***

- **显存=权重（参数量×每参数字节）+ KV cache（每token大小×并发×上下文长度）+ 10%–20%的额外开销；延迟上decode看"读取字节÷带宽"，prefill看"计算量÷算力"；成本=GPU每小时费用÷每小时生成的token数。**
- 例：Llama-3-70B FP8权重约70GB，每token KV约320KB，64并发×8K上下文时KV约168GB，需要多卡TP。
- Decode TPOT约为（权重+KV字节）÷TP度÷有效带宽，prefill TTFT约为 $2 N_{params} n_{prompt}$ 除以有效算力。
- 输入token便宜（并行、计算密集）、输出token贵（访存密集），所以API的输出单价通常远高于输入，缓存命中的输入更便宜。
- 降本杠杆：提高利用率、量化、前缀缓存、投机解码、按难度路由到小模型、离线任务走批处理。
- 核心公式：$\text{每百万token成本} = \frac{\text{GPU小时单价} \times \text{GPU数}}{\text{吞吐(token/s)} \times 3600} \times 10^6$

Q25：***线上LLM服务延迟过高，你会如何系统地排查和优化？***

- **先拆指标定位是TTFT高还是TPOT高、是整体高还是只有P99抖动，再看GPU资源和profiling找瓶颈，然后从模型、系统、应用三层对症优化，每项改动都要用质量评估和真实流量回放验证。**
- TTFT高：排队、prompt过长、prefill被阻塞 → 扩容、前缀缓存、chunked prefill、PD分离、限制输入长度。
- TPOT高：batch过大、KV不足导致抢占重算、长上下文 → 调batch/token预算、KV量化、GQA模型、提高TP、投机解码。
- P99抖动：长prefill插队、抢占、CPU瓶颈（tokenizer、采样、约束解码mask）。
- 其他手段：量化、蒸馏小模型、CUDA graph、kernel融合、缓存感知路由；应用侧用流式输出、限制输出长度、稳定前缀放前面、并行化独立调用。
