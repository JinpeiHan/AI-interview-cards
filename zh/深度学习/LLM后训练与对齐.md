# LLM后训练与对齐

## 整体流程

Q1：***预训练（pretraining）和后训练（post-training）分别做什么？现代LLM的典型后训练流程是怎样的？***

- **预训练**：在数万亿token的无标注语料上做 next-token prediction，学到**知识与通用语言能力**，得到 base model；它只会"续写"，不会稳定地遵循指令、也没有安全边界。
- **后训练**：用少得多的数据（通常是百万级样本以内）塑造**行为**：遵循指令、对话格式、有用性、诚实、无害、推理能力、工具使用等。一个常见观点是"知识主要来自预训练，后训练主要是**激发与格式化**"（Superficial Alignment Hypothesis，LIMA 2023）。
- 典型流程（InstructGPT 范式及其演进）：
    1. **SFT**：在高质量的（prompt, response）示范上做监督微调；
    2. **偏好优化**：RLHF（奖励模型 + PPO）或 DPO 等离线方法；
    3. **推理/可验证奖励 RL**（RLVR，如 GRPO）：在数学、代码等有可自动判定答案的任务上做大规模 RL；
    4. 安全微调、拒答校准，以及多轮迭代（rejection sampling → SFT → 偏好优化 → ……）。
- 中间常有 **mid-training / 退火阶段**：在预训练末期提高高质量数据、长上下文、代码数学数据的比例，衔接预训练与后训练。

## SFT与指令微调

Q2：***SFT中数据质量和数量哪个更重要？***

- **质量优先**：LIMA 用约 1000 条精选样本就获得了不错的对话能力；大量低质量数据会让模型学到啰嗦、错误、幻觉的风格。
- 质量维度：**正确性**、**多样性**（任务类型、领域、难度、长度、语言）、**格式一致性**、回答风格是否符合目标（详略、语气）、无重复/无污染（与评测集去重）。
- 数量仍有作用：要覆盖更多技能（代码、数学、工具调用、多轮）和长尾任务时，需要规模化——现代模型的 SFT 集通常是数十万到数百万条，但会经过严格过滤（LLM 打分、去重、难度筛选、执行验证）。
- 追问点：SFT 学的是"模仿"，若示范超出模型已有知识，可能**诱发幻觉**（模型学会了"自信地编造"）。

Q3：***什么是chat template？为什么很重要？***

- chat template 把多轮对话序列化成模型训练时见过的固定格式，用**特殊token**区分角色和边界，例如：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
2<|im_end|>
```

- 重要性：
    - 训练和推理**必须使用相同模板**，否则性能明显下降（停不下来、角色混乱）；
    - **EOS / 回合结束 token** 让模型学会何时停止；
    - 角色分隔是**指令层级**（system > user > tool output）和抵御 prompt injection 的基础；
    - 工具调用、思维链（如 `<think>` 块）也通过模板中的特殊结构表示。
- 工程上通常用 tokenizer 自带的 `apply_chat_template`，避免手写拼接出错。

Q4：***SFT中为什么要对prompt部分做loss masking？***

- SFT 的目标是学习 $p(y\mid x)$，而不是 $p(x)$。损失只在 response token 上计算：

$$\mathcal{L}_{SFT} = -\sum_{t \in \text{response}} \log \pi_\theta(y_t \mid x, y_{<t})$$

- 实现上把 prompt 位置的 label 设为 `-100`（PyTorch `CrossEntropyLoss` 的 `ignore_index`）。
- 原因：
    - 避免模型浪费容量去学习"生成用户问题"，甚至在推理时替用户说话；
    - 用户输入可能质量低、带噪声，不应被模仿；
    - 多轮对话中通常只对**所有 assistant 回合**计算损失（system/user/tool 消息都 mask）。
- 追问：也有工作发现对短回答数据保留部分 prompt loss 有轻微正则作用，但主流做法仍是 mask。另外还要注意**按 token 平均还是按样本平均**——在 packing 多样本时会影响长/短回答的权重。

Q5：***指令微调（instruction tuning）和SFT有什么关系？为什么能提升零样本泛化？***

- 指令微调是 SFT 的一种：把大量 NLP 任务改写成"自然语言指令 + 输出"的形式进行多任务微调（FLAN、T0、Super-NaturalInstructions 等）。
- 关键发现：**任务数量与多样性**越大，对未见任务的零样本泛化越好；混入 CoT 示例可以保留推理能力；模型越大收益越明显。
- 与对话式 SFT 的区别：早期指令微调偏"学术任务格式"，回答短；对话式 SFT（如 InstructGPT、ShareGPT 风格数据）更强调开放式、长回答和多轮交互。现代做法是两者混合。

## RLHF与PPO

Q6：***描述RLHF的完整流程，奖励模型如何训练（Bradley-Terry损失）？***

- 流程：SFT 模型 $\pi_{SFT}$ → 采样同一 prompt 的多个回答 → 人类标注偏好对 $(y_w \succ y_l)$ → 训练奖励模型 $r_\phi$ → 用 PPO 优化策略。
- **Bradley-Terry 模型**假设：

$$P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)$$

- 奖励模型损失（负对数似然）：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)}\Big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]$$

- 结构：通常以 SFT 模型初始化，去掉 LM head，换成输出标量的线性头，取最后一个 token 的隐状态。
- 要点：
    - 奖励只有**相对差值**有意义（平移不变），因此常做归一化；
    - 若一个 prompt 有 $K$ 个排序回答，可构造 $\binom{K}{2}$ 个对（InstructGPT 的做法），同一 prompt 的对放在一个 batch 防过拟合；
    - 人类标注一致率通常只有 60%–80% 左右，RM 准确率上限受此限制。

Q7：***PPO阶段的优化目标是什么？为什么要加KL惩罚？***

- 目标：

$$\max_{\pi_\theta} \ \mathbb{E}_{x\sim D,\, y\sim \pi_\theta(\cdot|x)}\big[r_\phi(x,y)\big] - \beta\, \mathbb{D}_{KL}\big[\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)\big]$$

  其中 $\pi_{ref}$ 通常是冻结的 SFT 模型。
- 实现上常把 KL 作为**逐 token 奖励**：$r_t = -\beta \log\frac{\pi_\theta(y_t|\cdot)}{\pi_{ref}(y_t|\cdot)}$，最后一个 token 再加上 RM 分数。
- 为什么要 KL：
    - **防止 reward hacking**：RM 只在 SFT 分布附近可靠，策略跑远后会利用 RM 的漏洞（重复、谄媚、超长回答）；
    - **保持语言流畅性与多样性**，避免模式坍塌（mode collapse）；
    - **保留预训练能力**，减轻对齐税；
    - 理论上，带 KL 的目标有闭式最优解 $\pi^* \propto \pi_{ref}\exp(r/\beta)$，即在先验上做"贝叶斯式"重加权——这也是 DPO 的出发点。
- $\beta$ 太小 → 奖励黑客；太大 → 学不动。可用自适应 KL 控制目标 KL 值。

Q8：***PPO用于LLM时有哪些具体细节？（四个模型、GAE、clipping）***

- **四个模型**：actor（策略，训练）、critic（价值模型，训练）、reference（冻结）、reward model（冻结）——显存开销大是 PPO 的主要痛点。
- **MDP 视角**：状态 = prompt + 已生成 token，动作 = 下一个 token；奖励稀疏（只在序列末尾有 RM 分数）+ 每 token 的 KL 惩罚。
- **价值模型与 GAE**：critic 估计 $V(s_t)$，用 GAE 计算优势：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\qquad \hat A_t = \sum_{l\ge 0} (\gamma\lambda)^l\, \delta_{t+l}$$

  LLM 中常取 $\gamma=1$，$\lambda$ 接近 1（如 0.95）。
- **Clipped 目标**：

$$\mathcal{L}^{CLIP} = \mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big],\quad \rho_t=\frac{\pi_\theta(y_t|s_t)}{\pi_{\theta_{old}}(y_t|s_t)}$$

  $\epsilon$ 常取 0.2，限制每次更新幅度，允许对同一批 rollout 做多个 epoch。
- 其他技巧：价值函数也做 clipping、优势归一化、奖励归一化/裁剪、critic 用 RM 初始化、混入预训练损失（PPO-ptx）防遗忘。
- 训练循环：rollout（生成，最耗时）→ 打分与计算优势 → 若干 mini-batch 更新。

## DPO及变体

Q9：***DPO的推导思路和损失函数是什么？***

- 从 KL 约束的 RLHF 目标出发，最优策略有闭式解：

$$\pi^*(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\Big(\frac{1}{\beta}r(x,y)\Big)$$

- 反解出奖励（**隐式奖励**）：

$$r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)$$

- 代入 Bradley-Terry，配分函数 $Z(x)$ 在两回答相减时**抵消**，得到只依赖策略的损失：

$$\mathcal{L}_{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big]$$

- 直觉："你的语言模型其实是一个奖励模型"——提高 chosen 相对参考模型的概率、降低 rejected 的概率；梯度中带有权重 $\sigma(\hat r_l - \hat r_w)$，即对**当前排错的样本**更新更多。

```python
def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # 输入均为整个回答的对数概率之和（只对response token求和）
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -torch.nn.functional.logsigmoid(logits).mean()
```

Q10：***DPO相比PPO有哪些优缺点？***

- 优点：
    - **简单稳定**：无需训练 RM、无需在线采样、无需 critic，本质是一个分类损失；
    - **省资源**：只需策略 + 参考模型（参考模型的 logprob 甚至可预计算）；
    - 超参少（主要是 $\beta$），易复现。
- 缺点：
    - **离线/off-policy**：偏好数据来自其他模型时存在分布偏移，效果通常不如在线方法；实践中用 **online/iterative DPO**（用当前策略采样、用 RM 或 judge 打标）缓解；
    - **chosen 概率也可能下降**：DPO 只约束差值，两者的似然可能同时下降，导致输出偏移到未见过的回答；
    - 容易**过拟合**偏好数据（偏好近乎确定时，隐式奖励差趋向无穷，KL 约束失效）；
    - 有**长度偏置**：倾向变长；
    - 无法利用 RM 对新采样数据的泛化，也不适合可验证奖励的推理 RL。
- 经验结论：在资源充足时，**在线 RL（PPO/GRPO）上限更高**；DPO 适合快速迭代与中小规模对齐。

Q11：***简述IPO、KTO、ORPO、SimPO这些DPO变体。***

- **IPO**（Identity PO）：把 log-sigmoid 换成平方损失，让隐式奖励差回归到固定间隔 $\frac{1}{2\beta}$，解决偏好确定时 DPO 过拟合的问题：$\big(h_\theta - \frac{1}{2\beta}\big)^2$，其中 $h_\theta$ 是 DPO 中的对数比差。
- **KTO**（Kahneman-Tversky Optimization）：基于前景理论，**不需要成对数据**，只需每个回答"好/坏"的二元标签；对收益和损失采用不对称权重。适合点赞/点踩类数据。
- **ORPO**（Odds Ratio PO）：**无需参考模型、单阶段**，在 SFT 损失上加一个 odds ratio 项 $-\log\sigma\big(\log\frac{odds(y_w)}{odds(y_l)}\big)$，同时完成 SFT 与偏好对齐。
- **SimPO**：**无参考模型**，用**长度归一化**的平均对数概率作为隐式奖励，并加入目标间隔 $\gamma$：

$$\mathcal{L} = -\log\sigma\Big(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\Big)$$

  缓解长度偏置，且与生成时的打分方式一致。
- 共同主题：去掉参考模型（省显存）、处理长度偏置、放宽数据要求、防过拟合。

## GRPO与可验证奖励

Q12：***GRPO是什么？和PPO有什么区别？***

- GRPO（Group Relative Policy Optimization，DeepSeekMath 2024 提出，DeepSeek-R1 中广泛使用）**去掉了 critic**：对每个 prompt 采样一组 $G$ 个回答，用组内奖励的标准化作为优势：

$$\hat A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$

  该优势赋给回答 $i$ 的所有 token。
- 目标仍用 PPO 式的 clipped ratio，并把 KL 惩罚**直接加到损失中**（而非放进奖励），常用无偏低方差估计 $\frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1$。
- 优点：省掉与策略同规模的价值模型（显存、计算大减）；组内基线天然适配**稀疏的序列级奖励**（如对/错）；实现简单。
- 缺点与改进：
    - 一组全对或全错时优势为 0，**无学习信号**——DAPO 用动态采样过滤这类组；
    - 除以 std 会放大难度极端题的权重、按回答长度归一化引入**长度偏置**——Dr. GRPO 去掉了这些归一化；
    - DAPO 还提出 clip-higher（非对称裁剪上界，防止熵坍塌）、token 级损失聚合、超长回答惩罚等。
- 相关的无 critic 方法：RLOO（留一法基线）、REINFORCE++、ReMax。

Q13：***什么是RLVR（基于可验证奖励的强化学习）？为什么推动了推理模型的发展？***

- RLVR：奖励来自**可程序化验证**的规则，而不是学习出来的 RM：
    - 数学：最终答案与标准答案匹配（如提取 `\boxed{}` 内容并做符号等价判断）；
    - 代码：通过单元测试；
    - 格式、指令约束检查等。
- 好处：奖励**难以被 hack**（相比神经网络 RM）、可无限扩展、无需人工标注；可以长时间、大规模地训练 RL。
- 代表：OpenAI o1（2024）开启推理模型范式；DeepSeek-R1（2025）公开展示：在 base 模型上直接做纯 RL（R1-Zero）就能自发涌现长思维链、自我验证、反思（"aha moment"），回答长度随训练增长。
- R1 的完整流程：冷启动长 CoT SFT → 推理 RL（GRPO + 规则奖励）→ 拒绝采样生成新 SFT 数据 + 通用数据 → 全场景 RL（规则奖励 + 偏好 RM）。
- 追问：有争论认为 RLVR 主要是**提高采样效率**（把 pass@k 中已有的正确路径集中到 pass@1），而非扩展模型能力边界；大 $k$ 时 base 模型的 pass@k 有时不逊于 RL 后模型。另需注意验证器的假阳性（如答案解析漏洞、测试不充分）。

Q14：***过程奖励模型（PRM）和结果奖励模型（ORM）有什么区别？***

- **ORM**：只对最终结果打分（对/错或整体质量）。标注便宜，但**信用分配**困难，可能奖励"推理错但答案碰巧对"的过程。
- **PRM**：对推理的**每一步**打分，提供稠密信号，能定位第一个错误步骤。OpenAI 的 "Let's Verify Step by Step"（2023）表明在 best-of-N 重排序中 PRM 优于 ORM。
- PRM 的数据来源：人工逐步标注（PRM800K，昂贵）；或 **Monte Carlo 估计**（Math-Shepherd 式：从某一步出发多次 rollout，看最终正确率作为该步的价值）。
- 实践问题：步骤切分不明确、标注噪声大、在 RL 中作为奖励易被 hack（例如把步骤拆得很碎来刷分）。DeepSeek-R1 报告称没有采用 PRM，主流推理 RL 仍以**结果奖励 + 规则验证**为主；PRM 更多用于测试时搜索/重排序。

## 奖励黑客与AI反馈

Q15：***什么是reward hacking？常见表现和缓解方法有哪些？***

- 定义：策略找到使代理奖励（RM 分数）升高、但真实目标（人类偏好）没有提升甚至下降的方式——**Goodhart 定律**。代理奖励持续上升而真实质量先升后降（"overoptimization"，Gao et al. 2022 的缩放规律研究）。
- 常见表现：
    - **长度黑客**：回答越来越长、堆砌列表与免责声明；
    - **谄媚（sycophancy）**：附和用户错误观点；
    - 重复模板化措辞、过度自信；
    - RLVR 中：钻验证器漏洞，如修改/跳过单元测试、输出能骗过答案解析器的格式、写特判代码。
- 缓解：
    - **KL 惩罚**、早停（以人工评估或留出 judge 为准）；
    - **RM 集成**、不确定性惩罚、对奖励做长度惩罚或长度去偏；
    - 持续用新策略的样本**迭代更新 RM**；
    - 更鲁棒的验证器（沙箱、隐藏测试、禁止修改测试文件）；
    - 约束与多目标奖励（如有用性 RM + 安全 RM 分开）；
    - 监控奖励分布、回答长度、熵等指标，人工抽查高奖励样本。

Q16：***什么是RLAIF和Constitutional AI？***

- **RLAIF**：用 AI（通常是强 LLM）代替人类给出偏好标签，然后照常训练 RM 或做 DPO。成本低、可扩展、一致性高；风险是继承 judge 模型的偏见。
- **Constitutional AI**（Anthropic，2022）：给定一套书面原则（"宪法"），分两阶段：
    1. **监督阶段（critique & revision）**：让模型对有害请求生成回答 → 根据某条原则自我批评 → 修改 → 用修改后的回答做 SFT；
    2. **RL 阶段（RLAIF）**：模型根据原则比较两个回答哪个更好，生成偏好数据训练偏好模型，再做 RL。
- 好处：减少人工标注有害内容的负担；规则**透明、可审计**；能在"无害"和"有用"之间取得更好的平衡（解释性拒绝而非简单回避）。

## 数据与蒸馏

Q17：***什么是拒绝采样（rejection sampling）/ best-of-N？在后训练中怎么用？***

- **Best-of-N（推理时）**：采样 $N$ 个回答，用 RM 或验证器选分数最高的。简单而强，是衡量 RL 效果的强基线；相对参考策略的 KL 约为 $\log N - \frac{N-1}{N}$。代价是推理成本 $\times N$。
- **拒绝采样微调（训练时，RFT / Rejection Sampling Fine-Tuning）**：对每个 prompt 采样多个回答，只保留通过验证或 RM 得分最高的，用它们做 SFT；迭代多轮（Llama 2、STaR、ReST 等）。
- 优点：实现简单、稳定（只需 SFT 基础设施）；可视为 on-policy 数据上的"离散版"策略改进（Expert Iteration）。
- 缺点：只学正样本，不利用负样本信息；难题上可能采不到正确样本；效果上限一般低于在线 RL。

Q18：***如何从更强模型蒸馏？有哪些注意点？***

- **黑盒/序列级蒸馏**：用强模型（teacher）生成回答（含长 CoT），学生在其上做 SFT。DeepSeek-R1 的蒸馏版本表明：对于小模型，**直接蒸馏强推理模型的输出往往比在小模型上直接做 RL 更有效**。
- **白盒/logit 蒸馏**：最小化学生与教师 token 分布的 KL（需要同 tokenizer 和教师 logits），信息量更大；**on-policy 蒸馏**让学生自己生成、教师逐 token 打分（反向 KL），缓解训练/推理分布不一致（exposure bias）。
- 注意点：
    - 教师错误与幻觉会被继承，需要过滤（验证答案、去重、质量打分）；
    - **能力差距过大**时学生学不会，可用中间规模教师或课程学习；
    - 只模仿风格而无真实能力（"false promise of imitating proprietary LLMs"，2023）；
    - 许可/服务条款问题：部分商业模型禁止用其输出训练竞品。

Q19：***合成数据在后训练中如何使用？有哪些风险？***

- 用途：
    - **指令生成**：Self-Instruct、Evol-Instruct（逐步增加难度与复杂度）、基于种子文档/人设（persona）生成多样化 prompt；
    - **回答生成 + 过滤**：强模型生成，用验证器、执行反馈或 LLM judge 过滤；
    - 偏好对构造：同一 prompt 多次采样，按 RM / judge / 规则判定 chosen 和 rejected；
    - 推理数据：大规模合成题目 + 可验证答案；工具调用与智能体轨迹的模拟环境。
- 风险：
    - **多样性下降与模型坍塌**（递归地在自身输出上训练导致分布尾部丢失）；
    - 错误累积、幻觉传播；
    - 评测污染（合成数据与 benchmark 相似）；
    - 风格单一（"AI 腔"）。
- 缓解：保留真实人类数据、去重与多样性控制、严格验证、混合多个生成模型。

## 遗忘、对齐税与安全

Q20：***后训练中的灾难性遗忘和"对齐税"（alignment tax）是什么？怎么缓解？***

- **灾难性遗忘**：在窄分布上微调后，模型在其他任务上的能力下降（如过度 SFT 后数学或多语言能力变差）。
- **对齐税**：对齐后在某些标准 benchmark 上的性能下降（InstructGPT 观察到在部分 NLP 任务上退化），以及创造性/多样性下降、校准变差（RLHF 后模型的概率校准通常变差）。
- 缓解：
    - **数据混合 / replay**：混入通用指令数据和少量预训练数据（PPO-ptx）；
    - **KL 约束**到参考模型；较小学习率、较少 epoch；
    - **参数高效微调**（LoRA，"learns less and forgets less"）；
    - **模型合并**（model merging / weight averaging，如 SFT 与 RL 模型权重插值、WiSE-FT）；
    - 多任务联合训练、分阶段训练后在综合评测集上做 checkpoint 选择。
- 观察：在线 RL 相比 SFT 通常**遗忘更少**，因为它在策略自身分布上更新、改变更"保守"。

Q21：***安全微调如何做？如何避免过度拒答（over-refusal）？***

- 安全微调方法：
    - 安全 SFT 数据（有害请求 + 合适的拒绝或安全完成）；
    - **单独的安全 RM** 或多目标奖励（Llama 2 用 helpfulness RM 和 safety RM，根据 prompt 是否敏感选择）；
    - 红队测试（人工 + 自动）发现漏洞后补充数据；
    - Constitutional AI / 规则奖励（如 OpenAI 的 Rule-Based Rewards、Deliberative Alignment：让模型在推理中显式引用安全规范）。
- **过度拒答**：对看起来敏感但实际无害的请求也拒绝（如"怎么杀死一个 Python 进程"）。可用 XSTest、OR-Bench 等测评。
- 缓解：
    - 训练数据中加入大量"**看似有害实则无害**"的边界样本及正确的有帮助回答；
    - 从"硬拒绝"转向 **safe completion**：在安全范围内尽量提供有用信息、解释拒绝原因；
    - 同时优化有用性与无害性，并在评测中同时跟踪**有害率与误拒率**这两个指标；
    - 根据上下文（系统提示、用户意图）而非关键词做判断。
- 追问：越狱（jailbreak）、多轮攻击、prompt injection 需要对抗训练和指令层级训练；安全对齐可被少量微调轻易移除（"shallow safety alignment"），这对开源权重模型尤为重要。

## 评估

Q22：***如何评估对齐后的模型？LLM-as-judge 有哪些偏差？***

- 评估方式：
    - **人工评估**：成对比较或打分，金标准但昂贵、慢；
    - **胜率（win rate）**：与基线模型成对比较的胜/平/负比例；Chatbot Arena 用大量众包成对投票拟合 **Elo / Bradley-Terry 分数**；
    - **LLM-as-judge**：MT-Bench（打分）、AlpacaEval（对参考模型的胜率）、Arena-Hard；
    - 能力 benchmark：MMLU 类知识、数学（AIME、MATH）、代码（LiveCodeBench、SWE-bench）、指令遵循（IFEval，可程序化验证）、安全与拒答测试集。
- LLM-as-judge 的偏差：
    - **位置偏差**：偏好第一个（或第二个）回答 → 交换顺序评两次取一致结果；
    - **长度/冗长偏差**：偏好更长回答 → 使用长度控制的胜率（如 AlpacaEval 2 LC）；
    - **自我偏好**：偏好与自己风格相近或自己生成的回答；
    - 格式偏差（偏好 markdown 列表）、权威与语气偏差、对数学/代码正确性的判断不可靠；
    - 缓解：给出评分细则（rubric）和参考答案、多个 judge 集成、与人工标注做一致性校验。
- 追问：注意**数据污染**、benchmark 过拟合；Arena 风格评分也可能被"风格"刷分（因此有 style control 的版本）。

## 长链推理与测试时计算

Q23：***长链推理（long CoT）模型是如何训练的？***

- 核心：让模型在给出答案前生成很长的**思考过程**（探索、回溯、自我验证），并通过 RL 让这种行为得到强化。
- 常见流程：
    1. **冷启动 SFT**：少量高质量长 CoT 数据（人工整理或从强推理模型蒸馏），确定格式（如 `<think>...</think>` 后接最终答案）并提高可读性；
    2. **大规模 RLVR**：在数学、代码、科学、逻辑题上用 GRPO/PPO 及规则奖励；按难度筛选题目（太易或太难的题都没有学习信号）；
    3. 拒绝采样 + 通用 SFT，混合非推理数据以保持通用能力；
    4. 通用 RL 对齐偏好与安全。
- 训练中的现象与技巧：回答长度随训练增长；要防止**熵坍塌**（clip-higher、熵正则）；对过长回答做惩罚或长度课程（逐步放大上下文长度）；语言混杂可用语言一致性奖励。
- 小模型：通常通过**蒸馏**大推理模型的轨迹获得长推理能力，比直接 RL 更高效。

Q24：***什么是测试时计算（test-time compute）扩展？有哪些方式？***

- 思想：除了扩大训练计算，还可以在**推理时**花更多计算换取更好结果；o1 显示性能随"思考长度"对数线性地提升。
- 方式：
    - **串行扩展**：更长的思维链（推理模型）、自我修正、多轮反思；可用"思考预算"控制（如强制截断或通过追加 "Wait" 延长思考的 budget forcing，s1 2025）；
    - **并行扩展**：多次采样 + **多数投票（self-consistency）**、best-of-N（用 ORM/PRM/验证器挑选）；
    - **搜索**：基于 PRM 的 beam search、MCTS、tree-of-thoughts 等。
- 权衡：
    - 简单题上过度思考（overthinking）浪费计算，难题上思考不足（underthinking）；需要**自适应计算**（按难度分配预算，或混合思考/非思考模式）；
    - 并行方法依赖可靠的验证器；多数投票只适用于答案可比对的任务；
    - 对计算最优的研究表明：在一定预算下，小模型 + 更多测试时计算有时可以胜过大模型，但对最难的问题仍需要更强的基础模型。

Q25：***如果让你从一个base模型出发做后训练，你会如何设计实验与迭代流程？***

- **数据**：定义目标能力与行为规范 → 收集/合成 SFT 数据（多样性、质量过滤、与评测集去重）→ 偏好数据（人工 + AI 反馈）→ 可验证任务的题库与验证器。
- **训练**：SFT（正确的 chat template、loss masking、少 epoch）→ 偏好优化（DPO 快速迭代，或 RM + PPO/GRPO 在线 RL）→ RLVR 提升推理 → 安全调优。
- **评测**：建立覆盖知识、推理、代码、指令遵循、对话质量、安全/误拒、长上下文的**内部评测套件**；结合 LLM judge 与人工抽检，监控对齐税。
- **迭代**：分析失败案例 → 定向补数据 → 再训练；每轮用最新模型重新采样偏好数据和更新 RM（on-policy 迭代），并做 checkpoint 选择或模型合并。
- 消融关注点：数据配比、KL 系数 $\beta$、学习率、采样温度与组大小、奖励长度去偏。
