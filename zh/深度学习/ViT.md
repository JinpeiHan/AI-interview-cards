# ViT

Q1：***简述ViT（Vision Transformer）的整体结构***

- ViT（Dosovitskiy et al., 2020，"An Image is Worth 16x16 Words"）把图像当作一串 token，直接使用标准的 **Transformer Encoder** 做图像分类。
- 流程：
    1. 将图像 $x\in\mathbb{R}^{H\times W\times C}$ 切成 $N=\frac{HW}{P^2}$ 个 $P\times P$ 的 **patch**；
    2. 每个 patch 展平后经过线性映射得到 $D$ 维 **patch embedding**；
    3. 在序列最前面拼接一个可学习的 **[class] token**，并加上 **位置编码**；
    4. 送入 $L$ 层 Transformer Encoder（Pre-LN：LayerNorm → MSA → 残差，LayerNorm → MLP → 残差）；
    5. 取 [class] token 的最终输出，经过 MLP head 得到分类结果。
- 常见配置：ViT-B（12层，768维，12头）、ViT-L（24层，1024维，16头）、ViT-H（32层，1280维，16头）；"ViT-B/16" 表示 Base 模型、patch 大小为 16。

Q2：***Patch Embedding 是怎么实现的？为什么等价于卷积？***

- 将每个 $P\times P\times C$ 的 patch 展平为 $P^2C$ 维向量，乘以可学习矩阵 $E\in\mathbb{R}^{(P^2C)\times D}$：$z_0=[x_{class};x_p^1E;\dots;x_p^NE]+E_{pos}$。
- 这等价于一个 **卷积核大小 = 步长 = $P$、输出通道为 $D$ 的卷积**，代码中通常就用 `Conv2d(C, D, kernel_size=P, stride=P)` 实现。
- 以 224×224 输入、$P=16$ 为例，得到 $14\times14=196$ 个 patch，加上 class token 共 197 个 token。
- **patch 越小**，token 越多，精度通常越高，但计算量按 $N^2$ 增长。

Q3：***[class] token 的作用是什么？能否不用？***

- [class] token 是一个可学习向量，本身不对应任何图像区域，通过自注意力**聚合所有 patch 的信息**，其最终输出作为整张图的表示，思路沿用自 BERT 的 [CLS]。
- 可以不用：对所有 patch token 的输出做 **全局平均池化（GAP）** 同样可行，ViT 原论文中两者效果接近，但需要调整学习率；DeiT III、MAE 微调等工作中也常用 GAP。
- 在 DINO 等自监督模型中，[class] token 的注意力图能很好地呈现前景物体的分割。

Q4：***ViT 使用什么样的位置编码？分辨率改变时如何处理？***

- ViT 默认使用 **可学习的一维绝对位置编码**，每个位置一个 $D$ 维向量。论文实验表明一维、二维、相对位置编码效果差别不大，学到的一维编码本身就呈现出二维的行列结构。
- 当微调或推理时分辨率改变（如 224 → 384），patch 大小不变，token 数量从 $14\times14$ 变为 $24\times24$，原有位置编码数量不匹配。
- 处理方法：把位置编码 reshape 成二维网格，按其在原图中的位置做 **二维插值（通常为双三次插值）** 到新的网格大小，class token 的位置编码单独保留。
- 这是 ViT 中为数不多手动引入的二维结构先验；也可以采用相对位置偏置（Swin）、条件位置编码（CPVT）或 RoPE 等更易外推的方案。

Q5：***ViT 与 CNN 的归纳偏置有何不同？***

- CNN 具有很强的 **归纳偏置**：**局部性**（卷积核只看局部邻域）、**平移等变性**（权重共享）、层级结构（逐步扩大感受野）。
- ViT 的归纳偏置很弱：只有 patch 切分和 MLP 是局部、平移等变的，自注意力是 **全局** 的，二维空间关系需要从数据中学习。
- 后果：
    - 数据少时，ViT 容易过拟合，表现不如同等规模的 CNN；
    - 数据足够大时，弱归纳偏置反而成为优势，模型上限更高，且从第一层开始就能建模全局依赖。
- 一句话总结：**归纳偏置是用先验换数据效率**。

Q6：***为什么 ViT 需要大量数据？有哪些缓解方法？***

- 原论文中，仅在 ImageNet-1k 上从头训练的 ViT 不如同等规模的 ResNet；在 ImageNet-21k、JFT-300M 等更大数据集上预训练后，ViT 才超过 CNN。原因是缺少归纳偏置，需要从数据中学习局部性与空间结构。
- 缓解方法：
    - **强数据增强与正则化**：RandAugment、Mixup、CutMix、Random Erasing、Stochastic Depth、Repeated Augmentation（DeiT 的做法）；
    - **知识蒸馏**：用 CNN 教师蒸馏（DeiT）；
    - **自监督预训练**：MAE、DINO、BEiT 等，摆脱对大规模标注数据的依赖；
    - **引入卷积先验**：混合模型、卷积 stem、局部窗口注意力（Swin）等。

Q7：***DeiT 做了哪些改进？distillation token 是什么？***

- DeiT（Touvron et al., 2021）证明 ViT **只用 ImageNet-1k** 也能训练得很好，关键在于训练策略：强数据增强、正则化以及精心调整的优化超参。
- **distillation token**：在 class token 之外再加一个可学习 token，它与其他 token 一起参与自注意力，但其输出用于拟合 **教师模型**（通常是 CNN，如 RegNet）的预测：
    - class token 的输出用真实标签计算交叉熵；
    - distillation token 的输出用教师预测计算损失，**硬蒸馏**（教师的 argmax 作为标签）效果优于软蒸馏。
- 推理时将两个 token 头的预测融合。
- 用 CNN 当教师效果更好，一般认为是 CNN 把自身的归纳偏置通过蒸馏"传授"给了 ViT。

Q8：***Swin Transformer 的核心思想是什么？***

- Swin Transformer（Liu et al., 2021）的目标是构建适用于检测、分割等密集预测任务的 **通用视觉骨干**，主要设计：
    - **层级结构**：通过 **Patch Merging**（相邻 2×2 patch 拼接后线性降维）逐级下采样，得到 1/4、1/8、1/16、1/32 的多尺度特征图，可直接接入 FPN 等结构；
    - **窗口注意力（W-MSA）**：只在不重叠的 $M\times M$ 局部窗口内做自注意力（默认 $M=7$），复杂度对图像大小呈 **线性**；
    - **移位窗口（SW-MSA）**：相邻两层交替使用常规窗口与平移 $\lfloor M/2\rfloor$ 的窗口，让信息跨窗口流动；实现上通过 **cyclic shift + attention mask** 高效计算；
    - **相对位置偏置**：在注意力 logits 上加可学习的相对位置偏置 $B$：$\text{softmax}(QK^T/\sqrt{d}+B)V$。
- 本质上是在 Transformer 中重新引入了局部性与层级结构等 CNN 先验。

Q9：***ViT 的计算复杂度是多少？Swin 是如何降低的？***

- 对于 $N$ 个 token、维度 $C$，全局自注意力的复杂度为 $O(N^2C)$，线性投影与 MLP 为 $O(NC^2)$。对于 $h\times w$ 的特征图，$N=hw$：

$$\Omega(\text{MSA})=4hwC^2+2(hw)^2C$$

- token 数随分辨率平方增长，注意力代价随分辨率 **四次方** 增长，这让高分辨率输入（检测、分割）代价很高。
- Swin 的窗口注意力：

$$\Omega(\text{W-MSA})=4hwC^2+2M^2hwC$$

- 当窗口大小 $M$ 固定时，复杂度对 $hw$ 是 **线性** 的。
- 其他降低复杂度的方法：空间降采样注意力（PVT 对 K/V 降采样）、token 剪枝/合并（DynamicViT、ToMe）、线性注意力、FlashAttention（不改变复杂度，但大幅减少显存访问开销）。

Q10：***什么是混合（Hybrid）模型？有什么好处？***

- **Hybrid ViT**：先用 CNN（如 ResNet 的前几个 stage）提取特征图，再把特征图的每个空间位置（或 1×1 的 patch）作为 token 送入 Transformer。ViT 原论文就包含 ResNet + ViT 的混合版本。
- 好处：CNN 提供局部特征与归纳偏置，在小模型、少数据的情况下更有优势；原论文中随模型增大，混合模型与纯 ViT 的差距缩小。
- 相关工作：用几层卷积替代 patchify 的 **卷积 stem**（训练更稳定、对优化器更不敏感）、CvT、CoAtNet（卷积与注意力分阶段堆叠）、ConViT 等。

Q11：***ViT 如何用于目标检测和语义分割？***

- **检测**：
    - DETR 用 CNN 骨干 + Transformer 编解码器，通过 object query 和二分图匹配直接预测集合，无需 NMS 和 anchor；
    - 用 Swin 等层级 ViT 替换 ResNet 作为 Mask R-CNN / Cascade R-CNN 的骨干；
    - **ViTDet**：直接使用 plain ViT（单尺度）作为骨干，用简单的特征金字塔（对最后一层特征做上/下采样），加少量全局注意力块配合窗口注意力，配合 MAE 预训练表现很好。
- **分割**：
    - SETR、Segmenter：ViT 编码器 + 上采样解码器；
    - DPT：将 ViT 多层的 token 重组为多分辨率特征图，用于深度估计与分割；
    - Mask2Former：基于 mask 分类的统一分割框架；SAM 使用 MAE 预训练的 ViT 作为图像编码器。
- 难点：高分辨率导致 token 数巨大；plain ViT 缺少多尺度特征。常用对策是窗口注意力、层级结构与特征金字塔。

Q12：***如何可视化 ViT 的注意力？***

- **原始注意力图**：取某一层某个头中 [class] token 对各 patch 的注意力权重，reshape 为 $\sqrt{N}\times\sqrt{N}$ 的网格，上采样后叠加到原图上。
- **Attention Rollout**（Abnar & Zuidema, 2020）：考虑残差连接，将每层注意力矩阵（多头平均）与单位矩阵相加并归一化，再逐层相乘：$\tilde{A}=\prod_l(0.5A_l+0.5I)$，追踪信息从输入 patch 流向输出 token 的路径。
- **基于梯度的方法**：Grad-CAM 的变体、结合梯度与相关性传播的 Transformer 可解释性方法（Chefer et al., 2021）。
- 注意：注意力权重并不等于特征重要性，只能作为参考；大模型的注意力图中可能出现背景区域的高范数"伪影"token，**Register token**（Darcet et al., 2023，"Vision Transformers Need Registers"）通过加入额外的可学习 token 缓解该问题。

Q13：***ViT 中不同层学到了什么？与 CNN 有何不同？***

- **平均注意力距离**：ViT 浅层中有些头关注局部、有些头已经关注全局，说明从第一层就能整合全局信息；随着层数加深，所有头的注意力距离都增大。
- CNN 的感受野随深度 **逐步** 增长，浅层只能看到局部。
- 相关分析研究（如 Raghu et al., 2021）发现 ViT 各层之间的表示更加一致，而残差连接在 ViT 中对传递信息作用更大。
- 线性映射层的滤波器呈现类似 Gabor 的基函数；位置编码之间的相似度呈现行列结构。

Q14：***CLIP 的图像编码器是如何使用 ViT 的？***

- CLIP（Radford et al., 2021）用 **图像-文本对比学习** 联合训练图像编码器与文本编码器，图像编码器可以是 ResNet 或 ViT（如 ViT-B/32、ViT-B/16、ViT-L/14）。
- ViT 版本的做法：patch embedding 加位置编码后多了一个 LayerNorm；取 [class] token 的输出，经过线性投影映射到与文本共享的嵌入空间，再做 L2 归一化；用余弦相似度与文本特征进行对比。
- 与有监督 ViT 的不同：监督信号来自自然语言而非固定类别，因此支持 **零样本分类**（把类别名填入 "a photo of a {class}" 模板，选相似度最高的文本）。
- CLIP 的 ViT 被广泛用作多模态大模型（如 LLaVA）的视觉编码器，通常取倒数第二层的 patch token 特征，通过投影层输入 LLM。

Q15：***ViT 的训练有哪些技巧？***

- 优化器：**AdamW** + 较大的权重衰减，配合 **学习率 warmup** 与余弦衰减。
- 正则化：Stochastic Depth（DropPath）、Label Smoothing、Mixup/CutMix、RandAugment。
- 梯度裁剪以稳定训练；大模型可使用 LayerScale（CaiT）、QK-Norm 等提升稳定性。
- 微调时常用 **更高分辨率** 并插值位置编码；分层学习率衰减（layer-wise lr decay）在自监督预训练模型的微调中很常见。

Q16：***ViT 与 CNN 在鲁棒性上有何差异？***

- 一些研究发现 ViT 相比 CNN 更偏向 **形状**（shape bias）而非纹理，对遮挡、patch 打乱等扰动更鲁棒。
- 但结论依赖于训练数据与增强策略：ConvNeXt（Liu et al., 2022）表明，把 Transformer 的训练配方与设计（大卷积核、LayerNorm、倒瓶颈等）迁移到纯 CNN 上，也能达到相近的性能，说明许多差距来自训练策略而非架构本身。
- 对抗鲁棒性方面两者都不具备天然优势，仍需对抗训练。
