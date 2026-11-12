# ViT

Q1：***简述ViT（Vision Transformer）的整体结构***

- **ViT 把图像切成固定大小的 patch，每个 patch 线性映射成一个 token，加上 [class] token 和位置编码后送进标准 Transformer Encoder，最后用 [class] token 的输出做分类。**
- 编码器是 Pre-LN 结构：LayerNorm → 多头自注意力 → 残差，LayerNorm → MLP → 残差。
- 常见配置：ViT-B（12 层、768 维）、ViT-L（24 层、1024 维）、ViT-H（32 层、1280 维）。
- "ViT-B/16" 表示 Base 模型、patch 大小为 16。

Q2：***Patch Embedding 是怎么实现的？为什么等价于卷积？***

- **把每个 patch 展平后乘一个可学习矩阵得到 D 维向量；这等价于卷积核大小和步长都等于 patch 大小、输出 D 通道的卷积，代码里通常直接用 `Conv2d(C, D, kernel_size=P, stride=P)`。**
- 每个 patch 只被处理一次、互不重叠，且所有 patch 共享同一个矩阵，这正是这种卷积的计算方式。
- 例：224×224 输入、patch 16，得到 14×14=196 个 patch，加 class token 共 197 个 token。
- patch 越小 token 越多，精度通常越高，但注意力计算量按 token 数平方增长。

Q3：***[class] token 的作用是什么？能否不用？***

- **[class] token 是一个不对应任何图像区域的可学习向量，通过自注意力聚合所有 patch 的信息，它的输出作为整张图的表示，思路来自 BERT 的 [CLS]。**
- 可以不用：对所有 patch 输出做全局平均池化（GAP）效果相近，只是学习率要调整；MAE 微调等工作常用 GAP。
- 在 DINO 等自监督模型中，[class] token 的注意力图能很好地分割出前景物体。

Q4：***ViT 使用什么样的位置编码？分辨率改变时如何处理？***

- **ViT 默认用可学习的一维绝对位置编码；分辨率变化时 token 数变了，就把位置编码排回二维网格，用双三次插值缩放到新网格大小，class token 的编码单独保留。**
- 论文发现一维、二维、相对位置编码效果差不多，学到的一维编码自己就呈现出行列结构。
- 例：224 → 384 时，token 网格从 14×14 变成 24×24。
- 也可以用更易外推的相对位置偏置（Swin）、条件位置编码或 RoPE。

Q5：***ViT 与 CNN 的归纳偏置有何不同？***

- **CNN 自带很强的归纳偏置（局部性、平移等变、层级感受野），ViT 几乎没有，二维空间关系要从数据里学。**
- 数据少时，ViT 容易过拟合，不如同规模 CNN。
- 数据足够多时，弱先验反而让上限更高，而且第一层就能建模全局依赖。
- 一句话：归纳偏置是用先验换数据效率。

Q6：***为什么 ViT 需要大量数据？有哪些缓解方法？***

- **因为 ViT 缺少归纳偏置，局部性和空间结构都要从数据里学：只在 ImageNet-1k 上从头训练不如 ResNet，在 ImageNet-21k、JFT-300M 上预训练后才超过 CNN。**
- 强数据增强和正则化：RandAugment、Mixup、CutMix、Stochastic Depth 等（DeiT 的做法）。
- 知识蒸馏：用 CNN 当教师（DeiT）。
- 自监督预训练：MAE、DINO、BEiT，不依赖大规模标注。
- 引入卷积先验：混合模型、卷积 stem、窗口注意力（Swin）。

Q7：***DeiT 做了哪些改进？distillation token 是什么？***

- **DeiT 证明 ViT 只用 ImageNet-1k 也能训好，关键是强数据增强、正则化和调好的超参，再加上一个用于蒸馏的 distillation token。**
- distillation token 和 class token 一样参与自注意力，但它的输出去拟合教师模型（通常是 CNN）的预测，class token 仍拟合真实标签。
- 用教师 argmax 做标签的硬蒸馏比软蒸馏更好；推理时融合两个头的预测。
- CNN 教师效果更好，一般认为是把 CNN 的归纳偏置"传授"给了 ViT。

Q8：***Swin Transformer 的核心思想是什么？***

- **Swin 在 Transformer 里重新引入了 CNN 的局部性和层级结构：只在局部窗口里做注意力，并逐级合并 patch 得到多尺度特征，从而成为适合检测、分割的通用骨干。**
- 层级结构：Patch Merging 把相邻 2×2 patch 合并降采样，得到 1/4 到 1/32 的多尺度特征图，可直接接 FPN。
- 窗口注意力：只在不重叠的 7×7 窗口内计算，复杂度对图像大小是线性的。
- 移位窗口：相邻层交替使用平移半个窗口的划分，让信息跨窗口流动，用循环移位加掩码高效实现。
- 注意力上加可学习的相对位置偏置。

Q9：***ViT 的计算复杂度是多少？Swin 是如何降低的？***

- **全局自注意力的代价随 token 数平方增长，而 token 数又随分辨率平方增长，所以高分辨率下代价极高；Swin 把注意力限制在固定大小的窗口里，复杂度就变成对 token 数线性。**
- 投影和 MLP 部分对 token 数本来就是线性的，瓶颈在注意力项。
- 其他降复杂度的方法：对 K/V 降采样（PVT）、token 剪枝/合并（ToMe）、线性注意力。
- FlashAttention 不改变复杂度，但大幅减少显存读写。
- 核心公式（$h\times w$ 特征图，窗口大小 $M$）：$\Omega(\text{MSA})=4hwC^2+2(hw)^2C$，$\Omega(\text{W-MSA})=4hwC^2+2M^2hwC$

Q10：***什么是混合（Hybrid）模型？有什么好处？***

- **混合模型先用 CNN（如 ResNet 前几个 stage）提取特征图，再把特征图的每个位置当作 token 送进 Transformer。**
- 好处是 CNN 提供局部特征和归纳偏置，在小模型、少数据时更有优势；模型变大后与纯 ViT 的差距缩小。
- 相关做法：用几层卷积替代 patchify 的卷积 stem（训练更稳定）、CvT、CoAtNet 等。

Q11：***ViT 如何用于目标检测和语义分割？***

- **主要有两条路：用 Swin 这类层级 ViT 直接替换检测/分割框架里的 CNN 骨干，或者像 ViTDet 那样用普通 ViT 再构造简单的特征金字塔。**
- 检测：DETR 用 Transformer 编解码器和 object query 直接预测目标集合，不需要 NMS 和 anchor；ViTDet 配合 MAE 预训练效果很好。
- 分割：SETR、Segmenter（ViT 编码器 + 上采样解码器），DPT 把多层 token 重组成多分辨率特征，Mask2Former 统一了分割任务，SAM 用 MAE 预训练的 ViT 做图像编码器。
- 难点是高分辨率下 token 太多、普通 ViT 没有多尺度特征，常用窗口注意力和特征金字塔应对。

Q12：***如何可视化 ViT 的注意力？***

- **最简单的方法是取 [class] token 对各 patch 的注意力权重，排成网格后上采样叠加到原图上。**
- Attention Rollout：考虑残差连接，把各层注意力矩阵加上单位矩阵后逐层相乘，追踪信息从输入 patch 到输出的流动。
- 也可以用基于梯度的方法，如 Grad-CAM 变体或结合相关性传播的方法。
- 注意力权重不等于特征重要性；大模型注意力图里常有背景处的高范数伪影 token，可用 Register token 缓解。

Q13：***ViT 中不同层学到了什么？与 CNN 有何不同？***

- **ViT 从第一层起就有部分注意力头关注全局，层数越深所有头的注意力距离越大；而 CNN 的感受野是随深度逐步扩大的，浅层只能看局部。**
- 研究发现 ViT 各层之间的表示更一致，残差连接在传递信息上作用更大。
- patch 映射层的滤波器类似 Gabor 基函数；位置编码之间的相似度呈现行列结构。

Q14：***CLIP 的图像编码器是如何使用 ViT 的？***

- **CLIP 用图文对比学习联合训练图像编码器和文本编码器；ViT 版本取 [class] token 的输出，线性投影到与文本共享的嵌入空间并 L2 归一化，再用余弦相似度和文本特征对比。**
- 监督信号来自自然语言而不是固定类别，所以能零样本分类：把类别名填进 "a photo of a {class}"，选相似度最高的那个。
- 结构上只在 patch embedding 加位置编码后多了一个 LayerNorm。
- CLIP 的 ViT 常被多模态大模型（如 LLaVA）当视觉编码器，通常取倒数第二层的 patch 特征经投影层输入 LLM。

Q15：***ViT 的训练有哪些技巧？***

- **核心配方是 AdamW 加较大权重衰减、学习率 warmup 加余弦衰减，再配合强正则化和数据增强。**
- 正则化：DropPath、Label Smoothing、Mixup/CutMix、RandAugment。
- 稳定性：梯度裁剪，大模型可用 LayerScale、QK-Norm。
- 微调时常用更高分辨率并插值位置编码；微调自监督模型时常用分层学习率衰减。

Q16：***ViT 与 CNN 在鲁棒性上有何差异？***

- **一些研究发现 ViT 更偏向形状而非纹理，对遮挡、patch 打乱等扰动更鲁棒，但这很大程度上取决于训练数据和增强策略。**
- ConvNeXt 表明，把 Transformer 的训练配方和设计迁移到纯 CNN 上也能达到相近性能，说明不少差距来自训练策略而非架构。
- 对抗鲁棒性上两者都没有天然优势，仍需对抗训练。
