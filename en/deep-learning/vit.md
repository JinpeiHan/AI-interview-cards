# ViT

Q1: ***Describe the overall architecture of ViT (Vision Transformer)***

- ViT (Dosovitskiy et al., 2020, "An Image is Worth 16x16 Words") treats an image as a sequence of tokens and applies a standard **Transformer Encoder** directly to image classification.
- Pipeline:
    1. Split the image $x\in\mathbb{R}^{H\times W\times C}$ into $N=\frac{HW}{P^2}$ **patches** of size $P\times P$;
    2. Flatten each patch and apply a linear projection to obtain a $D$-dimensional **patch embedding**;
    3. Prepend a learnable **[class] token** to the sequence and add **position embeddings**;
    4. Feed the sequence through $L$ Transformer Encoder layers (Pre-LN: LayerNorm → MSA → residual, LayerNorm → MLP → residual);
    5. Take the final output of the [class] token and pass it through an MLP head to get the classification result.
- Common configurations: ViT-B (12 layers, 768 dim, 12 heads), ViT-L (24 layers, 1024 dim, 16 heads), ViT-H (32 layers, 1280 dim, 16 heads); "ViT-B/16" means the Base model with patch size 16.

Q2: ***How is patch embedding implemented? Why is it equivalent to a convolution?***

- Each $P\times P\times C$ patch is flattened into a $P^2C$-dimensional vector and multiplied by a learnable matrix $E\in\mathbb{R}^{(P^2C)\times D}$: $z_0=[x_{class};x_p^1E;\dots;x_p^NE]+E_{pos}$.
- This is equivalent to a **convolution with kernel size = stride = $P$ and $D$ output channels**; in code it is usually implemented as `Conv2d(C, D, kernel_size=P, stride=P)`.
- For a 224×224 input with $P=16$, we get $14\times14=196$ patches, plus the class token for 197 tokens in total.
- **Smaller patches** produce more tokens and usually higher accuracy, but the computation grows as $N^2$.

Q3: ***What is the role of the [class] token? Can we do without it?***

- The [class] token is a learnable vector that does not correspond to any image region. Through self-attention it **aggregates information from all patches**, and its final output serves as the representation of the whole image — an idea borrowed from BERT's [CLS].
- It is not required: **global average pooling (GAP)** over the outputs of all patch tokens works as well. In the original ViT paper the two perform similarly, but the learning rate needs to be adjusted; GAP is also commonly used in work such as DeiT III and MAE fine-tuning.
- In self-supervised models such as DINO, the attention map of the [class] token produces a good segmentation of foreground objects.

Q4: ***What kind of position embedding does ViT use? How is a change in resolution handled?***

- By default ViT uses **learnable 1D absolute position embeddings**, one $D$-dimensional vector per position. Experiments in the paper show little difference between 1D, 2D and relative position embeddings; the learned 1D embeddings themselves exhibit a 2D row/column structure.
- When the resolution changes during fine-tuning or inference (e.g., 224 → 384), the patch size stays the same and the number of tokens goes from $14\times14$ to $24\times24$, so the existing position embeddings no longer match in number.
- Solution: reshape the position embeddings into a 2D grid and perform **2D interpolation (usually bicubic)** to the new grid size according to their location in the original image; the class token's position embedding is kept separately.
- This is one of the few places where a 2D structural prior is injected manually into ViT; alternatives that extrapolate more easily include relative position bias (Swin), conditional position encoding (CPVT) and RoPE.

Q5: ***How does the inductive bias of ViT differ from that of CNNs?***

- CNNs have strong **inductive biases**: **locality** (kernels only look at a local neighborhood), **translation equivariance** (weight sharing), and a hierarchical structure (gradually growing receptive field).
- ViT's inductive bias is weak: only the patch splitting and the MLP are local and translation-equivariant; self-attention is **global**, and 2D spatial relationships must be learned from data.
- Consequences:
    - With little data, ViT overfits easily and performs worse than CNNs of comparable size;
    - With enough data, the weak inductive bias becomes an advantage: the model has a higher ceiling and can model global dependencies from the very first layer.
- One-sentence summary: **inductive bias trades prior knowledge for data efficiency**.

Q6: ***Why does ViT need a lot of data? What are the remedies?***

- In the original paper, ViT trained from scratch only on ImageNet-1k underperformed ResNets of similar size; only after pre-training on larger datasets such as ImageNet-21k and JFT-300M did ViT surpass CNNs. The reason is the lack of inductive bias: locality and spatial structure have to be learned from data.
- Remedies:
    - **Strong data augmentation and regularization**: RandAugment, Mixup, CutMix, Random Erasing, Stochastic Depth, Repeated Augmentation (as in DeiT);
    - **Knowledge distillation**: distill from a CNN teacher (DeiT);
    - **Self-supervised pre-training**: MAE, DINO, BEiT, etc., removing the dependence on large-scale labeled data;
    - **Introducing convolutional priors**: hybrid models, convolutional stems, local window attention (Swin), etc.

Q7: ***What improvements does DeiT make? What is the distillation token?***

- DeiT (Touvron et al., 2021) showed that ViT can be trained well **using only ImageNet-1k**, the key being the training recipe: strong data augmentation, regularization and carefully tuned optimization hyperparameters.
- **Distillation token**: in addition to the class token, another learnable token is added. It participates in self-attention with the other tokens, but its output is trained to match the predictions of a **teacher model** (usually a CNN such as RegNet):
    - The class token's output is trained with cross-entropy against the ground-truth label;
    - The distillation token's output is trained against the teacher's prediction; **hard distillation** (using the teacher's argmax as the label) works better than soft distillation.
- At inference, the predictions of the two token heads are fused.
- Using a CNN as the teacher works better; it is generally believed that the CNN "passes on" its inductive bias to ViT through distillation.

Q8: ***What is the core idea of the Swin Transformer?***

- The goal of Swin Transformer (Liu et al., 2021) is to build a **general-purpose vision backbone** suitable for dense prediction tasks such as detection and segmentation. Main designs:
    - **Hierarchical structure**: **Patch Merging** (concatenating neighboring 2×2 patches followed by a linear dimensionality reduction) downsamples stage by stage, producing multi-scale feature maps at 1/4, 1/8, 1/16 and 1/32 resolution that can be plugged directly into structures like FPN;
    - **Window attention (W-MSA)**: self-attention is computed only within non-overlapping $M\times M$ local windows (default $M=7$), so complexity is **linear** in image size;
    - **Shifted windows (SW-MSA)**: consecutive layers alternate between regular windows and windows shifted by $\lfloor M/2\rfloor$, letting information flow across windows; this is implemented efficiently using **cyclic shift + attention mask**;
    - **Relative position bias**: a learnable relative position bias $B$ is added to the attention logits: $\text{softmax}(QK^T/\sqrt{d}+B)V$.
- In essence, it reintroduces CNN priors such as locality and hierarchy into the Transformer.

Q9: ***What is the computational complexity of ViT? How does Swin reduce it?***

- For $N$ tokens of dimension $C$, global self-attention costs $O(N^2C)$, while linear projections and the MLP cost $O(NC^2)$. For an $h\times w$ feature map, $N=hw$:

$$\Omega(\text{MSA})=4hwC^2+2(hw)^2C$$

- The number of tokens grows quadratically with resolution, so the attention cost grows with the **fourth power** of resolution, which makes high-resolution inputs (detection, segmentation) very expensive.
- Swin's window attention:

$$\Omega(\text{W-MSA})=4hwC^2+2M^2hwC$$

- When the window size $M$ is fixed, complexity is **linear** in $hw$.
- Other ways to reduce complexity: spatial-reduction attention (PVT downsamples K/V), token pruning/merging (DynamicViT, ToMe), linear attention, and FlashAttention (which does not change complexity but greatly reduces memory-access overhead).

Q10: ***What is a hybrid model? What are its benefits?***

- **Hybrid ViT**: first extract a feature map with a CNN (e.g., the first few stages of a ResNet), then feed each spatial location of the feature map (or 1×1 patches) as tokens into the Transformer. The original ViT paper includes a ResNet + ViT hybrid version.
- Benefits: the CNN provides local features and inductive bias, which helps for small models and limited data; in the original paper, the gap between hybrid and pure ViT shrinks as model size increases.
- Related work: a **convolutional stem** replacing patchify with a few conv layers (more stable training, less sensitive to the optimizer), CvT, CoAtNet (stacking convolution and attention in different stages), ConViT, etc.

Q11: ***How is ViT used for object detection and semantic segmentation?***

- **Detection**:
    - DETR uses a CNN backbone + Transformer encoder-decoder, predicting a set directly via object queries and bipartite matching, with no NMS or anchors;
    - Hierarchical ViTs such as Swin replace ResNet as the backbone of Mask R-CNN / Cascade R-CNN;
    - **ViTDet**: uses a plain (single-scale) ViT directly as the backbone with a simple feature pyramid (up/downsampling the last-layer features), plus a few global attention blocks combined with window attention; it works very well with MAE pre-training.
- **Segmentation**:
    - SETR, Segmenter: ViT encoder + upsampling decoder;
    - DPT: reassembles tokens from multiple ViT layers into multi-resolution feature maps for depth estimation and segmentation;
    - Mask2Former: a unified segmentation framework based on mask classification; SAM uses an MAE-pretrained ViT as its image encoder.
- Difficulties: high resolution leads to huge token counts; plain ViT lacks multi-scale features. Common countermeasures are window attention, hierarchical structures and feature pyramids.

Q12: ***How can ViT attention be visualized?***

- **Raw attention maps**: take the attention weights of the [class] token over all patches from a given head in a given layer, reshape them into a $\sqrt{N}\times\sqrt{N}$ grid, upsample, and overlay on the original image.
- **Attention Rollout** (Abnar & Zuidema, 2020): to account for residual connections, add the identity matrix to each layer's attention matrix (averaged over heads), normalize, and multiply layer by layer: $\tilde{A}=\prod_l(0.5A_l+0.5I)$, tracing how information flows from input patches to output tokens.
- **Gradient-based methods**: variants of Grad-CAM, and Transformer interpretability methods that combine gradients with relevance propagation (Chefer et al., 2021).
- Caveat: attention weights are not the same as feature importance and should only be taken as a reference. In large models, high-norm "artifact" tokens may appear in background regions of attention maps; **register tokens** (Darcet et al., 2023, "Vision Transformers Need Registers") mitigate this by adding extra learnable tokens.

Q13: ***What do different layers of ViT learn? How does this differ from CNNs?***

- **Mean attention distance**: in shallow ViT layers some heads attend locally while others already attend globally, showing that global information can be integrated from the first layer; as depth increases, the attention distance of all heads increases.
- In CNNs the receptive field grows **gradually** with depth, and shallow layers only see local regions.
- Analyses (e.g., Raghu et al., 2021) found that representations across ViT layers are more uniform, and that residual connections play a larger role in propagating information in ViT.
- The filters of the linear projection layer resemble Gabor-like basis functions; the similarity between position embeddings shows a row/column structure.

Q14: ***How does CLIP's image encoder use ViT?***

- CLIP (Radford et al., 2021) jointly trains an image encoder and a text encoder with **image-text contrastive learning**; the image encoder can be a ResNet or a ViT (e.g., ViT-B/32, ViT-B/16, ViT-L/14).
- In the ViT version: an extra LayerNorm is applied after adding position embeddings to the patch embeddings; the output of the [class] token is linearly projected into the embedding space shared with text and then L2-normalized; cosine similarity with text features is used for contrast.
- Difference from supervised ViT: the supervision comes from natural language rather than a fixed set of classes, which enables **zero-shot classification** (fill class names into a template like "a photo of a {class}" and pick the text with the highest similarity).
- CLIP's ViT is widely used as the visual encoder of multimodal large models (e.g., LLaVA), typically taking patch token features from the second-to-last layer and feeding them to the LLM via a projection layer.

Q15: ***What are the training tricks for ViT?***

- Optimizer: **AdamW** with relatively large weight decay, combined with **learning-rate warmup** and cosine decay.
- Regularization: Stochastic Depth (DropPath), Label Smoothing, Mixup/CutMix, RandAugment.
- Gradient clipping stabilizes training; large models can use LayerScale (CaiT), QK-Norm, etc. to improve stability.
- Fine-tuning often uses a **higher resolution** with interpolated position embeddings; layer-wise learning-rate decay is very common when fine-tuning self-supervised pre-trained models.

Q16: ***How do ViT and CNNs differ in robustness?***

- Some studies found that ViT is more biased toward **shape** (shape bias) than texture compared to CNNs, and is more robust to perturbations such as occlusion and patch shuffling.
- However, conclusions depend on training data and augmentation: ConvNeXt (Liu et al., 2022) showed that transferring Transformer training recipes and designs (large kernels, LayerNorm, inverted bottlenecks, etc.) to a pure CNN reaches comparable performance, suggesting that many gaps come from the training recipe rather than the architecture itself.
- Neither has an inherent advantage in adversarial robustness; adversarial training is still needed.
