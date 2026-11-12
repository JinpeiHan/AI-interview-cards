# ViT

Q1: ***Describe the overall architecture of ViT (Vision Transformer)***

- **ViT cuts an image into fixed-size patches, linearly maps each patch to a token, adds a [class] token and position embeddings, feeds the sequence through a standard Transformer encoder, and classifies from the [class] token's output.**
- The encoder is Pre-LN: LayerNorm → multi-head self-attention → residual, LayerNorm → MLP → residual.
- Common configs: ViT-B (12 layers, 768 dims), ViT-L (24 layers, 1024 dims), ViT-H (32 layers, 1280 dims).
- "ViT-B/16" means the Base model with patch size 16.

Q2: ***How is patch embedding implemented? Why is it equivalent to a convolution?***

- **Each patch is flattened and multiplied by a learnable matrix to get a D-dimensional vector; this is exactly a convolution whose kernel size and stride both equal the patch size with D output channels, so code usually just uses `Conv2d(C, D, kernel_size=P, stride=P)`.**
- Each patch is processed once with no overlap and all patches share the same matrix, which is precisely how that convolution computes.
- Example: a 224×224 input with patch 16 gives 14×14=196 patches, 197 tokens with the class token.
- Smaller patches mean more tokens and usually higher accuracy, but attention cost grows with the square of the token count.

Q3: ***What is the role of the [class] token? Can we do without it?***

- **The [class] token is a learnable vector that corresponds to no image region; through self-attention it aggregates information from all patches, and its output is the representation of the whole image, an idea borrowed from BERT's [CLS].**
- It can be dropped: global average pooling (GAP) over the patch outputs works about as well with a tuned learning rate; MAE fine-tuning and others often use GAP.
- In self-supervised models like DINO, the [class] token's attention map segments foreground objects well.

Q4: ***What kind of position embedding does ViT use? How is a change in resolution handled?***

- **ViT uses learnable 1D absolute position embeddings by default; when the resolution changes the token count changes, so the embeddings are reshaped into a 2D grid and bicubically interpolated to the new grid size, with the class token's embedding kept separately.**
- The paper found 1D, 2D and relative embeddings perform about the same, and the learned 1D embeddings show row/column structure on their own.
- Example: going from 224 to 384, the token grid goes from 14×14 to 24×24.
- Extrapolation-friendlier options include relative position bias (Swin), conditional position encoding, or RoPE.

Q5: ***How does the inductive bias of ViT differ from that of CNNs?***

- **CNNs have strong built-in inductive biases (locality, translation equivariance, hierarchical receptive fields); ViT has almost none and must learn 2D spatial relations from data.**
- With little data, ViT overfits and trails a CNN of the same size.
- With enough data, the weak prior becomes an advantage: a higher ceiling, and global dependencies from the very first layer.
- In one line: inductive bias trades priors for data efficiency.

Q6: ***Why does ViT need a lot of data? What are the remedies?***

- **Because ViT lacks inductive bias, it must learn locality and spatial structure from data: trained from scratch on ImageNet-1k it trails ResNet, and only after pre-training on ImageNet-21k or JFT-300M does it beat CNNs.**
- Strong augmentation and regularization: RandAugment, Mixup, CutMix, Stochastic Depth, etc. (DeiT's recipe).
- Knowledge distillation from a CNN teacher (DeiT).
- Self-supervised pre-training: MAE, DINO, BEiT, which need no large labeled sets.
- Adding convolutional priors: hybrid models, a conv stem, window attention (Swin).

Q7: ***What improvements does DeiT make? What is the distillation token?***

- **DeiT shows ViT can be trained well on ImageNet-1k alone, thanks to strong augmentation, regularization and tuned hyperparameters, plus an extra distillation token.**
- The distillation token joins self-attention like the class token, but its output is trained to match the teacher's (usually a CNN's) predictions, while the class token still fits the true labels.
- Hard distillation (teacher's argmax as label) beats soft distillation; at inference the two heads' predictions are fused.
- A CNN teacher works better, commonly explained as passing the CNN's inductive bias to ViT.

Q8: ***What is the core idea of the Swin Transformer?***

- **Swin brings CNN-style locality and hierarchy back into the Transformer: attention is computed only within local windows and patches are merged stage by stage into multi-scale features, making it a general backbone for detection and segmentation.**
- Hierarchy: Patch Merging combines neighboring 2×2 patches to downsample, giving 1/4 to 1/32 multi-scale feature maps that plug straight into FPN.
- Window attention: computed only inside non-overlapping 7×7 windows, so cost is linear in image size.
- Shifted windows: adjacent layers alternate with a partition shifted by half a window so information flows across windows, implemented efficiently with a cyclic shift plus a mask.
- A learnable relative position bias is added to the attention.

Q9: ***What is the computational complexity of ViT? How does Swin reduce it?***

- **Global self-attention cost grows with the square of the token count, which itself grows with the square of the resolution, so high resolutions are very expensive; Swin restricts attention to fixed-size windows, making the cost linear in the token count.**
- The projections and MLP are already linear in the token count; the bottleneck is the attention term.
- Other ways to cut cost: downsampling K/V (PVT), token pruning/merging (ToMe), linear attention.
- FlashAttention does not change the complexity but greatly reduces memory traffic.
- Key formula ($h\times w$ feature map, window size $M$): $\Omega(\text{MSA})=4hwC^2+2(hw)^2C$, $\Omega(\text{W-MSA})=4hwC^2+2M^2hwC$

Q10: ***What is a hybrid model? What are its benefits?***

- **A hybrid model first extracts a feature map with a CNN (e.g. the early ResNet stages) and then feeds each spatial position of that map as a token into the Transformer.**
- The CNN contributes local features and inductive bias, which helps with small models and little data; the gap to pure ViT shrinks as models grow.
- Related ideas: a conv stem replacing patchify (more stable training), CvT, CoAtNet.

Q11: ***How is ViT used for object detection and semantic segmentation?***

- **There are two main routes: plug a hierarchical ViT like Swin in place of the CNN backbone of a detection/segmentation framework, or, like ViTDet, use a plain ViT and build a simple feature pyramid from it.**
- Detection: DETR uses a Transformer encoder-decoder with object queries to predict the set of objects directly, without NMS or anchors; ViTDet works very well with MAE pre-training.
- Segmentation: SETR and Segmenter (ViT encoder + upsampling decoder), DPT reassembles multi-layer tokens into multi-resolution features, Mask2Former unifies segmentation tasks, and SAM uses an MAE-pretrained ViT as its image encoder.
- The difficulties are the huge token count at high resolution and the lack of multi-scale features in plain ViT, handled with window attention and feature pyramids.

Q12: ***How can ViT attention be visualized?***

- **The simplest method takes the [class] token's attention weights over the patches, reshapes them into a grid, upsamples and overlays them on the image.**
- Attention Rollout: accounting for residuals, add the identity to each layer's attention matrix and multiply them layer by layer to trace how information flows from input patches to the output.
- Gradient-based methods also work, such as Grad-CAM variants or relevance-propagation methods.
- Attention weights are not feature importance; large models often show high-norm artifact tokens in the background, which Register tokens mitigate.

Q13: ***What do different layers of ViT learn? How does this differ from CNNs?***

- **From the first layer some ViT heads already attend globally, and attention distance grows for all heads with depth; in a CNN the receptive field grows gradually and shallow layers only see local regions.**
- Studies find ViT representations are more uniform across layers, and residual connections matter more for carrying information.
- The patch projection filters resemble Gabor-like basis functions; similarities between position embeddings show row/column structure.

Q14: ***How does CLIP's image encoder use ViT?***

- **CLIP jointly trains an image encoder and a text encoder with image-text contrastive learning; the ViT version takes the [class] token's output, linearly projects it into the embedding space shared with text, L2-normalizes it, and compares it with text features by cosine similarity.**
- The supervision comes from natural language rather than fixed classes, enabling zero-shot classification: fill class names into "a photo of a {class}" and pick the most similar one.
- Architecturally it only adds one LayerNorm after patch embedding plus position embedding.
- CLIP's ViT is widely used as the vision encoder of multimodal LLMs (e.g. LLaVA), usually feeding penultimate-layer patch features through a projector into the LLM.

Q15: ***What are the training tricks for ViT?***

- **The core recipe is AdamW with fairly large weight decay, learning-rate warmup plus cosine decay, and strong regularization and augmentation.**
- Regularization: DropPath, Label Smoothing, Mixup/CutMix, RandAugment.
- Stability: gradient clipping; large models can use LayerScale or QK-Norm.
- Fine-tuning often uses a higher resolution with interpolated position embeddings; layer-wise lr decay is common when fine-tuning self-supervised models.

Q16: ***How do ViT and CNNs differ in robustness?***

- **Some studies find ViT is more shape-biased than texture-biased and more robust to occlusion and patch shuffling, but this depends heavily on training data and augmentation.**
- ConvNeXt shows that porting Transformer training recipes and design choices to a pure CNN reaches similar performance, so much of the gap comes from training rather than architecture.
- Neither has a built-in advantage in adversarial robustness; adversarial training is still needed.
