# MAE

Q1: ***Describe the basic idea of MAE (Masked Autoencoder)***

- MAE (He et al., 2021, "Masked Autoencoders Are Scalable Vision Learners") is a self-supervised pre-training method based on **Masked Image Modeling (MIM)**.
- Pipeline:
    1. Split the image into patches and **randomly mask the majority of them** (75% by default);
    2. The **encoder** (a ViT) processes only the visible patches;
    3. **Mask tokens** are inserted into the encoded sequence, position embeddings are added, and the result is fed to a lightweight **decoder**;
    4. The decoder **reconstructs the pixels of the masked patches**, and the loss is computed only on the masked patches.
- After pre-training, the decoder is discarded and only the encoder is used for feature extraction or fine-tuning on downstream tasks.

Q2: ***Why does MAE use such a high mask ratio as 75%?***

- **Images are highly redundant**: neighboring pixels/patches are highly correlated. With a low mask ratio, the model can recover missing parts by simple interpolation from neighboring patches and learns no high-level semantics.
- A high mask ratio makes the task **hard enough**, forcing the model to understand the overall structure of objects and scenes in order to fill in large missing regions.
- In the paper's experiments, both linear probing and fine-tuning accuracy are good around a 75% mask ratio, whereas BERT masks only 15%.
- Extra benefit: the encoder processes only 25% of the tokens, so **pre-training compute and memory drop dramatically**, making it feasible to train large models with longer schedules.

Q3: ***What is MAE's asymmetric encoder-decoder design? Why is it designed this way?***

- The **asymmetry** has two aspects:
    - The encoder **sees only the visible patches** and does not process mask tokens; the decoder processes the full sequence (encoded visible tokens + mask tokens);
    - The decoder is **lightweight** (8 layers, 512 dim by default), narrower and shallower than the encoder, accounting for only a small fraction of the compute.
- Benefits:
    - **Efficiency**: encoder compute drops greatly as the number of visible tokens shrinks, speeding up pre-training by roughly 3× or more;
    - **Eliminating train-test mismatch**: the encoder never sees mask tokens during pre-training, and downstream inputs are all real patches, so the two are consistent (in contrast to BEiT/SimMIM, which feed mask tokens into the encoder);
    - **Division of labor**: low-level detail reconstruction is handled by the decoder, so the encoder can focus on learning semantic representations.
- The paper found that decoder **depth** matters a lot for linear probing (deeper is better) but much less for fine-tuning.

Q4: ***What is MAE's loss function? Why is it computed only on masked patches?***

- The loss is the **mean squared error (MSE)** between reconstructed and original pixels, computed only on masked patches:

$$\mathcal{L}=\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}\|\hat{x}_i-x_i\|_2^2$$

- Reason: reconstructing visible patches only requires "copying" the input; the task is too easy and including it in the loss would dilute the learning signal. In the paper, computing the loss only on masked patches gives higher accuracy (consistent with BERT predicting only the masked words).

Q5: ***What are normalized pixel targets? Why are they useful?***

- Each patch is normalized using its own pixel **mean and standard deviation** before being used as the reconstruction target: $\tilde{x}_i=\frac{x_i-\mu_i}{\sigma_i+\epsilon}$.
- Effect: removes low-frequency information such as overall patch brightness and contrast, **enhancing local contrast and high-frequency structure**, so the model focuses on texture, edges and shape rather than average color.
- In the paper's experiments, normalized pixel targets consistently improve representation quality; they are simpler than BEiT's dVAE token targets and require no separately trained tokenizer.

Q6: ***Why did masked modeling take off later for images than for NLP? How do images differ from text?***

- The MAE paper discusses three differences between vision and language:
    - **Architectural difference**: vision was long dominated by CNNs, which cannot easily incorporate mask tokens and position embeddings; this obstacle disappeared with ViT.
    - **Difference in information density**: language is a human-generated, highly semantic and information-dense signal, so predicting a few missing words is already a hard task; images are natural signals with heavy spatial redundancy, so a **high mask ratio** is needed to make the task meaningful.
    - **Different role of the decoder**: BERT predicts semantic-level words, so its decoder can be a simple MLP; MAE reconstructs pixels, which are low-level, so a separate decoder is needed and its design affects the semantic level of the learned representation.
- In addition, BERT's encoder processes [MASK] tokens, whereas MAE's encoder completely drops masked patches.

Q7: ***What masking strategies does MAE consider? Which is better?***

- **Random masking**: masked patches are sampled uniformly at random; this is the default and works best at high mask ratios.
- **Block-wise masking** (used by BEiT): masks large contiguous regions; at high mask ratios the task becomes too hard and reconstructions are blurry.
- **Grid masking**: regularly keeps one patch out of every few; the task is too easy (easy to interpolate) and representation quality is worse.
- Another benefit of random masking is that it has no regular pattern, preventing the model from exploiting shortcuts in the mask pattern.

Q8: ***How much does MAE depend on data augmentation?***

- MAE works very well with only simple **random resized cropping + horizontal flipping**; strong augmentations like color jitter may even hurt.
- Reason: **random masking itself is a strong data augmentation**: the mask differs at each iteration, generating many different training samples from the same image.
- This is in sharp contrast to contrastive learning: SimCLR, BYOL and similar methods **rely heavily** on carefully designed augmentations (especially color distortion and cropping) to define positive pairs.

Q9: ***What characterizes MAE's linear probe and fine-tuning results?***

- **Fine-tuning is strong**: MAE-pretrained ViTs perform very well on ImageNet fine-tuning, COCO detection, ADE20K segmentation, etc., and keep improving as model size grows (good scalability).
- **Linear probing is relatively weak**: compared with contrastive/distillation methods such as MoCo v3 and DINO, MAE features are less linearly separable.
- Analysis: MAE features are **more non-linear**; semantic information is spread across several layers and requires non-linear layers to be fully exploited. The pixel reconstruction objective also leaves many low-level details in the last-layer features.
- The paper proposes **partial fine-tuning**: fine-tuning only the last one or two Transformer blocks already yields accuracy far above linear probing, indicating that linear probing does not fully reflect representation quality.

Q10: ***How does MAE differ from BEiT?***

- **BEiT** (Bao et al., 2021):
    - First uses DALL-E's **dVAE** to discretize images into visual tokens; the pre-training objective is to predict the **visual token class** at masked positions (a classification task);
    - Uses **block-wise masking** with a mask ratio of about 40%;
    - The encoder processes all tokens (masked positions are replaced by mask tokens).
- **MAE**:
    - Reconstructs **pixels** directly (MSE regression), with no pre-trained tokenizer;
    - Random masking at 75%;
    - The encoder processes only visible patches, which is more efficient.
- Conclusion: MAE is simpler and more efficient, showing that pixel targets are sufficient with the right design (high mask ratio, normalized pixels). BEiT v2 later switched to a semantic tokenizer distilled from teachers such as CLIP to make the targets more semantic.

Q11: ***How does MAE differ from SimMIM?***

- SimMIM (Xie et al., 2021) was proposed almost simultaneously with MAE and also regresses raw pixels directly, but:
    - **The encoder processes mask tokens**, i.e., the full sequence enters the encoder, so it can be applied directly to hierarchical architectures like Swin (MAE's token-dropping is hard to combine with window attention);
    - The prediction head is **a single linear layer**, with no separate decoder;
    - It uses an **L1 loss**;
    - It uses relatively large mask patches (e.g., 32×32) with random masking and achieves good results with a moderate mask ratio.
- Cost: the encoder must process all tokens, so pre-training is more expensive than MAE.

Q12: ***What are the pros and cons of MAE compared to contrastive learning methods?***

- **Pros of MAE**:
    - No negatives, no large batches, no reliance on strong data augmentation;
    - The encoder processes only part of the tokens, so training is efficient and **scales to very large models**;
    - Strong in fine-tuning and dense prediction tasks (detection, segmentation), retaining rich spatial detail.
- **Cons of MAE**:
    - Weaker in settings that use frozen features directly, such as linear probing, few-shot and kNN;
    - The pixel reconstruction objective makes the model spend capacity on modeling low-level details;
    - The input distribution differs between pre-training and inference (only 25% of patches are seen during pre-training).
- **Contrastive learning** (SimCLR, MoCo, DINO) learns global features with stronger semantics and linear separability, but depends on augmentation design and is biased toward image-level invariance, so it is not necessarily better on dense tasks.
- The two can be combined: iBOT, CMAE and others combine masked modeling with contrastive/self-distillation objectives.

Q13: ***How does VideoMAE extend MAE to video?***

- Video is **even more redundant in time** (adjacent frames are nearly identical), so VideoMAE (Tong et al., 2022) uses an **even higher mask ratio (90%–95%)**.
- It uses **tube masking**: the same spatial location is masked across all frames, preventing the model from "copying" information from the same location in adjacent frames and avoiding information leakage.
- The video is split into spatio-temporal cubes (e.g., 2×16×16) as tokens, with joint space-time attention.
- Because of the very high mask ratio, encoder compute drops sharply, allowing effective pre-training even on medium-sized video datasets, with high data efficiency.

Q14: ***How has MAE been extended to audio and other modalities?***

- **Audio-MAE** (Huang et al., 2022): converts audio into a **mel spectrogram**, treats the spectrogram as an image split into patches, masks a high proportion and reconstructs them; the decoder uses local window attention because spectrograms have strong local correlations in time and frequency.
- Multimodal versions: **MultiMAE** masks multiple modalities such as RGB, depth and semantic segmentation simultaneously for cross-modal reconstruction; there is also work applying masked modeling to point clouds (Point-MAE), graphs (GraphMAE), etc.
- Common pattern: the more redundant the modality, the higher the mask ratio it can tolerate.

Q15: ***What are improved variants of MAE's pre-training objective?***

- **MaskFeat**: reconstructs **HOG features** of the masked regions; HOG is less sensitive to illumination changes;
- **data2vec**: predicts the average of multiple layer features produced by an EMA teacher on the **full input**, unifying vision, speech and text;
- **iBOT**: self-distillation with an online tokenizer on masked tokens, combined with a DINO-style global objective;
- **I-JEPA**: predicts embeddings of target blocks in representation space instead of reconstructing pixels, yielding more semantic features;
- **EVA, BEiT v2**: use CLIP features as reconstruction targets, introducing language-aligned semantics.
- Trend: moving from low-level **pixel targets** toward higher-level **feature/semantic targets**.

Q16: ***Why is MAE called "scalable"?***

- **Compute scalability**: the encoder processes only 25% of the tokens, which significantly lowers the cost of training large models (ViT-L, ViT-H);
- **Performance scalability**: as the model grows, MAE fine-tuning accuracy keeps improving, with less of the overfitting that large models suffer on small datasets, so large models like ViT-H can be trained effectively with only ImageNet-1k data;
- **Simplicity**: no negatives, momentum encoder, tokenizer or complex augmentations are needed, making it easy to reproduce and scale up.
