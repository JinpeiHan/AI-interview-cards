# MAE

Q1: ***Describe the basic idea of MAE (Masked Autoencoder)***

- **MAE is a masked image modeling self-supervised method: randomly mask most image patches (75% by default), let a ViT encoder see only the remaining patches, and have a lightweight decoder reconstruct the pixels of the masked ones.**
- The decoder input is the encoded tokens plus mask tokens and position embeddings.
- The loss is computed only on the masked patches.
- After pre-training the decoder is discarded and only the encoder is used downstream.

Q2: ***Why does MAE use such a high mask ratio as 75%?***

- **Images are highly redundant: with little masking the model can fill gaps by simple interpolation from neighboring patches and learns no semantics; masking 75% forces it to understand the overall structure of objects and scenes.**
- In experiments both linear probing and fine-tuning peak around 75%, whereas BERT masks only 15%.
- Bonus: the encoder processes only 25% of tokens, cutting pre-training compute and memory sharply, which makes large models and long schedules feasible.

Q3: ***What is MAE's asymmetric encoder-decoder design? Why is it designed this way?***

- **The encoder processes only visible patches and never sees mask tokens; the decoder is light (8 layers, 512 dims by default) and handles the full sequence to reconstruct pixels.**
- Efficiency: encoder cost drops with fewer visible tokens, speeding pre-training by 3x or more.
- Consistency: the encoder never sees mask tokens in pre-training, matching downstream inputs (BEiT/SimMIM, by contrast, feed mask tokens to the encoder).
- Division of labor: pixel details go to the decoder and the encoder focuses on semantics; a deeper decoder helps linear probing.

Q4: ***What is MAE's loss function? Why is it computed only on masked patches?***

- **The loss is the mean squared error between reconstructed and original pixels, computed only on masked patches, because visible patches only need to be "copied" and including them would dilute the learning signal.**
- Experiments also show higher accuracy with the masked-only loss, the same idea as BERT predicting only masked words.
- Key formula: $\mathcal{L}=\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}\|\hat{x}_i-x_i\|_2^2$

Q5: ***What are normalized pixel targets? Why are they useful?***

- **Each patch is normalized by its own pixel mean and standard deviation before being used as the reconstruction target.**
- This removes low-frequency information like overall brightness and contrast, so the model focuses on texture, edges and shape rather than average color.
- It consistently improves representation quality and is simpler than BEiT's dVAE token target, with no extra tokenizer to train.

Q6: ***Why did masked modeling take off later for images than for NLP? How do images differ from text?***

- **The MAE paper gives three reasons: early vision relied on CNNs, where mask tokens do not fit naturally; images have low information density and high redundancy, so only high mask ratios make the task meaningful; and reconstructing pixels is a low-semantic task that needs a carefully designed decoder.**
- Architecture: the obstacle to adding mask tokens and position embeddings disappeared with ViT.
- Information density: language is a highly semantic signal, so predicting a few missing words is already hard.
- Decoder: BERT predicts words, so an MLP suffices; MAE reconstructs pixels, and the decoder design affects the semantic level of the representation.

Q7: ***What masking strategies does MAE consider? Which is better?***

- **Random masking is best and the default: it performs best at high mask ratios and has no pattern the model can exploit.**
- Block-wise masking (used by BEiT): masks large contiguous regions, too hard at high ratios and gives blurry reconstructions.
- Grid masking: keeps patches in a regular pattern, too easy to interpolate, worse representations.

Q8: ***How much does MAE depend on data augmentation?***

- **MAE barely depends on augmentation: random resized crop plus horizontal flip works well, and strong augmentation like color jitter can even hurt.**
- The reason is that random masking is itself a strong augmentation; masks differ every iteration, producing many different samples from one image.
- This contrasts with contrastive methods like SimCLR and BYOL, which rely heavily on carefully designed augmentations to build positive pairs.

Q9: ***What characterizes MAE's linear probe and fine-tuning results?***

- **MAE is strong under fine-tuning but weaker under linear probing: fine-tuning excels at classification, detection and segmentation and keeps improving with scale, but frozen features are less linearly separable than those of MoCo v3 or DINO.**
- The reason is that MAE features are more nonlinear, semantics are spread across layers, and the last layer still keeps many pixel details.
- Fine-tuning only the last one or two blocks (partial fine-tuning) far exceeds linear probing, showing linear probing does not fully reflect representation quality.

Q10: ***How does MAE differ from BEiT?***

- **BEiT first turns images into discrete visual tokens with a dVAE and predicts the token class at masked positions; MAE regresses pixels directly with no tokenizer, which is simpler and more efficient.**
- Masking: BEiT uses about 40% block-wise masking, MAE uses 75% random masking.
- Encoder: BEiT processes all tokens (including mask tokens), MAE only visible patches.
- Conclusion: pixel targets with a high mask ratio and normalized pixels are enough; BEiT v2 later switched to a semantic tokenizer distilled from teachers like CLIP.

Q11: ***How does MAE differ from SimMIM?***

- **SimMIM also regresses pixels directly, but its encoder processes the full sequence including mask tokens, so it works directly with hierarchical models like Swin, at the cost of more compute than MAE.**
- There is no separate decoder; the prediction head is a single linear layer.
- It uses an L1 loss.
- It uses large mask patches (e.g. 32×32) with random masking.

Q12: ***What are the pros and cons of MAE compared to contrastive learning methods?***

- **MAE is simple, efficient, scales to large models, and is strong for fine-tuning and dense prediction; but its frozen features do worse than contrastive methods on linear probing, kNN and few-shot.**
- Pros: no negatives, large batches or strong augmentation needed, and rich spatial detail is preserved.
- Cons: the pixel target spends capacity on low-level details; pre-training sees only 25% of patches, a different input distribution from inference.
- Contrastive learning (SimCLR, MoCo, DINO) gives more semantic, linearly separable global features but depends on augmentation design and is not necessarily better on dense tasks.
- The two can be combined, e.g. iBOT, CMAE.

Q13: ***How does VideoMAE extend MAE to video?***

- **Video is even more redundant in time, so VideoMAE uses an extreme 90%–95% mask ratio and tube masking: the same spatial location is masked across all frames so the model cannot copy from neighboring frames.**
- Tokens are spatio-temporal cubes (e.g. 2×16×16) with joint space-time attention.
- The extreme mask ratio keeps encoder cost tiny, allowing effective pre-training even on mid-sized video datasets.

Q14: ***How has MAE been extended to audio and other modalities?***

- **Audio-MAE turns audio into a mel spectrogram, patchifies it like an image, masks a high ratio and reconstructs; its decoder uses local window attention because spectrograms are strongly locally correlated in time and frequency.**
- MultiMAE masks several modalities at once (RGB, depth, segmentation) for cross-modal reconstruction.
- There are also versions for point clouds (Point-MAE) and graphs (GraphMAE).
- Common rule: the more redundant the modality, the higher the mask ratio it can tolerate.

Q15: ***What are improved variants of MAE's pre-training objective?***

- **The overall trend is moving from reconstructing low-level pixels to predicting higher-level feature or semantic targets.**
- MaskFeat: reconstructs HOG features, which are less sensitive to lighting.
- data2vec: predicts an EMA teacher's layer-averaged features on the full input, unifying vision, speech and text.
- iBOT: self-distillation on masked tokens; I-JEPA: predicts target-block embeddings in representation space instead of pixels.
- EVA, BEiT v2: use CLIP features as the target, bringing in language-aligned semantics.

Q16: ***Why is MAE called "scalable"?***

- **Because it is cheap to train, keeps improving as models grow, and is simple.**
- Compute: the encoder processes only 25% of tokens, greatly lowering the cost of training ViT-L and ViT-H.
- Performance: fine-tuning accuracy rises with model size, and ViT-H trains effectively on ImageNet-1k alone without much overfitting.
- Simplicity: no negatives, momentum encoder, tokenizer or complex augmentation, so it is easy to reproduce at scale.
