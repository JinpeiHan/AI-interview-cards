# Contrastive Learning

Q1: ***What is contrastive learning? What is the core idea?***

- **Contrastive learning is label-free self-supervised representation learning: build positive pairs (e.g. two augmentations of the same image) and negative pairs (e.g. different images), and train an encoder so positives move close together in representation space and negatives move apart.**
- The learned representation is invariant to irrelevant changes such as augmentations while keeping the information that distinguishes samples.
- The encoder is then used for downstream classification, retrieval, detection and so on.

Q2: ***Write down the InfoNCE loss and explain its meaning***

- **InfoNCE is essentially a (K+1)-way softmax cross-entropy: the model must "pick out" the one positive among K negatives, with similarity measured by cosine similarity on normalized vectors divided by a temperature.**
- From information theory it is a lower bound on mutual information, and more negatives make the bound tighter, one theoretical reason contrastive learning likes many negatives.
- SimCLR's NT-Xent is a form of InfoNCE in which all other augmented samples in the batch serve as negatives.
- Key formula: $\mathcal{L}=-\log\frac{\exp(\text{sim}(q,k^+)/\tau)}{\exp(\text{sim}(q,k^+)/\tau)+\sum_{i=1}^{K}\exp(\text{sim}(q,k_i^-)/\tau)}$

Q3: ***What role does the temperature $\tau$ play?***

- **The temperature controls how sharp the softmax is, i.e. how much the model focuses on hard negatives.**
- Small temperature: a sharp distribution where the most similar hard negatives dominate the loss, giving more discriminative features, but too small is unstable and may push apart semantically similar "false negatives".
- Large temperature: a smooth distribution that treats all negatives alike, with a gentler penalty.
- It is essentially a trade-off between uniformity and tolerance; typical values are 0.05–0.5, and CLIP makes it learnable.

Q4: ***How are positive and negative samples constructed in contrastive learning?***

- **Positives are usually different "views" of the same sample, and negatives are usually just the other samples in the batch or a queue.**
- Positives: two augmentations of one image (SimCLR, MoCo), a paired image and text (CLIP), same-class samples when labels exist (SupCon), the same sentence under two dropout masks (SimCSE).
- Negatives: other samples in the batch (SimCLR, CLIP), a queue of past features (MoCo), a memory bank.
- Random negatives may include same-class false negatives, mitigated by a debiased contrastive loss or by removing suspicious negatives.

Q5: ***What are the key design choices of SimCLR?***

- **SimCLR is a simple end-to-end framework: augment each image twice, pass both through an encoder and an MLP projection head, and train with the NT-Xent loss; the keys are a strong augmentation combination, a nonlinear projection head and large batches.**
- The augmentation combination is crucial, especially random crop plus color distortion; with cropping alone the model "cheats" using color histograms.
- The loss is computed on the projection-head output, but downstream uses the features before the head, because the head discards augmentation-sensitive information.
- Negatives come from the batch, so bigger batches are better and need the LARS optimizer; the drawback is very high compute cost.

Q6: ***How does MoCo solve the problem of the number of negatives?***

- **MoCo stores key features from many past batches in a first-in-first-out queue as negatives, decoupling the number of negatives from the batch size, and uses a momentum encoder to keep the keys in the queue consistent.**
- Momentum encoder: the key encoder is not updated by backprop but is an EMA of the query encoder, with momentum around 0.999, so it changes slowly.
- Copying the query encoder directly makes keys from different times inconsistent, and training fails.
- MoCo v2 adds an MLP projection head and stronger augmentation; MoCo v3 targets ViT, dropping the queue for large batches.
- Key formula: $\theta_k\leftarrow m\theta_k+(1-m)\theta_q$

Q7: ***Why does BYOL not collapse even without negatives?***

- **BYOL avoids collapse with an asymmetric design: the online branch has an extra predictor head, and the target branch is an EMA of the online one with no gradient flowing back; this asymmetry makes the constant "all outputs equal" solution hard to reach during optimization.**
- Collapse means that if you only pull positives together, the easiest solution is a constant output, giving zero loss but a useless representation.
- The loss is the MSE between the online prediction and the target projection (both L2-normalized), computed again with the views swapped.
- The constant solution is still a global minimum of the loss in theory; BYOL relies on optimization dynamics, not the loss itself.

Q8: ***What did SimSiam find? What does stop-gradient do?***

- **SimSiam found BYOL's EMA is not necessary: with both branches sharing one encoder, no negatives and no large batch, keeping only the predictor and stop-gradient is enough to learn well; but removing stop-gradient collapses immediately.**
- The authors' explanation: it resembles EM-style alternating optimization, where stop-gradient fixes each image's target representation while the network parameters are optimized.
- The predictor approximately takes the expectation over different augmentations.
- Key formula: $\mathcal{L}=\frac{1}{2}D(p_1,\text{sg}(z_2))+\frac{1}{2}D(p_2,\text{sg}(z_1))$, where $D$ is negative cosine similarity

Q9: ***How does DINO work? How does it prevent collapse?***

- **DINO is label-free self-distillation: the teacher is an EMA of the student, and the student fits the teacher's output probability distribution with cross-entropy; collapse is prevented by centering and sharpening the teacher's outputs.**
- Multi-crop: the teacher sees only two global views while the student sees all global and small local crops, learning "local-to-global" correspondence.
- Centering subtracts a running-average center so no single dimension dominates, but it pushes the distribution toward uniform.
- Sharpening uses a low teacher temperature to make the distribution peaked, preventing uniform collapse; the two complement each other.
- On ViT, the [CLS] attention maps segment objects cleanly; DINOv2 adds masked objectives and large-scale curated data.

Q10: ***What are the ideas behind Barlow Twins and VICReg?***

- **These methods prevent collapse not with negatives or asymmetry but by directly constraining the statistics of the representation: embeddings of two views should be correlated in the same dimension and decorrelated across dimensions.**
- Barlow Twins: push the cross-correlation matrix of the two views' embeddings toward the identity; the diagonal encodes invariance, the off-diagonal reduces redundancy.
- VICReg has three terms: variance keeps each dimension's std above a threshold (directly preventing collapse), invariance is the MSE between the two views, and covariance drives off-diagonal entries to 0.
- Pros: no negatives, large batches, EMA or predictor needed; con: usually needs a high embedding dimension.
- Key formula: $\mathcal{L}_{BT}=\sum_i(1-\mathcal{C}_{ii})^2+\lambda\sum_i\sum_{j\ne i}\mathcal{C}_{ij}^2$

Q11: ***How does CLIP perform image-text contrastive learning?***

- **CLIP jointly trains an image encoder and a text encoder on huge numbers of web image-text pairs: for a batch of N pairs it computes an N×N similarity matrix, the diagonal is positives and everything else negatives, and it averages an image-to-text and a text-to-image InfoNCE loss.**
- The temperature is learnable; very large batches are needed for enough negatives.
- It enables zero-shot classification (build text prompts from class names and pick the most similar) and cross-modal retrieval, and is the vision encoder of many multimodal models.
- SigLIP replaces the softmax with a pairwise sigmoid loss, avoiding full-batch normalization and being friendlier to batch size.

Q12: ***What are hard negatives? How can they be exploited?***

- **Hard negatives are negatives that look similar to the anchor in representation space but differ semantically; they provide the most gradient, while easy negatives quickly give near-zero gradients.**
- Ways to exploit them: lower the temperature (implicit weighting), mine high-similarity negatives from the queue or retrieval results, reweight negatives by similarity, or synthesize hard negatives in feature space (e.g. MoCHi).
- Risk: the harder a negative, the more likely it is a false negative, and overemphasizing them hurts the representation.
- Retrieval tasks (e.g. DPR) often use non-answer passages retrieved by BM25 as hard negatives.

Q13: ***What are alignment and uniformity?***

- **Alignment means representations of positive pairs should be close; uniformity means all representations should spread as evenly as possible on the unit hypersphere to keep as much information as possible.**
- InfoNCE's numerator mainly optimizes alignment, and its denominator (the negatives) mainly optimizes uniformity.
- Both can be used directly to diagnose representations: alignment without uniformity collapses, and uniformity without alignment is not invariant to augmentations.

Q14: ***What is dimensional collapse?***

- **Dimensional collapse means representations do not shrink to a single point but occupy only a low-dimensional subspace of the embedding space, leaving many dimensions unused.**
- This differs from complete collapse, where all representations converge to one point.
- It happens even in contrastive learning with negatives, due to strong augmentation and the implicit regularization of multi-layer networks.
- Diagnosis: look at the singular value spectrum or effective rank of the embedding covariance; many near-zero singular values indicate dimensional collapse.
- Remedies: the decorrelation terms of Barlow Twins/VICReg, projection-head design (e.g. DirectCLR), whitening (W-MSE).

Q15: ***How are self-supervised/contrastive representations evaluated?***

- **The most common is linear probing: freeze the encoder and train only a linear classifier, measuring the linear separability of the features.**
- kNN classification: nearest-neighbor voting on frozen features with no training (often reported by DINO).
- Fine-tuning: unfreeze everything to measure value as an initialization; plus semi-supervised evaluation with only 1% or 10% of labels.
- Transfer to other tasks and datasets such as detection and segmentation.
- Protocols can disagree: contrastive methods are usually strong at linear probing, while masked modeling like MAE is strong at fine-tuning and weaker at linear probing.

Q16: ***Why does SimCLR use the features before the projection head downstream?***

- **Because the contrastive loss demands invariance to augmentation, the projection-head output throws away information changed by augmentation, such as color and orientation, which may be useful downstream; the features before the head keep more of it.**
- In experiments, linear classification on pre-head features is clearly better.
- Predicting which augmentation was applied is inaccurate from the head output but accurate from the pre-head features, supporting this explanation.
- Another explanation: the projection head absorbs dimensional collapse, keeping the backbone features at a higher effective rank.

Q17: ***What are the characteristics of contrastive learning compared with supervised learning and masked modeling?***

- **Compared with supervised learning, contrastive learning needs no labels and learns instance-level discrimination; compared with masked modeling, it learns global, invariant semantic features with strong linear separability, but depends on augmentation design and collapse-prevention mechanisms.**
- When transferred to downstream tasks like detection, it often matches or beats supervised pre-training.
- Masked modeling (MAE, BEiT) keeps more spatial detail, suits fine-tuning and dense prediction better, and scales more easily.
- The two families are merging, e.g. iBOT and DINOv2 use both self-distillation and masked objectives.
