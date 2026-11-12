# Contrastive Learning

Q1: ***What is contrastive learning? What is the core idea?***

- Contrastive learning is a **self-supervised representation learning** method: without human labels, it constructs **positive pairs** (semantically the same, e.g., two augmentations of the same image) and **negative pairs** (semantically different, e.g., different images), and learns an encoder such that:
    - Positives are **pulled together** in the representation space;
    - Negatives are **pushed apart**.
- The learned representation is **invariant** to irrelevant changes such as data augmentation, while retaining the information needed to distinguish different samples; it can be used for downstream classification, retrieval, detection, etc.

Q2: ***Write down the InfoNCE loss and explain its meaning***

- For an anchor $q$, a positive $k^+$ and $K$ negatives $\{k_i^-\}$, the InfoNCE loss is:

$$\mathcal{L}=-\log\frac{\exp(\text{sim}(q,k^+)/\tau)}{\exp(\text{sim}(q,k^+)/\tau)+\sum_{i=1}^{K}\exp(\text{sim}(q,k_i^-)/\tau)}$$

- $\text{sim}$ is usually **cosine similarity** (vectors are L2-normalized before the dot product), and $\tau$ is the **temperature**.
- Interpretations:
    - It is essentially a **$(K+1)$-way softmax cross-entropy** that asks the model to "pick out" the positive among $K+1$ candidates;
    - From an information-theoretic perspective, it gives a lower bound on mutual information: $I(q;k^+)\ge\log(K+1)-\mathcal{L}$. **More negatives make the bound tighter**, which is one theoretical motivation for contrastive learning's preference for many negatives.
- **NT-Xent** (normalized temperature-scaled cross-entropy) in SimCLR is a form of InfoNCE, where the other $2(N-1)$ augmented samples in the same batch serve as negatives.

Q3: ***What role does the temperature $\tau$ play?***

- $\tau$ controls the **sharpness** of the softmax distribution:
    - **Small $\tau$**: a sharper distribution; the loss is dominated by the **hard negatives** most similar to the anchor and the gradient concentrates on them, giving more discriminative representations. But if it is too small, training can become unstable, and semantically similar samples ("false negatives") may be forcibly pushed apart;
    - **Large $\tau$**: a smoother distribution; all negatives are treated more equally, the penalty is milder, and attention to hard negatives decreases.
- Wang & Liu (2021) showed that temperature controls a trade-off between **uniformity and tolerance**.
- In practice $\tau$ is usually around 0.05–0.5; CLIP makes $\tau$ a **learnable parameter** (parameterized in log space with an upper bound).

Q4: ***How are positive and negative samples constructed in contrastive learning?***

- **Positives**:
    - Two different data augmentations of the same image (SimCLR, MoCo);
    - Different frames/clips of the same video, or different modalities of the same object (image-text pairs, CLIP);
    - Samples of the same class when labels are available (supervised contrastive learning, SupCon);
    - In NLP, the same sentence passed twice with different dropout (SimCSE).
- **Negatives**:
    - Other samples in the same batch (SimCLR, CLIP);
    - A queue of historical features (MoCo);
    - A memory bank (InstDisc).
- Issue: random negatives may contain samples of the same class as the anchor — **false negatives** — which harm semantic structure; this can be mitigated with a debiased contrastive loss or by removing suspicious negatives.

Q5: ***What are the key design choices of SimCLR?***

- SimCLR (Chen et al., 2020) is a simple end-to-end framework: augment the same image twice → encoder (ResNet) → **projection head** (MLP) → NT-Xent loss. Key findings:
    - **The composition of data augmentations is crucial**: the combination of **random cropping + color distortion** matters most; with cropping alone, the model can "cheat" by using color histograms to distinguish positives from negatives; Gaussian blur is also used;
    - **Non-linear projection head**: an MLP is added after the representation $h$ to get $z=g(h)$; the loss is computed on $z$, but $h$ is used downstream. The projection head discards augmentation-sensitive information (e.g., color, orientation), so $h$ retains more general information;
    - **Large batches and long training**: negatives come from within the batch, so larger batches mean more negatives and better results; the LARS optimizer is needed to support large-batch training;
    - Contrastive learning benefits more from **bigger models** than supervised learning does.
- Drawback: relies on very large batches and thus demands substantial compute.

Q6: ***How does MoCo solve the problem of the number of negatives?***

- MoCo (He et al., 2019) views contrastive learning as **dictionary look-up**: a query encoder encodes the anchor and a key encoder encodes positives and negatives; the dictionary should be **large** and **consistent**.
- **Queue**: key features from the past several batches are stored in a FIFO queue as negatives, **decoupling** dictionary size from batch size so that small batches can still use many negatives.
- **Momentum encoder**: the key encoder's parameters are not updated by backpropagation but are an **exponential moving average (EMA)** of the query encoder's parameters:

$$\theta_k\leftarrow m\theta_k+(1-m)\theta_q$$

- $m$ is close to 1 (e.g., 0.999), so the key encoder changes slowly and keys produced at different times in the queue **remain consistent**; directly copying the query encoder ($m=0$) makes training fail.
- Follow-ups: MoCo v2 adopts SimCLR's MLP projection head and stronger augmentations; MoCo v3 applies it to ViT, drops the queue in favor of large batches, and finds that freezing the randomly initialized patch projection layer improves training stability.

Q7: ***Why does BYOL not collapse even without negatives?***

- **Collapse**: if positives are only pulled together, the simplest solution is for all inputs to map to the same constant vector — zero loss but a useless representation. One role of negatives is to prevent collapse.
- BYOL (Grill et al., 2020) architecture:
    - **Online network**: encoder + projection head + **predictor**;
    - **Target network**: encoder + projection head, whose parameters are an **EMA** of the online network and receive no gradients;
    - Loss: MSE between the online network's prediction and the target network's projection (both L2-normalized), symmetrized by swapping the two views.
- The key to avoiding collapse is the **asymmetric architecture**:
    - The **predictor** together with **stop-gradient** (no gradient through the target branch) breaks the symmetry, so the constant solution is no longer an equilibrium that the optimization dynamics easily reach;
    - The EMA target network provides a slowly changing, stable regression target.
- Theoretically the constant solution is still a global minimum of the loss; BYOL relies on **optimization dynamics** rather than the loss function itself to avoid collapse.

Q8: ***What did SimSiam find? What does stop-gradient do?***

- SimSiam (Chen & He, 2020) is a minimal simplification of BYOL: it **removes the EMA**, both branches **share the same encoder**, no negatives or large batches are used, and only the predictor and **stop-gradient** are kept:

$$\mathcal{L}=\frac{1}{2}D(p_1,\text{sg}(z_2))+\frac{1}{2}D(p_2,\text{sg}(z_1))$$

- Here $D$ is the negative cosine similarity and $\text{sg}$ denotes stop-gradient.
- Experiments show that **removing stop-gradient causes immediate collapse**, while the EMA is not necessary.
- The authors' explanation: SimSiam resembles an **EM-like alternating optimization** — stop-gradient amounts to fixing one set of variables (the target representation of each image) while optimizing the other (the network parameters), and the predictor approximates the expectation over augmentations.

Q9: ***How does DINO work? How does it prevent collapse?***

- DINO (Caron et al., 2021) is **self-distillation with no labels**: the student and teacher networks have the same architecture, and the teacher's parameters are an **EMA** of the student's.
- The teacher's output goes through a softmax to give a probability distribution, and the student fits it with **cross-entropy**: $\mathcal{L}=-\sum P_t(x)\log P_s(x')$.
- **Multi-crop**: the teacher sees only 2 global views while the student sees all global and small local views, encouraging "local-to-global" correspondence.
- Collapse is prevented by two complementary operations:
    - **Centering**: subtract a moving-average center $c$ from the teacher's output, preventing any one dimension from dominating, but this pushes the output toward a uniform distribution;
    - **Sharpening**: the teacher uses a low temperature to make its distribution sharp, preventing collapse to the uniform distribution.
- Properties: with ViT, the self-attention map of the [CLS] token shows clear **object segmentation**; the features are well suited to kNN classification and retrieval. The later DINOv2 combines iBOT's masked objective with large-scale curated data to obtain general-purpose visual features.

Q10: ***What are the ideas behind Barlow Twins and VICReg?***

- These methods avoid collapse by **constraining the statistics of the representation**, rather than relying on negatives or an asymmetric architecture.
- **Barlow Twins** (Zbontar et al., 2021): compute the **cross-correlation matrix** $\mathcal{C}$ between the embeddings of the two views (standardized along the batch) and push it toward the identity matrix:

$$\mathcal{L}=\sum_i(1-\mathcal{C}_{ii})^2+\lambda\sum_i\sum_{j\ne i}\mathcal{C}_{ij}^2$$

- Diagonal term: the same dimension is highly correlated between the two views (**invariance**);
- Off-diagonal term: different dimensions are decorrelated (**redundancy reduction**).
- **VICReg** (Bardes et al., 2021): three loss terms:
    - **Variance**: the standard deviation of each dimension within the batch must stay above a threshold (hinge loss), directly preventing collapse to a constant;
    - **Invariance**: MSE between the embeddings of the two views;
    - **Covariance**: off-diagonal elements of the covariance matrix are pushed toward 0, for decorrelation.
- Advantages: no negatives, large batches, EMA or predictor needed; drawback: usually requires a fairly high embedding dimension.

Q11: ***How does CLIP perform image-text contrastive learning?***

- CLIP (Radford et al., 2021) jointly trains an **image encoder** and a **text encoder** on large-scale web image-text pairs.
- For a batch of $N$ image-text pairs, an $N\times N$ similarity matrix is computed; the diagonal entries are positive pairs and the remaining $N^2-N$ are negatives.
- The loss is a **symmetric cross-entropy**: InfoNCE in the image→text direction and in the text→image direction, averaged:

$$\mathcal{L}=\frac{1}{2}\left(\mathcal{L}_{i\to t}+\mathcal{L}_{t\to i}\right)$$

- The temperature is learnable; very large batches are needed to provide enough negatives.
- Capabilities: **zero-shot classification** (build text prompts from class names and choose the text with the highest similarity), cross-modal retrieval, and serving as the visual encoder of many multimodal models.
- Improvement: **SigLIP** replaces the softmax with a pairwise sigmoid loss, avoiding normalization over the whole batch and being friendlier to batch size.

Q12: ***What are hard negatives? How can they be exploited?***

- **Hard negatives**: negatives that are close to the anchor in representation space but semantically different; they provide the most gradient information, whereas easy negatives quickly yield near-zero gradients.
- Ways to exploit them:
    - Lowering the temperature $\tau$, which implicitly upweights hard negatives;
    - **Hard negative mining**: selecting highly similar negatives from the queue or from retrieval results;
    - **Reweighting** negatives by similarity (e.g., the hard negative sampling of Robinson et al., 2021);
    - **Synthesizing** hard negatives in feature space (e.g., MoCHi mixes negatives).
- Risk: the harder the negative, the more likely it is a **false negative**; over-emphasizing them harms the representation. In retrieval tasks (e.g., DPR), non-answer passages retrieved by BM25 are commonly used as hard negatives.

Q13: ***What are alignment and uniformity?***

- Wang & Isola (2020) decompose the contrastive loss into two properties (with representations on the unit hypersphere).
- **Alignment**: representations of positive pairs should be close:

$$\mathcal{L}_{align}=\mathbb{E}_{(x,y)\sim p_{pos}}\|f(x)-f(y)\|_2^2$$

- **Uniformity**: all representations should be distributed as uniformly as possible on the hypersphere, preserving maximal information:

$$\mathcal{L}_{uniform}=\log\mathbb{E}_{x,y\sim p_{data}}e^{-t\|f(x)-f(y)\|_2^2}$$

- The numerator of InfoNCE mainly optimizes alignment, while the denominator (negatives) mainly optimizes uniformity.
- Both metrics can be used directly to diagnose representation quality: alignment without uniformity means collapse; uniformity without alignment means the representation is not invariant to augmentations.

Q14: ***What is dimensional collapse?***

- **Complete collapse**: all representations converge to a single point.
- **Dimensional collapse**: representations do not collapse to a point but occupy only a **low-dimensional subspace** of the embedding space; many dimensions are unused, and many singular values in the spectrum of the embedding covariance matrix are close to 0.
- Jing et al. (2021, "Understanding Dimensional Collapse in Contrastive Self-supervised Learning") showed that dimensional collapse happens even in contrastive learning with negatives, due to factors such as strong data augmentation and implicit regularization in multi-layer networks.
- Diagnosis: inspect the **singular value spectrum** or effective rank of the embedding covariance matrix.
- Mitigation: the decorrelation terms of Barlow Twins/VICReg, projection head design (e.g., DirectCLR), whitening (W-MSE), etc.

Q15: ***How are self-supervised/contrastive representations evaluated?***

- **Linear probe**: freeze the encoder and train only a linear classifier on its features, measuring **linear separability**; this is the most common standard evaluation protocol.
- **kNN classification**: classify directly with nearest-neighbor voting on frozen features, no training needed; DINO commonly reports this metric.
- **Fine-tuning**: unfreeze all parameters and train on the downstream task, measuring the value as an initialization.
- **Semi-supervised / few-shot**: fine-tune using only 1% or 10% of labels.
- **Transfer learning**: transfer to other tasks and datasets such as detection and segmentation.
- Note: different protocols can lead to different conclusions — contrastive methods usually excel at linear probing, whereas masked modeling methods such as MAE excel at fine-tuning but are weaker at linear probing.

Q16: ***Why does SimCLR use the features before the projection head downstream?***

- The contrastive loss requires representations to be **invariant** to data augmentation, so the projection head output $z$ discards information changed by augmentation, such as color and orientation.
- This information may be useful downstream; $h$ before the projection head retains more of it.
- SimCLR's experiments show that a linear classifier trained on $h$ is significantly better than one trained on $z$; moreover, predicting which augmentation was applied is less accurate from $z$ and more accurate from $h$, supporting this explanation.
- Another explanation: the projection head absorbs the dimensional collapse induced by the loss, keeping the backbone features at a higher effective rank.

Q17: ***What are the characteristics of contrastive learning compared with supervised learning and masked modeling?***

- **Compared with supervised learning**:
    - No labels are needed, so massive unlabeled data can be used;
    - It learns instance-level discrimination (each sample is its own class) rather than predefined categories;
    - When transferred to downstream tasks such as detection, it often matches or even surpasses supervised pre-training.
- **Compared with masked modeling (MAE, BEiT)**:
    - Contrastive learning learns **global, invariant** semantic features with strong linear separability;
    - It depends on carefully designed data augmentation and negatives/anti-collapse mechanisms;
    - Masked modeling retains more **spatial detail**, is better suited to fine-tuning and dense prediction, and scales more easily.
- The two families are converging, e.g., iBOT and DINOv2 use both self-distillation and masked objectives.
