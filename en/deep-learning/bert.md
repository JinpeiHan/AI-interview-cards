# BERT

Q1: ***What are the architecture and sizes of BERT?***

- BERT (Devlin et al., 2018, "Bidirectional Encoder Representations from Transformers") is a pre-trained language model built **solely from stacked Transformer Encoders**; through self-attention every token can **see both left and right context at the same time**.
- Two standard sizes ($L$ = number of layers, $H$ = hidden size, $A$ = number of attention heads):
    - **BERT-Base**: $L=12,H=768,A=12$, about 110M parameters;
    - **BERT-Large**: $L=24,H=1024,A=16$, about 340M parameters.
- The FFN inner dimension is $4H$, the activation is **GELU**, and it uses a Post-LN structure.
- The pre-training corpora are BooksCorpus and English Wikipedia; the pre-training tasks are **MLM** and **NSP**.

Q2: ***How is BERT's input representation constructed? What does WordPiece do?***

- Input sequence format: `[CLS] sentence A [SEP] sentence B [SEP]`; the final input vector is the sum of three embeddings:

$$E=E_{token}+E_{segment}+E_{position}$$

- **Token Embedding**: WordPiece subword embeddings (English vocabulary of about 30k);
- **Segment Embedding**: distinguishes sentence A/B, with only two values;
- **Position Embedding**: **learnable absolute position embeddings**, up to 512 positions.
- **WordPiece** is a subword tokenization algorithm that merges subwords according to a criterion of maximizing the likelihood of the training corpus; e.g., "playing" → "play", "##ing" ("##" marks a word-internal continuation).
- Effects:
    - Keeps the vocabulary size manageable while essentially eliminating the **OOV** problem — any word can be split into subwords or even characters;
    - Subword sharing lets rare words and morphological variants reuse the representations of common subwords.
- For downstream token-level tasks (e.g., NER), one has to handle label alignment when a word is split into several subwords (usually only the first subword is used to predict the label).

Q3: ***Why does MLM use a 15% mask ratio? What is the deeper motivation for the 80/10/10 rule?***

- **The 15% trade-off**:
    - Too low: only a few positions per sequence produce a loss; the training signal is sparse and more training steps are needed;
    - Too high: too much context is destroyed, predictions lack sufficient evidence, and the gap from the complete downstream inputs grows.
- **Deeper motivation for the 80/10/10 rule** (of the selected 15%, 80% are replaced with [MASK], 10% with a random token, and 10% are left unchanged):
    - Addresses the **pre-train/fine-tune mismatch**: [MASK] never appears downstream; if [MASK] were always used, the model might only learn good predictive representations at [MASK] positions while paying insufficient attention to the representations of ordinary tokens;
    - Since the model **cannot know which positions have been tampered with**, it must maintain a context-based representation for **every token** rather than simply copying the input embedding;
    - Random replacement affects only $15\%\times10\%=1.5\%$ of all tokens, so the noise is small and does not noticeably harm language understanding.
- The ablation in the BERT paper shows that the mixed strategy is robust for both fine-tuning and feature-based usage, while the pure [MASK] strategy is clearly worse for feature-based (no fine-tuning) NER.
- Later work (e.g., Wettig et al., 2022, "Should You Mask 15% in Masked Language Modeling?") found that higher mask ratios can also work well for larger models; 15% is an empirical choice rather than an optimum.

Q4: ***What is the NSP task? Why did RoBERTa drop NSP?***

- **NSP (Next Sentence Prediction)**: given sentences A and B, 50% of the time B is the actual next sentence after A in the original text (IsNext), and 50% of the time it is a sentence randomly sampled from the corpus (NotNext); the output of [CLS] is used for binary classification. The goal is to make the model learn inter-sentence relationships to help tasks such as question answering and natural language inference.
- Reasons RoBERTa (Liu et al., 2019) dropped NSP:
    - Ablations showed that removing NSP and training on **contiguous full sentences (FULL-SENTENCES / DOC-SENTENCES)** packed to length 512 matches or improves downstream performance;
    - The NSP task is **too easy**: negatives come from different documents, so the model only needs to judge **whether the topics match** and does not learn real inter-sentence coherence (ALBERT pointed this out too and replaced it with SOP);
    - The sentence-pair format makes each input segment shorter, making it **hard for the model to learn long-range dependencies**.

Q5: ***How is BERT fine-tuned for downstream tasks?***

- A small task-specific output layer is added on top of BERT, and **all parameters are fine-tuned together**:
    - **Single-sentence / sentence-pair classification** (sentiment analysis, NLI): take the final hidden vector of [CLS] $C\in\mathbb{R}^H$ and add a softmax classification layer $P=\text{softmax}(CW^T)$;
    - **Sequence labeling** (NER, word segmentation): add a classification layer on each token's output vector to predict its label (optionally with a CRF layer to model label dependencies);
    - **Extractive QA** (SQuAD): input `[CLS] question [SEP] passage [SEP]` and introduce learnable start vector $S$ and end vector $E$; the probability that token $i$ is the answer start is $P_i=\frac{e^{S\cdot T_i}}{\sum_j e^{S\cdot T_j}}$, and similarly for the end; the span with $j\ge i$ maximizing $S\cdot T_i+E\cdot T_j$ is chosen as the answer;
    - **Multiple choice** (SWAG): each option is concatenated with the question and encoded separately; [CLS] is used to score each, followed by a softmax.
- Fine-tuning commonly uses a small learning rate (e.g., $2\times10^{-5}$ to $5\times10^{-5}$), a few epochs, with warmup and linear decay.

Q6: ***Why can [CLS] be used as a sentence representation? What are its limitations?***

- [CLS] sits at the beginning of the sequence and corresponds to no real word; through multiple layers of self-attention it can **aggregate information from the whole sentence**. BERT adds a **pooler layer** with tanh after the [CLS] output, pre-trained on the NSP task, so it is naturally suited as a classification feature.
- Limitations:
    - Without fine-tuning, the [CLS] vector is **not a good semantic sentence representation** — it was only trained on NSP; used directly for semantic similarity it often underperforms mean pooling of token outputs, and even averaged GloVe embeddings;
    - BERT's representation space suffers from **anisotropy**: vectors are concentrated in a narrow cone, cosine similarities are generally high, and discriminability is poor;
- Therefore sentence-embedding tasks generally use specially trained methods such as Sentence-BERT or SimCSE, or post-process the representations with whitening (BERT-whitening) or flow transformations (BERT-flow).

Q7: ***What improvements does RoBERTa make over BERT?***

- RoBERTa ("A Robustly Optimized BERT Pretraining Approach") **does not change the model architecture**; instead it shows that BERT was undertrained and improves results through a better training recipe:
    - **Dynamic masking**: a new mask is generated every time a sequence is fed to the model, instead of fixing the mask during data preprocessing (BERT duplicated the data 10 times with different masks, which is static masking);
    - **Removing NSP** and filling the input with contiguous full sentences;
    - **Larger batches** (e.g., 8k) and larger learning rates;
    - **More data** (about 160GB of text vs. about 16GB for BERT) and **longer training**;
    - **Byte-level BPE** (vocabulary of about 50k), which can encode any text without preprocessing and has no unknown characters.
- Lesson: the performance of a pre-trained model depends heavily on **data size, number of training steps and hyperparameters**, not only on the architecture and pre-training objective.

Q8: ***How does ALBERT reduce the number of parameters?***

- ALBERT (Lan et al., 2019) has three main improvements:
    - **Factorized Embedding Parameterization**: in BERT the embedding size $E$ equals the hidden size $H$, giving $V\times H$ embedding parameters. ALBERT first maps words into a low-dimensional space $E$ (e.g., 128) and then projects to $H$, reducing parameters from $O(V\times H)$ to $O(V\times E+E\times H)$. Rationale: word embeddings learn **context-independent** representations while hidden layers learn **context-dependent** ones, and the latter need more capacity;
    - **Cross-layer parameter sharing**: all Transformer layers share one set of parameters (both attention and FFN), so the parameter count is independent of depth, and the transitions between layers' inputs and outputs become smoother;
    - **SOP (Sentence Order Prediction)**: replaces NSP; positives are two consecutive segments from the same document, negatives are the same pair with **their order swapped**, forcing the model to learn inter-sentence **coherence** rather than topic.
- Note: ALBERT reduces **parameter count and memory**, but **not computation** (the number of layers is unchanged and each layer must still be computed), so inference is not faster; ALBERT-xxlarge is even more computationally expensive than BERT-Large.

Q9: ***How does DistilBERT perform knowledge distillation?***

- DistilBERT (Sanh et al., 2019) performs task-agnostic distillation **during pre-training**:
    - The student network has **half the layers** (6 layers) with the same hidden size, and drops the token-type embeddings and the pooler;
    - **The student is initialized from the teacher's layers**: one teacher layer out of every two is taken to initialize the student;
    - The loss has three parts:
        1. **Distillation loss**: cross-entropy between the student's and teacher's soft target distributions over MLM, using temperature $T$ to soften the softmax: $p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$;
        2. **MLM loss**: the standard masked language modeling loss;
        3. **Cosine embedding loss**: aligns the directions of the student's and teacher's hidden states.
- Result: a smaller model with faster inference that retains most of the teacher's language understanding ability.
- Other BERT compression approaches: TinyBERT (distilling intermediate layers and attention matrices, with two-stage distillation during pre-training and task fine-tuning), MobileBERT, pruning, quantization, etc.

Q10: ***What is ELECTRA's Replaced Token Detection (RTD)? Why is it more efficient?***

- ELECTRA (Clark et al., 2020) consists of two networks:
    - **Generator**: a small MLM that samples replacement tokens for masked positions;
    - **Discriminator**: determines for **every token** in the input sequence whether it is original or was replaced by the generator (binary classification), i.e., **Replaced Token Detection**.
- Total loss: $\mathcal{L}=\mathcal{L}_{MLM}(\theta_G)+\lambda\mathcal{L}_{Disc}(\theta_D)$; after pre-training, only the discriminator is kept for downstream tasks.
- Why it is more efficient:
    - MLM computes the loss only on the 15% masked positions, whereas RTD provides a learning signal on **all tokens**, giving higher sample efficiency;
    - The discriminator's input contains no [MASK], **reducing the pre-train/fine-tune mismatch**.
- Details:
    - The generator is trained with **maximum likelihood**, not adversarially (it samples discrete tokens, so gradients cannot flow back from the discriminator); hence it is not a GAN;
    - The generator should not be too strong, or the discrimination task becomes too hard; the paper found a generator 1/4–1/2 the size of the discriminator works best;
    - The generator and discriminator share word embeddings; if the generator happens to produce the original token, that position is labeled "original".

Q11: ***What is DeBERTa's Disentangled Attention?***

- One of DeBERTa's (He et al., 2020) core improvements is **disentangled attention**: each token is represented by two vectors — a **content vector** $H_i$ and a **relative position vector** $P_{i|j}$. The attention score is decomposed as:

$$A_{i,j}=\underbrace{H_iH_j^T}_{\text{content-to-content}}+\underbrace{H_iP_{j|i}^T}_{\text{content-to-position}}+\underbrace{P_{i|j}H_j^T}_{\text{position-to-content}}$$

- The position-to-position term carries little information with relative positions and is omitted; the scaling factor of the attention score is changed accordingly to $\frac{1}{\sqrt{3d}}$.
- **Enhanced Mask Decoder (EMD)**: since the backbone uses only relative positions, DeBERTa incorporates **absolute position** information just before the final softmax prediction layer (e.g., to distinguish the syntactic roles of store and mall in "a new **store** opened beside the new **mall**");
- During fine-tuning, virtual adversarial training (SiFT) is used to improve robustness.
- DeBERTa v3 switches the pre-training objective to ELECTRA-style RTD and proposes **gradient-disentangled embedding sharing (GDES)** to solve the "tug-of-war" problem when the generator and discriminator share embeddings.

Q12: ***What are the core design differences between BERT and GPT?***

- **Architecture**: BERT is encoder-only with **bidirectional** self-attention; GPT is decoder-only with **causal (unidirectional) masked** self-attention.
- **Pre-training objective**: BERT is a **denoising autoencoder** (MLM, predicting masked words); GPT is an **autoregressive language model** (predicting the next word), $P(x)=\prod_tP(x_t|x_{<t})$.
- **Strengths**: BERT excels at **understanding tasks** (classification, extraction, matching), since each token's representation fuses global context; GPT excels at **generation tasks** and, as it scales, exhibits abilities such as in-context learning.
- **Training efficiency**: MLM produces a loss on only about 15% of positions per sequence; an autoregressive LM produces a loss at every position.
- **Usage**: BERT usually needs fine-tuning for each task; GPT models can perform zero-/few-shot tasks via prompting.
- Unifying attempts: XLNet (permutation language modeling), T5 (encoder-decoder, span corruption), UniLM (unifying bidirectional/unidirectional/seq2seq via different attention masks).

Q13: ***Why is BERT not good at text generation?***

- **MLM is not a valid autoregressive factorization**: BERT learns $P(x_i|x_{\setminus i})$, i.e., predicting one word given all other words; it cannot directly give the joint probability $P(x)$ of the whole sequence, so it cannot sample word by word from left to right.
- **Independence assumption**: when multiple words are masked simultaneously, BERT's predictions at each masked position are **mutually independent**, so generating multiple words easily produces incoherent combinations (as pointed out in the XLNet paper).
- **Bidirectional attention cannot be cached**: each time a word is added during generation, the representations of all positions change, so incremental computation with a KV cache (as in causal models) is impossible, making it inefficient.
- **Pre-training/generation mismatch**: BERT was never trained under the condition of "only left context".
- Workarounds: Gibbs-sampling-style iterative filling, non-autoregressive generation (Mask-Predict), using BERT to initialize the encoder of a seq2seq model — but results are generally worse than native generative models.

Q14: ***Why is BERT's maximum length 512? How are long texts handled?***

- **Reasons**:
    - Self-attention compute and memory are $O(n^2)$, so longer sequences are more expensive;
    - Position embeddings are **learnable absolute position embeddings** trained for only 512 positions; positions beyond 512 have no embedding and cannot be extrapolated directly;
    - For efficiency, pre-training used sequences of length 128 for the first 90% of steps and length 512 only for the last 10%.
- **Ways to handle long texts**:
    - **Truncation**: keep the beginning, the end, or a concatenation of both (e.g., head+tail);
    - **Sliding window**: encode in overlapping segments and fuse the results (doc stride in SQuAD);
    - **Hierarchical models**: encode paragraphs separately first, then aggregate paragraph representations with an upper-level model;
    - **Long-text models**: Longformer, BigBird (local window + global + random sparse attention);
    - **Improved position encoding**: relative position encodings or RoPE, which extrapolate more easily; newer encoders such as ModernBERT support longer contexts.

Q15: ***How does Sentence-BERT obtain sentence embeddings? Why not use BERT directly?***

- **Problems with using BERT directly**:
    - BERT computes sentence-pair similarity as a **cross-encoder**: two sentences are concatenated and fed in together, which is accurate but requires a full forward pass for every pair. Finding the most similar pair among $n=10000$ sentences requires $\frac{n(n-1)}{2}\approx5\times10^7$ inferences, which is impractical for large-scale retrieval;
    - Sentence vectors obtained directly from [CLS] or mean pooling are of poor quality (see Q6).
- **Sentence-BERT (Reimers & Gurevych, 2019)**:
    - Uses a **Siamese / bi-encoder** architecture: the two sentences pass separately through a BERT with **shared parameters**, followed by **mean pooling** to obtain sentence vectors $u$ and $v$;
    - Trained with a classification objective on NLI data: concatenate $(u,v,|u-v|)$ and feed it to a softmax classifier; on STS regression tasks it directly optimizes cosine similarity; a triplet loss can also be used;
    - At inference, sentence vectors can be **precomputed and indexed**, and similarity is computed with cosine distance, making it suitable for semantic retrieval and clustering.
- Follow-ups: **SimCSE** uses contrastive learning (in the unsupervised version, the same sentence passed twice with different dropout forms a positive pair), further improving sentence-vector quality and mitigating anisotropy; retrieval systems commonly combine "bi-encoder recall + cross-encoder reranking".

Q16: ***What is BERT's anisotropy problem? How can it be mitigated?***

- **Phenomenon**: BERT's output word/sentence vectors are distributed in a narrow cone, so the cosine similarity between any two sentences is high, making semantic similarity computation poorly discriminative.
- **Causes**: related to word frequency — embeddings of high-frequency words lie close to the origin while low-frequency words are sparsely distributed, so the embedding space is non-uniform; the softmax training objective and weight tying also cause representation degeneration.
- **Mitigation**:
    - **Post-processing**: BERT-flow (mapping the distribution to an isotropic Gaussian with a normalizing flow), BERT-whitening (whitening so that the covariance matrix becomes the identity);
    - **Contrastive learning**: methods such as SimCSE improve **uniformity** by pushing negatives apart;
    - Supervised sentence-embedding training (Sentence-BERT).

Q17: ***What are common problems and tricks when fine-tuning BERT?***

- **Catastrophic forgetting and overfitting**: full fine-tuning on small datasets overfits easily; use a smaller learning rate, few epochs, early stopping, or **layer-wise learning-rate decay** (lower learning rates for bottom layers, since they learn more general features).
- **Training instability**: when fine-tuning BERT-Large on small datasets, results vary greatly across random seeds and runs sometimes fail; this can be mitigated with warmup, using Adam with bias correction, training for more steps, and re-initializing the top few layers.
- **Domain adaptation**: **continue pre-training** on downstream-domain corpora (domain-adaptive pretraining), e.g., BioBERT, SciBERT, FinBERT.
- **Parameter-efficient fine-tuning**: Adapter, Prefix-Tuning, LoRA, etc. train only a small number of parameters, suitable for multi-task deployment.
- **Class imbalance**: use weighted cross-entropy, Focal Loss or resampling.
