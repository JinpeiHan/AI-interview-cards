# BERT

Q1: ***What are the architecture and sizes of BERT?***

- **BERT is a pre-trained language model built only from stacked Transformer encoders; bidirectional self-attention lets every token see both its left and right context.**
- BERT-Base: 12 layers, hidden size 768, 12 heads, about 110M parameters; BERT-Large: 24 layers, 1024, 16 heads, about 340M parameters.
- The FFN is 4x the hidden size, uses GELU, and the blocks are Post-LN.
- It is pre-trained on BooksCorpus and English Wikipedia with two tasks, MLM and NSP.

Q2: ***How is BERT's input representation constructed? What does WordPiece do?***

- **The input is `[CLS] sentence A [SEP] sentence B [SEP]`, and each position's input vector is the sum of a token embedding, a segment embedding and a learned absolute position embedding (up to 512 positions).**
- WordPiece is a subword tokenizer that splits words into common pieces, e.g. "playing" → "play", "##ing".
- Benefits: a controllable vocabulary (about 30k), almost no OOV, and rare or inflected words reuse representations of common subwords.
- For token-level tasks like NER you must align labels with subwords; usually only the first subword predicts the label.

Q3: ***Why does MLM use a 15% mask ratio? What is the deeper motivation for the 80/10/10 rule?***

- **15% is an empirical trade-off: too low gives a sparse training signal, too high destroys too much context and widens the gap to the full inputs seen downstream.**
- **80/10/10 (80% replaced by [MASK], 10% by a random token, 10% unchanged) reduces the pretrain-finetune mismatch: [MASK] never appears downstream, and because the model cannot tell which positions were altered, it must learn a good contextual representation for every token.**
- Random replacement affects only 1.5% of all tokens, so the noise is small.
- Later work showed larger models do well with higher mask ratios, so 15% is not optimal.

Q4: ***What is the NSP task? Why did RoBERTa drop NSP?***

- **NSP asks the model whether sentence B really follows sentence A (50% true next sentence, 50% random sentence), as a binary classification on [CLS], to learn sentence-level relations.**
- RoBERTa dropped it because training on contiguous full sentences that fill all 512 tokens matched or beat NSP.
- NSP is too easy: negatives come from other documents, so judging whether the topic matches is enough and real coherence is not learned (ALBERT replaced it with SOP).
- The sentence-pair format shortens each segment, which hurts learning long-range dependencies.

Q5: ***How is BERT fine-tuned for downstream tasks?***

- **Add a very small task-specific output layer on top of BERT and fine-tune all parameters together with a small learning rate.**
- Classification / sentence-pair tasks: feed the [CLS] output into a softmax classifier.
- Sequence labeling (NER): put a classifier on every token output, optionally with a CRF.
- Extractive QA: learn a start vector and an end vector, dot them with each token output, and pick the best-scoring valid span.
- Typical learning rate is 2e-5 to 5e-5 for a few epochs, with warmup and linear decay.

Q6: ***Why can [CLS] be used as a sentence representation? What are its limitations?***

- **[CLS] corresponds to no real word, so many layers of self-attention let it aggregate the whole sentence, and it was trained on NSP, which makes it a good classification feature after fine-tuning.**
- Without fine-tuning it is a poor semantic vector; for similarity, mean pooling of token outputs often works better.
- BERT's representation space is anisotropic: vectors sit in a narrow cone and cosine similarities are uniformly high.
- So sentence embeddings usually come from dedicated methods such as Sentence-BERT or SimCSE, or from post-processing like whitening.

Q7: ***What improvements does RoBERTa make over BERT?***

- **RoBERTa keeps the architecture and shows BERT was undertrained, improving results purely through a better training recipe.**
- Dynamic masking: a new mask is generated each time a sequence is fed in, instead of being fixed during preprocessing.
- NSP is removed; inputs are filled with contiguous full sentences.
- Larger batches, more data (about 160GB vs 16GB) and longer training.
- Byte-level BPE (about 50k vocabulary), so there are no unknown characters.

Q8: ***How does ALBERT reduce the number of parameters?***

- **ALBERT cuts parameters in two main ways: factorizing the word embedding into a low-dimensional embedding plus a projection, and sharing one set of parameters across all Transformer layers.**
- Embedding factorization: words map to 128 dimensions and are then projected to the hidden size, since context-free word embeddings do not need that much capacity.
- Cross-layer sharing: the parameter count no longer grows with depth.
- It also replaces NSP with SOP (were the two segments swapped?), forcing the model to learn coherence rather than topic.
- Note: it saves parameters and memory, but compute and inference speed do not improve.

Q9: ***How does DistilBERT perform knowledge distillation?***

- **DistilBERT does task-agnostic distillation during pre-training: a student with half the layers (6) learns to imitate the output distribution of the teacher BERT.**
- The student is initialized from every other layer of the teacher.
- The loss has three parts: a temperature-softened distillation loss on soft labels, the standard MLM loss, and a cosine loss aligning hidden-state directions.
- The result is smaller and faster while keeping most of the teacher's ability; related methods include TinyBERT, MobileBERT, pruning and quantization.

Q10: ***What is ELECTRA's Replaced Token Detection (RTD)? Why is it more efficient?***

- **ELECTRA uses a small generator (a small MLM) to replace some tokens with sampled words, then trains a discriminator to tell for every token whether it is original or replaced; only the discriminator is kept after pre-training.**
- More efficient: MLM has a loss on only 15% of positions, while RTD gives a learning signal on all tokens.
- The discriminator never sees [MASK], which reduces the pretrain-finetune mismatch.
- The generator is trained with maximum likelihood, not adversarially, so it is not a GAN; it should not be too strong, and 1/4 to 1/2 of the discriminator size works best.

Q11: ***What is DeBERTa's Disentangled Attention?***

- **DeBERTa represents each token with separate content and relative-position vectors, and splits the attention score into three summed terms: content-to-content, content-to-position and position-to-content.**
- The position-to-position term carries little information and is dropped.
- Because the backbone uses only relative positions, absolute positions are added just before the prediction layer (the Enhanced Mask Decoder, EMD).
- DeBERTa v3 switches to an ELECTRA-style RTD objective and uses gradient-disentangled embedding sharing to fix the generator-discriminator "tug-of-war".

Q12: ***What are the core design differences between BERT and GPT?***

- **BERT is a bidirectional encoder pre-trained by denoising (MLM) and is good at understanding; GPT is a unidirectional decoder pre-trained autoregressively (next-token prediction) and is good at generation.**
- Every BERT token fuses the global context, which suits classification, extraction and matching.
- MLM has a loss on only about 15% of positions; an autoregressive LM has a loss on every position.
- BERT usually needs per-task fine-tuning; GPT at scale can do zero-/few-shot tasks via prompting.
- Attempts to unify them: XLNet, T5, UniLM.

Q13: ***Why is BERT not good at text generation?***

- **BERT learns to predict one word given all the others, not a left-to-right factorization of the joint probability, so it cannot naturally sample text word by word.**
- When several words are masked, their predictions are independent, which produces incoherent combinations.
- With bidirectional attention every new word changes all representations, so there is no KV cache and generation is slow.
- It was never trained with left-context only; workarounds like iterative filling work only moderately well.

Q14: ***Why is BERT's maximum length 512? How are long texts handled?***

- **Because self-attention cost grows quadratically with length, and BERT uses learned absolute position embeddings trained for only 512 positions, so longer positions have no embedding.**
- Truncation: keep the head, the tail, or head plus tail.
- Sliding window: encode overlapping chunks and merge the results.
- Hierarchical models: encode paragraphs first, then aggregate them with an upper model.
- Long-context models: Longformer, BigBird (sparse attention), or newer encoders with extrapolation-friendly positions like RoPE (e.g. ModernBERT).

Q15: ***How does Sentence-BERT obtain sentence embeddings? Why not use BERT directly?***

- **Comparing sentences with plain BERT means concatenating each pair and running them together (a cross-encoder), which is far too expensive for large-scale search, and the raw [CLS] or mean-pooled vectors are poor.**
- Sentence-BERT encodes the two sentences separately with a weight-shared siamese BERT and mean-pools to get sentence vectors.
- It trains on NLI by classifying the concatenation $(u, v, |u-v|)$, or on STS by directly optimizing cosine similarity.
- Vectors can be precomputed and indexed for cosine search; in practice a bi-encoder retrieves and a cross-encoder re-ranks.
- Later, SimCSE used contrastive learning to improve sentence vectors further.

Q16: ***What is BERT's anisotropy problem? How can it be mitigated?***

- **Anisotropy means BERT's vectors are packed into a narrow cone, so any two sentences have high cosine similarity and semantic discrimination is poor.**
- The cause relates to word frequency: frequent and rare word embeddings are unevenly distributed, and the training objective also degrades the representations.
- Post-processing: BERT-flow (map to an isotropic Gaussian) and BERT-whitening.
- Training-based: contrastive methods such as SimCSE improve uniformity, or supervised training such as Sentence-BERT.

Q17: ***What are common problems and tricks when fine-tuning BERT?***

- **The common problems are overfitting on small data and unstable training, mainly handled with a small learning rate, few epochs, warmup and layer-wise learning rates.**
- Layer-wise learning-rate decay: lower layers learn general features and get smaller learning rates.
- BERT-Large on small data is sensitive to the random seed; use bias-corrected Adam, more steps, or re-initialize the top few layers.
- For a large domain gap, continue pre-training on in-domain text first (e.g. BioBERT, SciBERT).
- For multi-task deployment use parameter-efficient methods like Adapters or LoRA; for class imbalance use weighted loss or Focal Loss.
