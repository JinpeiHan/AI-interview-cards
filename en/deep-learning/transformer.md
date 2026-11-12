# Transformer-Related Algorithm Interview Questions

Q1: ***Why does the Transformer use multi-head attention?***

- Multiple heads ensure that the Transformer can attend to information in different subspaces and capture richer feature information. It is analogous to using multiple channels simultaneously in a CNN; intuitively, multi-head attention helps the network capture richer features/information.

Q2: ***In Transformer self-attention, why are Q and K generated with different weight matrices? Why can't the same value be used for a dot product with itself?***

- Using different weight matrices to generate Q, K and V ensures that the word embeddings are projected into different spaces, which strengthens expressive power and improves generalization

Q3: **Why is the score divided by the square root of dk before the softmax when computing self-attention?**

- Why is it divided by the square root of dk before the softmax when computing self-attention (to scale the gradients and alleviate the vanishing-gradient problem; the square root of dk is an empirically chosen parameter)

Q4: ***Where is the parallelism of the Transformer reflected?***

- During training, the encoder and decoder can be trained in parallel (via teacher forcing and the sequence mask), but Transformer inference cannot be parallelized; it requires step-by-step autoregressive inference, similar to an RNN)

Q5: ***What issues need attention when deploying Transformers in the audio/video domain?***

- Because attention computation requires global information, it causes high system latency and huge storage overhead; chunk-wise attention needs to be designed for streaming decoding

Q6: ***The mask mechanism in the Transformer***

- The Transformer contains a padding mask and a sequence mask. The purpose of the former is to keep the padded parts (zero-padded when not long enough) out of the attention operation; the purpose of the latter is to ensure that when the decoder generates the probability distribution of the current word, it only sees past information and not future information (ensuring consistency between training and testing)

Q7: ***Why does the Transformer generate Q and K with different weight matrices? Why can't the same value be used for a dot product with itself?***

- Using different Q/K/V ensures projection into different spaces, which strengthens expressive power and improves generalization.
- At the same time, due to the properties of the softmax function, what it essentially does is a soft version of the arg max operation, producing a vector close to a one-hot vector (how close depends on the magnitude of the numbers). If Q=K, the resulting model would very likely produce an attention matrix similar to the identity matrix, so self-attention would degenerate into a point-wise linear mapping. At the very least this violates the original design intent.

Q8: ***Why does the Transformer use the dot product rather than addition when computing attention? What are the differences in computational complexity and effectiveness?***

- The dot product of K and Q is used to obtain an attention score matrix, which is used to refine V. K and Q are computed with different W_k and W_Q, which can be understood as projections into different spaces. Precisely because of these projections into different spaces, expressive power is increased, and the computed attention score matrix generalizes better.
- For faster computation. Matrix addition is indeed simple as far as the addition itself is concerned, but when computing attention as a whole it is equivalent to a hidden layer, and the overall computational cost is similar to the dot product. In terms of effectiveness, experimental analysis shows that the performance of both is related to dk: the larger dk is, the more pronounced the advantage of addition.

Q9: ***Why is the dimensionality of each head reduced in multi-head attention?***

- The original high-dimensional space is transformed into multiple low-dimensional spaces that are concatenated at the end to form an output of the same dimension, thereby enriching the feature information

Q10: ***Briefly introduce the Transformer's positional encoding. What is its significance, and what are its pros and cons?***

- Because self-attention is position-agnostic, no matter what the order of the sentence is, the hidden embeddings of the tokens computed by self-attention are the same, which clearly does not match human thinking. Therefore there needs to be a way to express a token's position information in the model; the Transformer uses a fixed positional encoding to represent the absolute position of a token in the sentence.

Q11: ***Why does the Transformer use LayerNorm rather than BatchNorm? Where is LayerNorm located in the Transformer?***

- LayerNorm normalizes along the feature dimension, while BatchNorm normalizes along the sample dimension. LayerNorm is used because the text sequences processed by the Transformer are usually variable-length, and the data distribution within a batch varies greatly, so normalizing along the sample dimension makes little sense; by comparison, LayerNorm, which normalizes the features of each sequence itself, is more stable
- LayerNorm is located between the multi-head attention layer and the activation-function layer

Reference: [https://zhuanlan.zhihu.com/p/363466672](https://zhuanlan.zhihu.com/p/363466672)

Q12: ***Briefly describe the BatchNorm technique and its pros and cons***

- BN (batch normalization) normalizes each batch of data before it enters the activation function; it can improve convergence speed, prevent overfitting, prevent vanishing gradients, and increase the network's sensitivity to the data

Q14: ***In the Transformer, how do the encoder and decoder interact?***

- Through the encoder-decoder attention: the encoder output serves as K and V, while Q comes from the decoder

Q15: **In the Transformer, *what is the difference between the multi-head self-attention in the decoder and the multi-head self-attention in the encoder?***

- The decoder has two MHA layers and the encoder has one; the decoder's second MHA layer serves to convert between input and output sentence lengths. In the decoder, the second-to-last dimension of q, k and v can differ, but in the encoder the dimensions of q, k and v are the same.
- The decoder's attention adds a sequence mask so that the model cannot see future information, ensuring the causality of the overall system

Q16: ***Why doesn't BERT's mask adopt the Transformer's trick of masking scores in attention (sequence mask)?***

- BERT and the Transformer have different objectives: BERT is a pre-trained language model that needs to fully consider context relationships, whereas the Transformer needs to ensure causality, mainly considering the relationship between the i-th element in a sentence and the previous i-1 elements.

Q17: ***Briefly describe the WordPiece model and the BPE (byte pair encoding) method***

- The traditional tokenization method in NLP splits on whitespace to obtain a fixed vocabulary; the resulting problem is that rare words cannot be handled, i.e. the OOV (Out of Vocabulary) problem. Traditional tokenization also makes it hard for the model to learn relationships between affixes or between different tenses of a word. For example, the relationships among walked, walking and walker cannot be generalized to talked, talking and talker. WordPiece and BPE are both subword methods that solve the above problems
- Byte Pair Encoding is a data compression method that splits words into subwords; specifically, the most frequent pair of consecutive bytes in the string is replaced with a byte that does not occur in the data, separating the meaning of the word itself from its tense
- The WordPiece algorithm can be seen as a variant of BPE. The difference is that WordPiece generates new subwords based on probability rather than on the next most frequent byte pair

Reference: [https://medium.com/towards-data-science/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0](https://medium.com/towards-data-science/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0)

Q18: ***Explain self-attention. How is it similar to and different from other attention mechanisms?***

- Self-attention can be seen as a special case of general attention. In self-attention, each word (token) in the sequence computes attention with the remaining words (tokens) in that sequence; concretely, the Q, K and V matrices all come from the same input. The characteristic of self-attention is that it **"ignores the distance between words (tokens) and directly computes dependencies, and can therefore learn the internal structure of the sequence"**

Q19: ****What advantages does the Transformer have over RNN/LSTM? Why?****

- RNN-family models have poor parallel computing capability; the computation of an RNN at the current time step depends on the hidden-layer result of the previous time step, so RNN training cannot be parallelized; the Transformer, during training, can be trained in parallel via sequence masking and teacher forcing
- Mainstream experiments have shown that the Transformer's feature extraction capability is better than that of RNN-family models

Q20: ***Besides absolute positional encoding, what other positional encoding techniques are there?***

- **Relative positional encoding** (RPE), which comes in three kinds: 1. adding a trainable parameter representing relative position when computing the attention score and the weighted value. 2. when generating multi-head attention, converting the absolute position of the key into a position relative to the query. 3. complex-valued functions: given a word's vector representation at some position, its vector representation at any position can be computed. The first two methods are word vector + positional encoding, which is a patch after the fact, whereas the complex-domain approach generates the corresponding position information when the word vector is generated.
- ****Learned positional encoding:**** learned positional encoding is similar to the method of generating word vectors; an independent vector is learned for each position

Q21: *****In BERT, what are the 3 cases into which token masking is divided, and what is the purpose of each?*****

- In BERT's Masked LM training task, the [MASK] token replaces 15% of the words in the corpus, which are then predicted at the last layer. But the [MASK] token does not appear in downstream tasks, causing a mismatch between pre-training and fine-tuning. To reduce the impact of this mismatch on the model, among the 15% of replaced tokens: 80% are replaced with the [MASK] token, 10% are replaced with a random token, and 10% are kept unchanged but still need to be predicted
- The first kind of replacement is the main part of Masked LM; it can fuse truly bidirectional semantic information without leaking the label;
- The second kind, random replacement: because the model must predict the true word at the randomly replaced token position at the last layer, and the model does not know that this position was randomly replaced, it is forced to learn a representation in the global context for every word, which also allows BERT to obtain better context-dependent word vectors (exactly the most important property for resolving polysemy);
- The third kind, keeping the token unchanged, means that in 10% of cases there really is leakage (15% * 10% = 1.5% of all words), which gives the model a certain bias, equivalent to an extra reward, pulling the model's representation of a word toward the word's true representation

Reference: https://www.jianshu.com/p/55b4de3de410

- See also: [BERT](bert.md) Q3

Q22: *****What are the differences among ELMo, GPT and BERT?*****

- Feature extractor: ELMo uses LSTMs for extraction, while GPT and BERT use Transformers. Many tasks show that the Transformer's feature extraction capability is stronger than the LSTM's; ELMo uses 1 layer of static vectors + 2 LSTM layers, so its multi-layer extraction capability is limited, while the Transformers in GPT and BERT can be stacked in many layers and have strong parallel computing capability.
- Unidirectional/bidirectional language model: GPT uses a unidirectional language model, while ELMo and BERT use bidirectional language models. However, ELMo is actually a concatenation of two unidirectional language models (in opposite directions), and this way of fusing features is weaker than BERT's integrated feature fusion.
- GPT and BERT both use the Transformer, which is an encoder-decoder structure; GPT's unidirectional language model uses the decoder part, and the decoder only sees incomplete sentences; BERT's bidirectional language model uses the encoder part, which uses complete sentences
- See also: [BERT](bert.md) Q12

Q23: ***Briefly describe the pros and cons of the BERT model***

- Pros: parallelism, solves long-range dependencies, bidirectional feature representation, strong feature extraction capability, effectively captures the global information of the context, alleviates vanishing gradients, etc.; BERT excels at NLU tasks.
- Cons: poor performance on generation tasks: the mismatch between the pre-training process and the generation process leads to poor results on generation tasks;
It adopts an independence assumption: it does not consider the correlation between predicted [MASK] tokens, which is a biased estimate of the language model's joint probability (not density estimation); the input noise introduced by masking causes a discrepancy between the pre-training and fine-tuning stages; it cannot handle document-level NLP tasks and is only suitable for sentence- and paragraph-level tasks;

Q24: ***Briefly describe Label Smoothing and its role***

- Since the training set contains a small amount of erroneous data, label smoothing changes the label of positive samples to 0.9 and the label of negative samples to 0.1 (the degree of smoothing can be adjusted as needed). This is a kind of **soft** learning, which tells the model **not to be so confident**
- ***Label Smoothing*** can improve the robustness and generalization of neural networks

Q25: ****What learning-rate warm-up strategy is used when training BERT? Why is it done this way?****

- Warm-up is equivalent to a process of adapting the learning rate. Because the direction of parameter updates at the beginning of training is important, to prevent some noisy samples from steering the update direction off course at the start of training, a number of warm-up steps is set, during which the learning rate at the current step is proportional to the current step count, i.e. the learning rate is (current_step/warm_up_step)*learning_rate.


Q26: ***How does the attention mechanism work? Why does attention achieve good results?***

- How it works: attention is essentially a **weighted sum**, assigning weights to the values (Value) according to the relevance between the query (Query) and each key (Key). Taking Scaled Dot-Product Attention as an example: $\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$, where $Q=XW_Q,K=XW_K,V=XW_V$; the steps are: compute similarity scores $QK^T$ → scale and apply softmax to get attention weights → take the weighted sum of $V$ to get the output
- It was first used in Seq2Seq machine translation: at each step the decoder computes attention over all encoder hidden states, dynamically selecting the source-side information relevant to the word currently being generated, which removes the information bottleneck of a fixed-length context vector
- Why it works well:
    - **Dynamic weights**: the weights are determined by the input content (content-based) rather than being fixed as in convolution, so the model can adaptively focus on key information and suppress irrelevant information
    - **Long-range dependencies**: the path length between any two positions is $O(1)$, unlike RNNs, which must pass information step by step, avoiding long-range information decay and vanishing gradients
    - **Global receptive field**: a single layer can model the global context
    - **Parallel computation**: attention for all positions can be computed simultaneously with matrix multiplication, giving high training efficiency and making it easy to scale to large models and large data
    - **Strong expressive power**: multi-head attention can capture different types of relationships (e.g. syntax, semantics, coreference) in different subspaces
    - **Interpretability**: attention weights can, to some extent, reflect which positions the model attends to
- Cost: computation and memory complexity are $O(n^2)$ in the sequence length; long sequences require optimizations such as sparse attention, linear attention and FlashAttention
