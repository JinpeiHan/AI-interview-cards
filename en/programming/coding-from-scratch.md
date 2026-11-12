# Coding from Scratch

> Conventions: `import numpy as np`, `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `import math`. Later questions reuse functions defined earlier (e.g. `softmax`, `sigmoid`, `box_iou`).

## Basic Functions and Losses

Q1: ***Implement numerically stable softmax and log-softmax***

- **Subtract the max before exponentiating: softmax is shift-invariant, so the largest exponent becomes 1 and nothing overflows. Compute log-softmax directly with log-sum-exp; do not take the log of the softmax output (an underflowed 0 gives `-inf`).**
- Pitfall: `keepdims=True` is needed for correct broadcasting; for batched input take the max along the class axis.

```python
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
```

Q2: ***Implement cross-entropy loss from logits, and its gradient***

- **Compute log-softmax directly from the logits, pick the correct class's value and take the negative mean; the gradient is just "softmax probabilities minus one-hot", divided by N.**
- Pitfall: use `logp[np.arange(N), y]` to pick the correct class; `logp[:, y]` gives an N×N matrix.
- Key formula: $\frac{\partial L}{\partial z} = \frac{1}{N}(\mathrm{softmax}(z) - \mathrm{onehot}(y))$

```python
def cross_entropy(logits, y):
    """logits: (N, C), y: (N,) integer labels"""
    logp = log_softmax(logits, axis=1)
    return -logp[np.arange(len(y)), y].mean()

def cross_entropy_grad(logits, y):
    N = logits.shape[0]
    g = softmax(logits, axis=1)
    g[np.arange(N), y] -= 1
    return g / N
```

Q3: ***Implement numerically stable sigmoid and BCE (with logits)***

- **Write sigmoid as two branches by sign so the exponent's argument is always non-positive and never overflows; simplify BCE into a form that only involves $|z|$ and compute it directly from logits.**
- `np.log1p` is more accurate. This is why PyTorch recommends `BCEWithLogitsLoss` over `Sigmoid + BCELoss`.
- Key formula: $\ell(z,y) = \max(z,0) - zy + \log(1+e^{-|z|})$

```python
def sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1 / (1 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1 + ex)
    return out

def bce_with_logits(z, y):
    # max(z,0) - z*y + log(1+e^{-|z|})
    return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
```

## Classical Machine Learning

Q4: ***Implement linear regression (closed form + gradient descent)***

- **The closed form solves the normal equation directly (adding λ gives ridge regression) and suits small dimensions; gradient descent iterates along the MSE gradient and suits large scale.**
- Use `np.linalg.solve`, never an explicit inverse; do not regularize the bias; standardize features before gradient descent.
- Key formula: $w^*=(X^\top X+\lambda I)^{-1}X^\top y$

```python
def linreg_closed_form(X, y, lam=0.0):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])   # append bias column
    d = Xb.shape[1]
    reg = lam * np.eye(d); reg[-1, -1] = 0           # do not regularize the bias
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)

def linreg_gd(X, y, lr=0.1, epochs=1000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        err = X @ w + b - y
        w -= lr * (X.T @ err) / n
        b -= lr * err.mean()
    return w, b
```

Q5: ***Implement logistic regression with gradient descent***

- **Put a sigmoid on a linear model to output a probability and run gradient descent on BCE; the gradient has the same form as linear regression: "prediction minus label" times the features.**
- Pitfall: on linearly separable data without regularization the weights grow without bound, so add L2 or early stopping.
- Follow-up: MSE is not used because combined with sigmoid it is non-convex and its gradient vanishes in the saturated region.

```python
def logreg_fit(X, y, lr=0.1, epochs=1000, lam=0.0):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        g = p - y
        w -= lr * (X.T @ g / n + lam * w)   # L2 regularization
        b -= lr * g.mean()
    return w, b

def logreg_predict(X, w, b, thr=0.5):
    return (sigmoid(X @ w + b) >= thr).astype(int)
```

Q6: ***Implement K-means***

- **Alternate two steps: assign each point to its nearest center, then move each center to its cluster mean; the objective never increases, so it always converges to a local optimum.**
- Vectorize distances with the squared expansion; handle empty clusters; it is sensitive to initialization, so in practice use k-means++ with several restarts.

```python
def kmeans(X, k, n_iter=100, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(n_iter):
        d2 = (X**2).sum(1, keepdims=True) - 2 * X @ C.T + (C**2).sum(1)   # (n, k)
        labels = d2.argmin(1)
        C_new = C.copy()
        for j in range(k):
            mask = labels == j
            if mask.any():                  # empty cluster keeps its old center
                C_new[j] = X[mask].mean(0)
        shift = np.linalg.norm(C_new - C)
        C = C_new
        if shift < tol:
            break
    labels = ((X**2).sum(1, keepdims=True) - 2 * X @ C.T + (C**2).sum(1)).argmin(1)
    return C, labels
```

Q7: ***Implement KNN classification***

- **There is no training step; at prediction time compute distances to all training points, take the k nearest and do a majority vote.**
- Use `np.argpartition` to get the k smallest, faster than a full sort; for large data compute in chunks or use approximate nearest neighbors (FAISS, HNSW).
- Pitfall: the expansion can give tiny negative distances, so clamp to 0 before a square root; use an odd k for binary classification to avoid ties.

```python
def knn_predict(X_train, y_train, X_test, k=5):
    d2 = ((X_test**2).sum(1, keepdims=True) - 2 * X_test @ X_train.T
          + (X_train**2).sum(1))                                  # (m, n)
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
    neigh = y_train[idx]                                          # (m, k)
    return np.array([np.bincount(row).argmax() for row in neigh])
```

Q8: ***Implement PCA via SVD***

- **Center the data, then take the SVD of the data matrix: the right singular vectors are the principal directions, and the squared singular values divided by n−1 are the explained variances.**
- We skip computing the covariance because that squares the condition number and loses precision.
- Pitfalls: forgetting to center; new data must subtract the training mean; SVD signs are not unique.

```python
def pca(X, k):
    mu = X.mean(0)
    Xc = X - mu                                     # centering is mandatory
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]                             # (k, d)
    explained_var = S[:k]**2 / (len(X) - 1)
    Z = Xc @ components.T                           # (n, k), also equals U[:, :k] * S[:k]
    return Z, components, explained_var, mu
```

## Transformer Components

Q9: ***Implement scaled dot-product attention with a mask***

- **Dot Q with K and divide by $\sqrt{d_k}$, fill positions that must not be seen with `-inf`, softmax into weights, then take the weighted sum of V. Dividing by $\sqrt{d_k}$ stops large dot products from saturating the softmax.**
- Here `True` means may attend (as in `F.scaled_dot_product_attention`); the boolean mask of `nn.MultiheadAttention` is the opposite.
- Pitfalls: do not use `-1e9` in fp16 (out of range); a fully masked row gives nan after softmax and must be set to 0.

```python
def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=False):
    """q: (..., Tq, d), k/v: (..., Tk, d), mask: bool tensor broadcastable to (..., Tq, Tk)"""
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn)          # fully masked rows give nan in softmax; set to 0
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=training)
    return attn @ v, attn
```

Q10: ***How do you build a causal mask?***

- **Use a lower-triangular matrix: position i can only see j ≤ i. AND it with the padding mask and expand to (B, 1, T, T) to broadcast over all heads.**
- When generating one token at a time with a KV cache, the new query may see all history, so no causal mask is needed; with left padding, position ids must be shifted.

```python
def causal_mask(T, device=None):
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))   # (T, T)

def causal_padding_mask(pad_mask):
    """pad_mask: (B, T), True for real tokens -> (B, 1, T, T), broadcastable over heads"""
    T = pad_mask.size(1)
    return causal_mask(T, pad_mask.device) & pad_mask[:, None, None, :]
```

Q11: ***Implement a multi-head attention module***

- **One linear layer produces Q/K/V; split the hidden dimension into h heads, run attention in each, concatenate and apply the output projection. Compute is about the same as one head, but each head can attend to different patterns.**
- Pitfall: `view` to (B, T, h, d_k) first and then transpose; a direct `view(B, h, T, d_k)` mixes up tokens; after transposing, call `.contiguous()` before `view`.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)    # fused Q/K/V projection
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, h, T, d_k)
        q, k, v = (t.view(B, T, self.h, self.d_k).transpose(1, 2) for t in (q, k, v))
        y, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout, self.training)
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # merge heads
        return self.out(y)

# Usage: mha = MultiHeadAttention(512, 8); y = mha(x, causal_mask(x.size(1), x.device))
```

Q12: ***Implement applying RoPE (rotary position embedding)***

- **Pair up the dimensions of q and k and rotate each pair by angle $m\theta_i$ at position m, with a different frequency per pair; then q·k depends only on the relative position m−n. Apply it to q and k only, not v.**
- Pitfall: the interleaved and half-split pairing layouts are incompatible and must match when loading weights; with a KV cache, slice cos/sin at absolute positions.

```python
def rope_cache(T, d, base=10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=device).float() / d))   # (d/2,)
    t = torch.arange(T, device=device).float()
    freqs = torch.outer(t, inv_freq)                                                # (T, d/2)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    """x: (B, H, T, d); cos/sin: (T, d/2). Half-split layout (GPT-NeoX / HF LLaMA)"""
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# Applied to q and k only, not v: q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

## Normalization and Regularization

Q13: ***Implement LayerNorm***

- **For each sample, subtract the mean and divide by the standard deviation over the feature dimension, then multiply by γ and add β; it does not depend on the batch, so training and inference behave the same.**
- Pitfall: use the biased variance (`unbiased=False`); ε goes inside the square root.

```python
class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)    # biased variance
        return self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta
```

Q14: ***Implement BatchNorm (train/eval modes with running stats)***

- **In training, normalize with the current batch's mean and variance and update the running statistics with a moving average; at inference use only the running statistics.**
- Running stats are stored as buffers and updated under `no_grad`; running_var uses the unbiased variance.
- Pitfalls: forgetting `model.eval()` at inference uses batch statistics; BN is unstable with small batches, so switch to GroupNorm/LayerNorm.

```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # buffers: saved in state_dict and moved by .to(device), but not optimized
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):  # x: (N, C)
        if self.training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            n = x.size(0)
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * n / (n - 1), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
```

Q15: ***Implement RMSNorm***

- **Drop LayerNorm's mean subtraction and β: just divide by the root mean square and multiply by γ. Cheaper with comparable quality; the default in LLMs such as LLaMA.**
- Pitfall: in low precision the sum of squares loses precision or overflows, so compute in fp32 and cast back.

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)   # compute in fp32
        return (x.float() * rms).type_as(x) * self.weight
```

Q16: ***Implement dropout (inverted dropout)***

- **During training zero elements with probability p and divide the kept values by 1−p so the expectation is unchanged; at inference return the input as is.**
- Pitfalls: p is the drop probability, not the keep probability; backward must use the same mask.

```python
def dropout(x, p=0.5, training=True, rng=np.random.default_rng()):
    if not training or p == 0:
        return x
    mask = rng.random(x.shape) >= p          # keep with probability 1-p
    return x * mask / (1 - p)
```

## Detection and Evaluation

Q17: ***Implement IoU (batched)***

- **Intersection area divided by union area; the intersection's top-left corner is the larger of the two top-left corners, and its bottom-right is the smaller of the two bottom-right corners.**
- Pitfall: for disjoint boxes widths/heights must be clipped to 0, otherwise two negatives multiply into a positive "intersection".

```python
def box_iou(a, b):
    """a: (N,4), b: (M,4), format (x1,y1,x2,y2) -> (N,M)"""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])    # intersection top-left (N,M,2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])    # intersection bottom-right (N,M,2)
    wh = np.clip(rb - lt, 0, None)                     # 0 when boxes do not overlap
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
```

Q18: ***Implement NMS (non-maximum suppression)***

- **Go from highest to lowest score: each time keep the top-scoring box and delete boxes whose IoU with it exceeds the threshold, until none remain.**
- For multiple classes, run it per class; follow-up: Soft-NMS lowers scores instead of deleting, which helps with dense occlusion.

```python
def nms(boxes, scores, iou_thr=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep)
```

Q19: ***Implement AUC computation***

- **AUC is the probability that a random positive scores higher than a random negative. Compute it with ranks: the sum of the positives' ranks minus its smallest possible value, divided by the number of positive-negative pairs.**
- Pitfalls: ties get the average rank; AUC is undefined when only one class is present.
- Key formula: $\mathrm{AUC}=\frac{\sum_{i\in pos}\mathrm{rank}_i-\frac{n_+(n_++1)}{2}}{n_+n_-}$

```python
def average_ranks(s):
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2              # average 1-based rank of each tie group
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(avg, counts)
    return ranks

def roc_auc(y_true, scores):
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    ranks = average_ranks(scores)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

## Training and Optimization

Q20: ***Write a two-layer MLP training loop in NumPy with manual backprop***

- **Forward: linear, ReLU, linear, cross-entropy. Backward: start from the logits gradient of Q2, apply the chain rule layer by layer, then take a gradient descent step.**
- Checks: every gradient has its parameter's shape, bias gradients sum over the batch dimension, and a numerical gradient can verify them.
- Pitfalls: all-zero initialization cannot break symmetry; only `p -= ...` updates in place.

```python
def train_mlp(X, y, hidden=64, n_classes=3, lr=0.1, epochs=500, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.normal(0, np.sqrt(2 / d), (d, hidden)); b1 = np.zeros(hidden)          # He initialization
    W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, n_classes)); b2 = np.zeros(n_classes)
    for ep in range(epochs):
        # forward
        h_pre = X @ W1 + b1
        h = np.maximum(h_pre, 0)
        logits = h @ W2 + b2
        loss = cross_entropy(logits, y)
        # backward
        dlogits = cross_entropy_grad(logits, y)        # (n, C), already divided by n
        dW2 = h.T @ dlogits;  db2 = dlogits.sum(0)
        dh = dlogits @ W2.T
        dh_pre = dh * (h_pre > 0)                      # ReLU gradient
        dW1 = X.T @ dh_pre;   db1 = dh_pre.sum(0)
        # update (in place)
        for p, g in ((W1, dW1), (b1, db1), (W2, dW2), (b2, db2)):
            p -= lr * g
    return (W1, b1, W2, b2), loss
```

Q21: ***Implement one Adam / AdamW update step***

- **Adam keeps a first moment m and a second moment v of the gradient, bias-corrects them and updates with m/√v; AdamW applies weight decay directly to the parameters instead of adding it to the gradient (where the adaptive denominator would rescale it).**
- Bias correction offsets the early underestimate caused by m and v starting at 0; t must start at 1.
- Key formula: $\theta\leftarrow\theta-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$

```python
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params, self.lr, self.eps, self.wd = params, lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            if self.wd:
                p -= self.lr * self.wd * p              # decoupled weight decay
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

## Decoding

Q22: ***Implement top-k / top-p (nucleus) sampling***

- **Divide by the temperature, then filter: top-k keeps the k most likely tokens, top-p keeps the smallest set whose cumulative probability just exceeds p; set the rest to `-inf`, re-apply softmax and sample.**
- Pitfalls: decide removal by "cumulative probability excluding the token itself > p", which guarantees at least one token is kept; after sorting, `scatter` back to the original order.

```python
def sample_top_k_top_p(logits, k=0, p=1.0, temperature=1.0):
    """logits: (B, V) -> (B,) token ids"""
    logits = logits / max(temperature, 1e-5)
    if k > 0:
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(-1)
        remove = cum - probs > p                # drop tokens whose preceding mass already exceeds p; keeps at least 1
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)
```

Q23: ***Implement a (simplified) beam search***

- **Always keep the B candidates with the highest cumulative log-probability; at each step expand each with B tokens, keep the top B of the B×B results, and move those that hit EOS to a finished list; finally pick the best by length-penalized score.**
- Accumulate in log space; without a length penalty it favors short outputs.
- Follow-up: it suits translation and summarization; open-ended generation tends to repeat, so sampling is used instead.

```python
def beam_search(step_fn, bos, eos, beam_size=4, max_len=50, alpha=0.6):
    """step_fn(seq: list[int]) -> log-probs of the next token (1D torch tensor)"""
    beams = [([bos], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            logp = step_fn(seq)
            top_lp, top_id = logp.topk(beam_size)
            for lp, tok in zip(top_lp.tolist(), top_id.tolist()):
                candidates.append((seq + [tok], score + lp))
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for seq, score in candidates:
            (finished if seq[-1] == eos else beams).append((seq, score))
            if len(beams) == beam_size:
                break
        if not beams:
            break
    finished.extend(beams)                                 # sequences still unfinished at max_len
    length_penalty = lambda L: ((5 + L) / 6) ** alpha      # GNMT length penalty
    return max(finished, key=lambda c: c[1] / length_penalty(len(c[0])))
```
