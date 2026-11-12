# Coding from Scratch

> Conventions: `import numpy as np`, `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `import math`. Later questions reuse functions defined earlier (e.g. `softmax`, `sigmoid`, `box_iou`).

## Basic Functions and Losses

Q1: ***Implement numerically stable softmax and log-softmax***

- Softmax is **shift-invariant**: $\mathrm{softmax}(x)=\mathrm{softmax}(x-c)$. Choosing $c=\max_i x_i$ makes the largest exponent $e^0=1$ and prevents `exp` overflow (`np.exp(1000)` is `inf`).
- Log-softmax uses the **log-sum-exp trick**: $\log \mathrm{softmax}(x)_i = x_i - m - \log\sum_j e^{x_j-m}$. Do not write `np.log(softmax(x))` — taking the log of a probability that underflowed to 0 gives `-inf`.

```python
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
```

- Complexity $O(n)$. Pitfalls: `keepdims=True` is needed for correct broadcasting; for batched input reduce along the class axis (`axis=-1`), not the max over the whole array.

Q2: ***Implement cross-entropy loss from logits, and its gradient***

- $L=-\frac{1}{N}\sum_n \log \mathrm{softmax}(z_n)_{y_n}$; computing directly from logits (fused log-softmax) is more stable than softmax followed by log.
- Classic gradient result: $\frac{\partial L}{\partial z} = \frac{1}{N}(\mathrm{softmax}(z) - \mathrm{onehot}(y))$.

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

- Pitfall: use fancy indexing `logp[np.arange(N), y]` to pick the correct class; `logp[:, y]` gives an $N\times N$ matrix. Follow-up: label smoothing just replaces one-hot with $(1-\epsilon)\,\mathrm{onehot}+\epsilon/C$.

Q3: ***Implement numerically stable sigmoid and BCE (with logits)***

- The naive `1/(1+np.exp(-x))` overflows `exp(-x)` for $x\ll 0$ (the result is 0, but it raises warnings or even `nan`). Split by sign: $\frac{1}{1+e^{-x}}$ for $x\ge0$, $\frac{e^{x}}{1+e^{x}}$ for $x<0$.
- BCE simplifies to: $\ell(z,y) = -[y\log\sigma(z)+(1-y)\log(1-\sigma(z))] = \max(z,0) - zy + \log(1+e^{-|z|})$, which never overflows for any $z$.

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

- Pitfall: use `np.log1p` rather than `np.log(1+...)` for accuracy with small values. This is exactly why PyTorch recommends `BCEWithLogitsLoss` over `Sigmoid + BCELoss`. The gradient is equally simple: $\partial\ell/\partial z=\sigma(z)-y$.

## Classical Machine Learning

Q4: ***Implement linear regression (closed form + gradient descent)***

- Closed form (normal equation): $w^*=(X^\top X+\lambda I)^{-1}X^\top y$; $\lambda>0$ gives ridge regression. Use `np.linalg.solve` and **never invert explicitly** (slower and numerically worse); the bias is usually not regularized.
- Gradient descent: MSE $L=\frac{1}{2n}\|Xw+b-y\|^2$, $\nabla_w=\frac1n X^\top(Xw+b-y)$.

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

- Complexity: closed form $O(nd^2+d^3)$, good for small $d$; GD $O(nd)$ per epoch, good at scale. Pitfalls: standardize features, otherwise GD converges slowly and the learning rate is hard to tune; if $X^\top X$ is ill-conditioned, use ridge or `np.linalg.lstsq`.

Q5: ***Implement logistic regression with gradient descent***

- Model $p=\sigma(Xw+b)$ with BCE loss; the gradient has the same form as linear regression: $\nabla_w=\frac1n X^\top(p-y)$.

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

- Pitfalls: on linearly separable data without regularization $\|w\|\to\infty$ (no convergence) — add L2 or early stopping; use the stable sigmoid from Q3. Follow-up: why not MSE? — combined with sigmoid it is non-convex and gradients vanish in the saturated region.

Q6: ***Implement K-means***

- Alternate two steps: **assignment** (each point to its nearest center) and **update** (center = cluster mean). The objective $\sum_i\|x_i-\mu_{c_i}\|^2$ is monotonically non-increasing, so it converges to a local optimum.
- Vectorize distances with the expansion $\|x-c\|^2=\|x\|^2-2x^\top c+\|c\|^2$ to avoid a large $(n,k,d)$ intermediate tensor.

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

- Complexity $O(nkd)$ per iteration. Pitfalls: handle **empty clusters**; sensitive to initialization (in practice use k-means++ with several restarts and keep the best); standardize features.

Q7: ***Implement KNN classification***

- Lazy learning, no training phase; at prediction time find the $k$ nearest neighbors and take a majority vote.
- Use `np.argpartition` to get the $k$ smallest in $O(n)$ instead of a full $O(n\log n)$ sort.

```python
def knn_predict(X_train, y_train, X_test, k=5):
    d2 = ((X_test**2).sum(1, keepdims=True) - 2 * X_test @ X_train.T
          + (X_train**2).sum(1))                                  # (m, n)
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
    neigh = y_train[idx]                                          # (m, k)
    return np.array([np.bincount(row).argmax() for row in neigh])
```

- Complexity: prediction $O(mnd)$, memory $O(mn)$ — for large data compute in chunks or use a KD-tree / approximate nearest neighbors (FAISS, HNSW). Pitfalls: the expansion can produce tiny negative values from floating-point error (ordering is unaffected; clamp with `np.maximum(d2, 0)` before taking a square root); $y$ must be non-negative integers for `bincount`; use an odd $k$ to avoid ties in binary classification.

Q8: ***Implement PCA via SVD***

- After centering, $X_c=U\Sigma V^\top$; principal directions are the rows of $V^\top$ (the eigenvectors of the covariance $\frac{1}{n-1}X_c^\top X_c$), and explained variance is $\sigma_i^2/(n-1)$.
- Prefer SVD over forming the covariance and eigendecomposing it: forming $X^\top X$ squares the condition number and loses precision.

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

- Complexity $O(nd\min(n,d))$; use randomized SVD at scale. Pitfalls: forgetting to center; new data must be projected after subtracting the **training** mean; SVD signs are not unique (sklearn applies a sign flip).

## Transformer Components

Q9: ***Implement scaled dot-product attention with a mask***

- $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$. Dividing by $\sqrt{d_k}$ keeps the variance of the dot products near 1, preventing softmax saturation and vanishing gradients.
- Mask convention: boolean tensor, `True` means **may attend** (consistent with `F.scaled_dot_product_attention`); masked positions are filled with `-inf`.

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

- Complexity: time $O(T^2 d)$, memory $O(T^2)$ — exactly what FlashAttention's tiling + online softmax addresses. Pitfalls: in fp16 use `-inf` or `torch.finfo(dtype).min`, not `-1e9` (fp16 max is about 65504, so `-1e9` overflows to `-inf`); align the mask semantics (True = keep vs. mask out) with the caller — the boolean `attn_mask` of `nn.MultiheadAttention` is the opposite (True = masked out).

Q10: ***How do you build a causal mask?***

- Position $i$ may only see $j\le i$, i.e. the lower triangle is True.

```python
def causal_mask(T, device=None):
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))   # (T, T)

def causal_padding_mask(pad_mask):
    """pad_mask: (B, T), True for real tokens -> (B, 1, T, T), broadcastable over heads"""
    T = pad_mask.size(1)
    return causal_mask(T, pad_mask.device) & pad_mask[:, None, None, :]
```

- Combine with a padding mask using logical AND and expand to `(B, 1, T, T)` so it broadcasts over all heads. With a KV cache at inference, the new query has length 1 and keys have length $T_{past}+1$, so no causal mask is needed (or offset `tril` with `diagonal=T_k - T_q`). Pitfall: with left padding in batched generation, position ids must be shifted accordingly.

Q11: ***Implement a multi-head attention module***

- Split $d_{model}$ into $h$ heads with $d_k=d_{model}/h$ each, run attention per head, concatenate, and project linearly; total compute is similar to a single head, but heads can attend to different patterns in different subspaces.

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

- Pitfalls: after `transpose` the tensor is non-contiguous, so call `.contiguous()` before `view` (or use `reshape`); you must reshape to `(B,T,h,d_k)` and then transpose — a direct `view(B,h,T,d_k)` mixes data from different tokens. Parameter count $4d^2+4d$. Follow-up: GQA/MQA just uses $h_{kv}<h$ heads for K/V and `repeat_interleave`s them to $h$.

Q12: ***Implement applying RoPE (rotary position embedding)***

- Rotate each pair of dimensions $(x_{2i},x_{2i+1})$ (or $(x_i, x_{i+d/2})$ in the half-split layout) by angle $m\theta_i$ at position $m$, with $\theta_i=b^{-2i/d}$, $b=10000$.
- Key property: $\langle R_m q, R_n k\rangle$ depends only on $m-n$, so relative position is encoded into the dot product; the rotation preserves vector norms.

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

- Pitfalls: the **interleaved** layout (original paper / Meta's reference code, adjacent dimensions paired) and the **half-split** layout are incompatible — they must match when loading weights; with a KV cache, slice `cos/sin` at absolute positions `[pos: pos+T]`; for long-context extrapolation, adjust `base` or the frequencies with NTK-aware scaling / YaRN.

## Normalization and Regularization

Q13: ***Implement LayerNorm***

- Normalize each sample over the **feature dimension**: $y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$. Independent of batch size; identical behavior in training and inference.

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

- Pitfalls: use the **biased** variance (`unbiased=False`) to match the official implementation; $\epsilon$ goes inside the square root.

Q14: ***Implement BatchNorm (train/eval modes with running stats)***

- Training: normalize with the current batch mean/variance and update running statistics with an EMA; inference: use the running statistics, so the output does not depend on the batch.
- PyTorch convention: normalization uses the biased variance, while running_var is updated with the **unbiased** variance; `momentum=0.1` means $r\leftarrow(1-0.1)r+0.1\,\hat{x}$.

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

- Pitfalls: forgetting `model.eval()` makes inference use batch statistics (completely wrong with batch size 1); running-stat updates must be under `no_grad`; BatchNorm2d on `(N,C,H,W)` reduces over dims `(0,2,3)`. BN is unstable with small batches → GroupNorm/LayerNorm; under DDP use SyncBatchNorm.

Q15: ***Implement RMSNorm***

- Drop LayerNorm's mean subtraction and $\beta$: $y=\gamma\cdot\frac{x}{\sqrt{\frac1d\sum x_i^2+\epsilon}}$. Cheaper with comparable quality; the default in mainstream LLMs such as LLaMA.

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

- Pitfall: in bf16/fp16 the sum of squares loses precision or even overflows, so upcast to fp32 and cast back (as the LLaMA reference implementation does).

Q16: ***Implement dropout (inverted dropout)***

- During training zero each element with probability $p$ and **divide the kept values by $1-p$** so the expectation is preserved, $\mathbb{E}[y]=x$; at inference it is the identity with no rescaling.

```python
def dropout(x, p=0.5, training=True, rng=np.random.default_rng()):
    if not training or p == 0:
        return x
    mask = rng.random(x.shape) >= p          # keep with probability 1-p
    return x * mask / (1 - p)
```

- Pitfalls: $p$ is the **drop** probability (PyTorch convention), not the keep probability; in backprop the gradient is multiplied by the same mask and divided by $1-p$, so the forward pass must save the mask. Follow-up: dropout combined with BN causes a variance shift; modern LLM pre-training often uses $p=0$.

## Detection and Evaluation

Q17: ***Implement IoU (batched)***

- $\mathrm{IoU}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}$, boxes in $(x_1,y_1,x_2,y_2)$ format.

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

- Pitfalls: width/height must be clipped to 0 for disjoint boxes, otherwise two negatives multiply into a positive "intersection"; mind conversion between $(x,y,w,h)$ and $(x_1,y_1,x_2,y_2)$; an old integer-pixel convention adds `+1`. Follow-up: GIoU/DIoU/CIoU still provide gradients when boxes do not overlap.

Q18: ***Implement NMS (non-maximum suppression)***

- Sort by score descending; repeatedly keep the highest-scoring box and remove remaining boxes whose IoU with it exceeds the threshold, until none remain.

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

- Worst case $O(n^2)$. Pitfalls: for multi-class detection run NMS **per class** (common trick: offset each class's boxes by a large coordinate shift and run once); usually filter by a score threshold before NMS. Follow-up: Soft-NMS decays scores instead of deleting boxes, which helps with dense occlusion.

Q19: ***Implement AUC computation***

- AUC = the probability that a randomly chosen positive scores higher than a randomly chosen negative (ties count 0.5). Equivalent to the Mann–Whitney U statistic: $\mathrm{AUC}=\frac{\sum_{i\in pos}\mathrm{rank}_i-\frac{n_+(n_++1)}{2}}{n_+n_-}$, with 1-based ranks and average ranks for ties.

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

- Complexity $O(n\log n)$; brute-force pairwise comparison is $O(n_+n_-)$. Pitfalls: **tie handling**; AUC is undefined with only one class (zero denominator). Follow-up: AUC depends only on ranking and is insensitive to class ratio; under extreme imbalance PR-AUC is more informative.

## Training and Optimization

Q20: ***Write a two-layer MLP training loop in NumPy with manual backprop***

- Forward: $h=\mathrm{ReLU}(XW_1+b_1)$, $z=hW_2+b_2$, cross-entropy loss; backward applies the chain rule layer by layer, reusing `cross_entropy` and `cross_entropy_grad` from Q2.

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

- Checks: every gradient must have the same shape as its parameter; bias gradients sum over the batch dimension; verify with a **numerical gradient check** $\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}$. Pitfalls: all-zero initialization fails to break symmetry; `p -= ...` updates in place, whereas `p = p - ...` inside the loop would not modify the original arrays. Real training should use shuffled mini-batches.

Q21: ***Implement one Adam / AdamW update step***

- $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$; bias correction $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$; $\theta\leftarrow\theta-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$.
- **AdamW**'s decoupled weight decay: $\theta\leftarrow\theta-\eta\lambda\theta$ is applied directly to the parameters, instead of adding $\lambda\theta$ to the gradient (which would be rescaled by the adaptive denominator, i.e. Adam+L2).

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

- Pitfalls: $t$ starts at 1, otherwise bias correction divides by 0; the states $m,v$ must be updated in place (`m[:] = ...`); optimizer state takes 2x the parameter memory (fp32), which motivates ZeRO sharding. Follow-up: bias correction compensates for the early estimates being biased toward zero because $m_0=v_0=0$.

## Decoding

Q22: ***Implement top-k / top-p (nucleus) sampling***

- First divide by temperature $T$ ($T\to0$ approaches greedy, $T>1$ is more random); top-k keeps only the $k$ most likely tokens; top-p keeps the smallest set, in descending probability order, whose cumulative probability **just exceeds** $p$; set the other logits to `-inf`, re-apply softmax and sample.

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

- Pitfalls: `cum - probs > p` (cumulative mass excluding the token itself) ensures the token that crosses the threshold is kept and at least one token survives; after sorting you must `scatter` back to vocabulary order; apply top-k before top-p (HF's default order). Complexity $O(V\log V)$ (sort). Follow-up: min-p sampling sets the threshold as a fraction of $p_{max}$.

Q23: ***Implement a (simplified) beam search***

- Keep the $B$ candidate sequences with the highest score (cumulative log-probability); at each step expand each candidate with its top-$B$ tokens and select the global top-$B$ out of $B\times B$ candidates; sequences that hit EOS move to a finished list. Finally pick the best after **length-penalty** normalization, to avoid favoring short outputs.

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

- Complexity: $B$ forward passes per step + $O(B^2\log B)$ sorting. Pitfalls: accumulate in **log space** (multiplying probabilities underflows); without a length penalty it favors short sequences; real implementations batch the $B$ candidates into one forward pass and reuse/reorder the KV cache. Follow-up: beam search suits deterministic tasks like translation/summarization; open-ended generation tends to become repetitive and bland, so sampling is used instead.
