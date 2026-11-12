# Gradient Boosted Trees (GBDT / XGBoost / LightGBM / CatBoost)

## Boosting Basics

Q1: ***What is the difference between Boosting and Bagging?***

| | Bagging (e.g. random forest) | Boosting (e.g. AdaBoost, GBDT) |
|---|---|---|
| Training | **Parallel**, base learners are independent | **Sequential**, each learner depends on the previous ones |
| Sample usage | Bootstrap sampling with replacement | All samples, focusing on hard samples via weights/gradients |
| Base learner | Strong learners (deep trees, low bias, high variance) | Weak learners (shallow trees, high bias, low variance) |
| Main effect | **Reduces variance** | **Reduces bias** (also some variance reduction) |
| Combination | Equal-weight averaging / voting | Weighted sum (additive model) |
| Noise | Robust | Fairly sensitive, can fit noise/outliers |

- Intuition: Bagging is "several experts vote independently and you average"; Boosting is "one after another, each corrects the previous errors".

Q2: ***Briefly explain how AdaBoost works.***

- Binary classification $y\in\{-1,+1\}$, initialize sample weights $w_i = 1/N$. For $t=1,\dots,T$:
    1. Train weak classifier $h_t$ with the current weights and compute the weighted error $\epsilon_t = \sum_i w_i \mathbb{I}(h_t(x_i)\ne y_i)$;
    2. Compute classifier weight $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$ (lower error means larger weight; $\alpha_t>0$ when $\epsilon_t<0.5$);
    3. Update sample weights $w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(x_i))$ and normalize, so **misclassified samples get larger weights**.
- Final model $H(x)=\text{sign}\left(\sum_t \alpha_t h_t(x)\right)$.
- Theoretical view: AdaBoost is equivalent to a **forward stagewise additive model** with **exponential loss** $L=\exp(-yF(x))$. Exponential loss penalizes misclassified samples exponentially, so AdaBoost is **sensitive to noise and outliers**. GBDT generalizes this idea to any differentiable loss.

## GBDT

Q3: ***What is the core idea of GBDT? Why is it called "gradient descent in function space"?***

- GBDT is an **additive model**: $F_M(x) = F_0(x) + \sum_{m=1}^{M}\nu\, h_m(x)$, where each $h_m$ is a regression tree trained one at a time by a forward stagewise algorithm.
- At step $m$ we want $h_m$ minimizing $\sum_i L(y_i, F_{m-1}(x_i)+h_m(x_i))$. Treating the values of $F$ at each sample as "parameters", the **negative gradient** of the loss with respect to them is the steepest-descent direction:
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$
- Since the gradient is only available at training points, to generalize to new samples a regression tree is fitted to the **negative gradients** $\{(x_i, r_{im})\}$ (called **pseudo-residuals**), which amounts to projecting the gradient onto the "space of tree functions". Thus GBDT performs **gradient descent in function space**, mirroring parameter-space gradient descent $\theta \leftarrow \theta - \eta\nabla_\theta L$.
- For squared loss $L=\frac12(y-F)^2$, the negative gradient is exactly the residual $y-F$, so "fitting residuals" is merely the special case of squared loss.

Q4: ***Write out the GBDT algorithm.***

1. Initialize $F_0(x)=\arg\min_\gamma \sum_i L(y_i,\gamma)$ (the mean for squared loss, the log-odds for log loss).
2. For $m=1,\dots,M$:
    1. Compute pseudo-residuals $r_{im} = -\partial L(y_i,F_{m-1}(x_i))/\partial F_{m-1}(x_i)$;
    2. Fit a regression tree to $\{(x_i, r_{im})\}$ (split by MSE), giving leaf regions $R_{jm}, j=1..J$;
    3. For each leaf, do a **line search** for the optimal output: $\gamma_{jm} = \arg\min_\gamma \sum_{x_i\in R_{jm}} L(y_i, F_{m-1}(x_i)+\gamma)$;
    4. Update $F_m(x)=F_{m-1}(x)+\nu\sum_j \gamma_{jm}\mathbb{I}(x\in R_{jm})$.
3. Output $F_M(x)$.

- Note: the tree's **structure** is determined by fitting the negative gradient, while the **leaf values** are re-optimized under the original loss (the two coincide for squared loss).

Q5: ***How does GBDT do classification?***

- GBDT's base learners are always **regression trees**; for classification they fit values in log-odds (logit) space.
- **Binary classification**: $p=\sigma(F(x))$ with log loss $L=-[y\log p + (1-y)\log(1-p)]$, $y\in\{0,1\}$.
    - Negative gradient $r_i = y_i - p_i$ (true label minus predicted probability);
    - Leaf values have no closed form and are usually approximated by one **Newton step**: $\gamma_j = \frac{\sum_{i\in R_j} r_i}{\sum_{i\in R_j} p_i(1-p_i)}$;
    - Initialization $F_0 = \log\frac{\bar y}{1-\bar y}$.
- **Multi-class** ($K$ classes): each round trains **$K$ trees**, one per class; outputs go through softmax to get probabilities, and the negative gradient for class $k$ is $y_{ik}-p_{ik}$. Hence the number of trees for multi-class is $K \times$ the number of rounds.

Q6: ***What do shrinkage (learning rate) and subsampling do in GBDT?***

- **Shrinkage**: $F_m = F_{m-1} + \nu h_m$, $0<\nu\le1$ (commonly 0.01–0.1). Each tree takes only "a small step", not fully trusting any single tree and leaving room for later trees to correct, which acts as regularization. Empirically **small learning rate + many trees** generalizes better but trains longer; $\nu$ and the number of trees must be tuned jointly (usually with early stopping).
- **Row subsampling** (Stochastic Gradient Boosting, Friedman 2002): each round trains the tree on only a fraction of samples (e.g. 50%–80%, without replacement). The randomness reduces variance and overfitting and speeds up training; unsampled samples can also be used for an OOB-like estimate.
- **Column subsampling** (introduced by XGBoost, borrowed from random forests): randomly select a subset of features per tree/level/node, further reducing overfitting and computation.

Q7: ***Which loss functions are commonly used with GBDT?***

- Regression:
    - **Squared loss** (L2): negative gradient is the residual, sensitive to outliers;
    - **Absolute loss** (L1): negative gradient is $\text{sign}(y-F)$, leaf value is the median of residuals, more robust;
    - **Huber loss**: L2 for small errors, L1 for large errors, combining both;
    - **Quantile loss**: for quantile regression and prediction intervals;
    - Poisson / Gamma / Tweedie: counts, positive values, insurance claims, etc.
- Classification: log loss (binary / multi-class softmax), exponential loss (equivalent to AdaBoost).
- Ranking: LambdaMART (LambdaRank gradients + MART), `rank:ndcg` and `lambdarank` in XGBoost/LightGBM.
- Any loss that is **first-order differentiable** works for GBDT; XGBoost additionally needs the **second derivative** (custom losses must supply grad and hess).

## XGBoost

Q8: ***What is XGBoost's objective function? How is it derived with a second-order Taylor expansion?***

- Regularized objective (adding tree $f_t$ at round $t$):
$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat y_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t),\quad \Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_{j=1}^{T} w_j^2$$
where $T$ is the number of leaves and $w_j$ the leaf weights (an L1 term $\alpha\sum|w_j|$ can also be added).
- Second-order Taylor expansion at $\hat y^{(t-1)}$, with $g_i = \partial_{\hat y} l(y_i,\hat y^{(t-1)})$ and $h_i = \partial^2_{\hat y} l(y_i,\hat y^{(t-1)})$, dropping constants:
$$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{n}\left[g_i f_t(x_i) + \frac12 h_i f_t^2(x_i)\right] + \gamma T + \frac12\lambda\sum_j w_j^2$$
- The tree maps samples to leaves, $f_t(x)=w_{q(x)}$. Let $I_j=\{i\mid q(x_i)=j\}$, $G_j=\sum_{i\in I_j} g_i$, $H_j = \sum_{i\in I_j} h_i$, and regroup by leaf:
$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T}\left[G_j w_j + \frac12 (H_j+\lambda) w_j^2\right] + \gamma T$$
Each leaf is an independent univariate quadratic.

Q9: ***Derive XGBoost's optimal leaf weight and split gain formula.***

- Setting the derivative with respect to $w_j$ to zero gives the **optimal leaf weight**:
$$w_j^* = -\frac{G_j}{H_j+\lambda}$$
- Substituting back gives the optimal objective for a fixed tree structure $q$ (the **structure score**, lower is better):
$$\tilde{\mathcal{L}}^*(q) = -\frac12\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda} + \gamma T$$
- The reduction in objective when a node is split into left and right children is the **split gain**:
$$\text{Gain} = \frac12\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$
- Exact greedy algorithm: sort each feature by value, scan left to right accumulating $G_L,H_L$ ($G_R=G-G_L$), and a single pass yields the gain of every split point.
- Connection to Newton's method: $w^*=-G/(H+\lambda)$ is an L2-regularized **Newton step**, so XGBoost can be viewed as Newton's method in function space while GBDT is gradient descent in function space.

Q10: ***What roles do $\gamma$, $\lambda$ and `min_child_weight` play in XGBoost?***

- **$\gamma$** (`gamma` / `min_split_loss`): the cost of adding one leaf. A split is made only if $\text{Gain}>0$ (i.e. the bracketed gain exceeds $\gamma$), which is a **built-in pre-pruning**. In practice XGBoost first grows to `max_depth` and then **prunes splits with negative gain bottom-up**, avoiding the "short-sightedness" of pre-pruning.
- **$\lambda$** (`reg_lambda`): L2 regularization on leaf weights, shrinking $|w^*|$ so each tree's predictions are smoother; it also appears in the gain denominators, suppressing leaves with few samples (small $H$). **$\alpha$** (`reg_alpha`) is L1 regularization, which drives small weights to zero.
- **`min_child_weight`**: the minimum $\sum h_i$ in a child. For squared loss $h_i=1$, so it is a minimum sample count; for log loss $h_i=p_i(1-p_i)$, which is small where predictions are already confident, preventing splits made just for a handful of "already learned" samples.

Q11: ***Why does XGBoost use second-order derivatives? What are the benefits?***

- **Accuracy and convergence**: second-order information corresponds to Newton's method, approximating the loss more accurately than first-order-only GBDT; it usually converges faster and needs fewer trees.
- **Unified framework**: the split criterion and leaf weights depend only on $g_i, h_i$, so any **twice-differentiable** loss can be plugged in by providing grad/hess, without deriving a separate line search per loss.
- **Natural regularization**: the quadratic form lets $\lambda$ and $\gamma$ appear in closed form in the leaf weights and gain.
- $h_i$ also acts as a **sample weight**: the objective can be rewritten as $\sum_i \frac12 h_i (f_t(x_i) - (-g_i/h_i))^2 + \Omega$, a weighted squared loss with weights $h_i$ and labels $-g_i/h_i$, which is the theoretical basis of the weighted quantile sketch.
- Requirement: the loss must be twice differentiable with $h_i>0$ (e.g. MAE has zero second derivative and needs special handling).

Q12: ***What engineering optimizations does XGBoost have? (sparsity awareness, weighted quantile sketch, block structure)***

- **Approximate splitting and the Weighted Quantile Sketch**: exact greedy has to scan every split point, which is expensive on large data. The approximate algorithm proposes candidate split points by quantiles (global: once per tree; local: re-proposed after each split). Since the objective is a weighted squared loss with weights $h_i$, candidates are chosen by **$h_i$-weighted quantiles**, and the paper proposes a mergeable weighted quantile sketch data structure with theoretical guarantees.
- **Sparsity-aware Split Finding**: for sparse data from missing values, zeros or one-hot encoding, each node learns a **default direction**: compute the gain with all missing samples sent left and with all sent right, and keep the larger. Only non-missing values are scanned, so the cost is linear in the number of non-missing entries, greatly speeding up sparse data.
- **Column Block structure**: before training, data is **pre-sorted by column in CSC format** and stored in blocks, reused by every subsequent split to avoid repeated sorting; gain computation across features can run **in parallel** (XGBoost parallelism is at the **feature level**; trees are still sequential).
- **Cache-aware access**: fetching gradients by sorted index causes non-contiguous memory access and poor cache hit rates, so XGBoost allocates a buffer per thread and **prefetches** gradients into contiguous memory.
- **Out-of-core computation**: when data does not fit in memory, blocks are stored on disk, with **block compression** and **sharding** across multiple disks, and an independent thread pre-reads them.
- Note: XGBoost later also implemented the histogram algorithm (`tree_method="hist"`, the default since 2.0) and GPU training (`device="cuda"`); exact greedy is rarely used in practice now.

Q13: ***What improvements does XGBoost make over traditional GBDT?***

1. **Second-order Taylor expansion** (Newton's method in function space) vs first-order gradients only in GBDT;
2. **Explicit regularization in the objective** (number of leaves $\gamma T$ and L2/L1 on leaf weights) to control model complexity;
3. **Base learners**: besides CART, supports linear models (`gblinear`) and DART (trees with dropout);
4. **Column subsampling**, borrowed from random forests, reducing overfitting and computation;
5. **Missing-value handling**: sparsity awareness, automatically learning default directions;
6. **Shrinkage** applied to each tree (GBDT has it too, but XGBoost makes it standard);
7. **Split finding**: supports exact, approximate (weighted quantile sketch) and histogram algorithms;
8. **Engineering optimizations**: feature-level parallelism, pre-sorted column blocks, cache awareness, out-of-core computation, distributed training, GPU;
9. **Pruning**: grow to max depth first, then post-prune using $\gamma$;
10. Built-in cross-validation, early stopping, and support for custom losses and evaluation metrics.

## LightGBM

Q14: ***What is LightGBM's histogram algorithm? What are its advantages?***

- Before training, each continuous feature is discretized into at most `max_bin` (default 255) **bins**, and each sample stores only the bin index (e.g. uint8). When splitting, the data is traversed accumulating $\sum g$, $\sum h$ and sample counts per bin to form a histogram, and the best split point is searched over **bin boundaries**.
- Advantages:
    - **Low memory**: no need to store pre-sorted values and sorted indices; feature values are stored as 8-bit integers, reducing memory to roughly 1/8 of pre-sorting;
    - **Fast computation**: split candidates drop from $O(\#\text{data})$ to $O(\#\text{bin})$; building histograms costs $O(\#\text{data}\times\#\text{feature})$ and finding splits $O(\#\text{bin}\times\#\text{feature})$;
    - **Histogram subtraction**: parent histogram = left child histogram + right child histogram, so only the histogram of the child with **fewer samples** needs to be built and the other is obtained by subtraction, nearly doubling speed;
    - **Regularization effect**: coarse split points are themselves a form of regularization and are insensitive to noise;
    - Low communication cost, good for distributed parallelism.
- Drawback: split points are not exact and some accuracy may be lost (in practice the impact is usually small and often negligible within the GBDT framework).

Q15: ***What is the difference between level-wise and leaf-wise tree growth?***

- **Level-wise (depth-wise)**, XGBoost's default (`grow_policy="depthwise"`): split all leaves on the same level each time.
    - Pros: easy to parallelize, trees are balanced, less prone to overfitting;
    - Cons: many low-gain leaves get split too, wasting computation.
- **Leaf-wise (best-first)**, LightGBM's default: each time, split the one leaf with the **largest gain** among all current leaves.
    - Pros: for the same number of leaves, it reduces loss faster and gives **higher accuracy**;
    - Cons: can grow very deep, unbalanced trees and **overfits easily on small data**, so it must be constrained with `num_leaves` (the main complexity parameter), `max_depth` and `min_data_in_leaf`.
- Tuning note: `num_leaves` should be smaller than $2^{\text{max\_depth}}$, otherwise the `max_depth` constraint is meaningless. XGBoost can also grow leaf-wise via `grow_policy="lossguide"`.

Q16: ***What is GOSS (Gradient-based One-Side Sampling)?***

- Motivation: samples with small absolute gradients are already well fitted and contribute little to information gain; but simply dropping them would change the data distribution.
- Procedure:
    1. Sort by absolute gradient in descending order and **keep the top $a\times100\%$ large-gradient samples**;
    2. **Randomly sample $b\times100\%$** from the remaining samples;
    3. When computing gain, multiply the sampled small-gradient samples by weight $\frac{1-a}{b}$ to compensate for the distribution shift, giving an approximately unbiased gain estimate.
- Effect: gain can be estimated fairly accurately using only an $(a+b)$ fraction of the data, significantly speeding up training; the paper gives an error bound.
- In recent LightGBM versions it is enabled via `data_sample_strategy="goss"`, with parameters `top_rate` ($a$) and `other_rate` ($b$).

Q17: ***What is EFB (Exclusive Feature Bundling)?***

- Motivation: in high-dimensional sparse data (e.g. one-hot), many features **almost never take non-zero values simultaneously** (mutually exclusive), so they can be merged into one feature without losing information, reducing the number of features.
- Procedure:
    1. **Decide which features to bundle**: build a graph whose nodes are features and whose edge weights are the number of conflicts (simultaneous non-zeros). Bundling is modeled as **graph coloring** (NP-hard) and approximated greedily: sort features by degree and add each to an existing bundle whose conflicts stay below threshold $K$, otherwise create a new bundle; a small number of conflicts is tolerated in exchange for more bundling.
    2. **Merge features**: add **offsets** to the values of different features in a bundle so they fall into non-overlapping bin ranges. E.g. if feature A takes values in [0,10) and B in [0,20), add offset 10 to B to get [10,30); the merged feature takes values in [0,30).
- Effect: histogram construction drops from $O(\#\text{data}\times\#\text{feature})$ to $O(\#\text{data}\times\#\text{bundle})$, with $\#\text{bundle}\ll\#\text{feature}$.

Q18: ***How does LightGBM handle categorical features?***

- Native support: specify via `categorical_feature`; features must be encoded as non-negative integers, no one-hot needed.
- **Many-vs-many** splits: at each node, categories are **sorted** by $\frac{\sum g}{\sum h + \text{cat\_smooth}}$, and the best split point is searched on the sorted histogram as for a continuous feature, i.e. the category set is partitioned into two subsets. This applies Fisher's (1958) optimal partitioning idea, with complexity $O(k\log k)$ instead of $O(2^k)$.
- When the number of categories is small ($\le$ `max_cat_to_onehot`, default 4), it falls back to one-vs-rest.
- Anti-overfitting parameters: `min_data_per_group`, `cat_smooth`, `cat_l2`, `max_cat_threshold`. High-cardinality categoricals still overfit easily; consider target encoding or CatBoost in that case.
- Why not one-hot: with one-hot each split can only peel off one category, gains are small, and trees become unbalanced and deep; native splits can group similar categories in one go.

Q19: ***How do LightGBM and XGBoost differ?***

| | XGBoost | LightGBM |
|---|---|---|
| Split finding | Classically pre-sorted exact / weighted-quantile approximate (now hist by default) | Histogram algorithm + histogram subtraction |
| Growth strategy | Level-wise by default | Leaf-wise by default |
| Sampling | Random row/column sampling | Additionally GOSS gradient-based one-side sampling |
| Sparse features | Sparsity-aware splits | EFB exclusive feature bundling |
| Categorical features | Manual encoding earlier; newer versions have experimental/native support (`enable_categorical`) | Native many-vs-many splits |
| Speed/memory | Slower, more memory (classic versions) | Faster, less memory |
| Parallelism | Feature parallel | Feature parallel, data parallel, **voting parallel** (lower communication) |
| Overfitting | Balanced trees, relatively robust | Leaf-wise overfits easily on small data |

- Practical conclusion: with thorough tuning their accuracy is usually similar; LightGBM is faster on large, high-dimensional data; since XGBoost 2.x made hist the default and added GPU and categorical support, the speed gap has narrowed considerably.

## CatBoost

Q20: ***What are CatBoost's Ordered Target Statistics? What problem do they solve?***

- Categorical features are often handled with **target encoding** (replace a category with the mean label of that category). But computing it on all training data (including the sample itself) causes **target leakage**: the encoding of sample $x_i$ contains information about $y_i$, so during training the encoding is strongly correlated with the label while the test distribution differs, causing **conditional shift** and overfitting (in the extreme, if each category appears once, the encoding equals the label itself).
- CatBoost's approach: take a random permutation $\sigma$ of the training samples; the encoding of sample $x_i$ **uses only samples that precede it**:
$$\hat x_i^k = \frac{\sum_{j:\sigma(j)<\sigma(i)} \mathbb{I}(x_j^k = x_i^k)\, y_j + a\,P}{\sum_{j:\sigma(j)<\sigma(i)} \mathbb{I}(x_j^k = x_i^k) + a}$$
where $P$ is a prior (e.g. the global positive rate) and $a>0$ is a smoothing weight. Like "online learning", each sample is encoded using only "history", avoiding leakage.
- Multiple random permutations are used to reduce the variance of a single permutation. At test time, statistics are computed on all training data.
- CatBoost also **greedily constructs combinations of categorical features** (feature crosses) and applies ordered TS to them as well.

Q21: ***What is Ordered Boosting? What is prediction shift?***

- **Prediction shift**: in standard GBDT, the model $F_{t-1}$ used to compute the gradient of sample $x_i$ at round $t$ was **trained on data including $x_i$**, so the residual distribution on training samples differs from that on test samples (training residuals are too small); the gradient estimate is biased, causing overfitting. It is essentially the same kind of problem as target leakage.
- **Ordered Boosting**: also based on random permutations; in theory, for each sample $i$ it maintains a model $M_{i-1}$ **trained only on the first $i-1$ samples of the permutation**, and uses $M_{i-1}$ to compute the residual of sample $i$, ensuring the residual is "unbiased".
- In practice maintaining $n$ models is infeasible, so CatBoost maintains approximate models only for exponentially growing prefix lengths ($2^j$), sharing the tree structure and differing only in leaf values, which keeps the overhead acceptable.
- The effect is most noticeable on small datasets; on large data `boosting_type="Plain"` can be used for speed.

Q22: ***Why does CatBoost use symmetric (oblivious) trees?***

- **Symmetric tree**: all nodes on the same level use **the same feature and split threshold**; a tree of depth $d$ has $2^d$ leaves, each indexed by the $d$ binary decisions.
- Advantages:
    - **Extremely fast inference**: prediction only needs $d$ comparisons to form a $d$-bit binary index and a direct table lookup, easy to vectorize/SIMD, so CPU inference is significantly faster than with ordinary trees;
    - **Regularization effect**: the strong structural constraint resists overfitting and makes the model less sensitive to hyperparameters, so default parameters already perform well;
    - Training only needs to pick one best (feature, threshold) per level, and it combines more easily with ordered boosting.
- Drawbacks: limited expressiveness, needing more trees at the same depth; CatBoost also supports `grow_policy="Depthwise"` / `"Lossguide"`.

## Tuning and Practice

Q23: ***What are the key hyperparameters of GBDT-style models? What is the tuning order?***

- Key hyperparameters (using XGBoost/LightGBM as examples):
    - Iterations and step size: `n_estimators`, `learning_rate`;
    - Tree complexity: `max_depth`, `num_leaves` (LightGBM), `min_child_weight` / `min_data_in_leaf`;
    - Randomness: `subsample` / `bagging_fraction`, `colsample_bytree` / `feature_fraction`;
    - Regularization: `reg_lambda`, `reg_alpha`, `gamma` / `min_split_gain`;
    - LightGBM-specific: `max_bin`, categorical-related parameters.
- Common tuning order:
    1. Fix a relatively large learning rate (e.g. 0.1) and use **early stopping** to determine the number of trees;
    2. Tune tree structure: `max_depth` / `num_leaves`, `min_child_weight` / `min_data_in_leaf` (largest impact on performance);
    3. Tune sampling: `subsample`, `colsample_bytree` (often 0.6–0.9);
    4. Tune regularization: `lambda`, `alpha`, `gamma`;
    5. **Lower the learning rate** (e.g. 0.01–0.05), increase the number of trees, and early-stop again to get the final model.
- In practice, Bayesian optimization tools such as Optuna are often used to search these parameters jointly, which is more efficient than grid search.

Q24: ***How is early stopping used in GBDT? What should you watch out for?***

- During training, monitor a metric on the **validation set**; if it does not improve for `early_stopping_rounds` consecutive rounds (e.g. 50–200), stop and use the model at the best round (`best_iteration`).

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(n_estimators=10000, learning_rate=0.05, num_leaves=63)
model.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          eval_metric="auc",
          callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(200)])
print(model.best_iteration_)
```

- Caveats:
    - The validation set used for early stopping is itself "fitted", so final evaluation must use an **independent test set** or nested cross-validation;
    - With cross-validation each fold stops at a different round; when retraining on the full data you can use the average best round (or scale it up somewhat with the data size);
    - The monitored metric should match the business objective (e.g. AUC, logloss, RMSE), and the validation split should mirror production (time series must be split by time).

Q25: ***How does GBDT handle class imbalance?***

- **Sample/class weighting**: `scale_pos_weight` in XGBoost/LightGBM (often set to #negatives/#positives), LightGBM's `is_unbalance=True`, or custom weights via `sample_weight`. Weighting amplifies the gradients and Hessians of the minority class, affecting splits and leaf values.
- **Sampling**: downsample the majority class, upsample the minority class (SMOTE usually brings limited gains for tree models).
- **Custom loss**: e.g. Focal Loss, which requires supplying first and second derivatives.
- **Evaluation and threshold**: use PR-AUC, F1, Recall@K, etc. instead of accuracy; do not use the default 0.5 threshold, but **tune the classification threshold** on the validation set.
- Note: weighting or sampling **distorts predicted probabilities**; when accurate probabilities are needed (e.g. CTR prediction), calibrate (Platt / Isotonic) or correct for downsampling with $p = \frac{p'}{p' + (1-p')/w}$ ($w$ is the negative sampling rate).

Q26: ***What kinds of feature importance does GBDT offer? What are their issues? What are SHAP's advantages?***

- XGBoost's `importance_type`:
    - **weight / split**: the **number of times** a feature is used to split, biased toward high-cardinality and continuous features;
    - **gain**: the **average gain** of splits using the feature (`total_gain` is the total), which better reflects actual contribution but is still based on the training set;
    - **cover**: the average number of samples (sum of Hessians) covered by splits on the feature.
- LightGBM provides `split` and `gain`. Rankings under different metrics can differ a lot, and these global metrics are **inconsistent**: when a model relies more on a feature, its importance does not necessarily increase.
- **SHAP** (TreeSHAP, Lundberg et al.): based on Shapley values, satisfying **local accuracy, consistency and missingness**; TreeSHAP computes them exactly for tree models in polynomial time; it gives **per-sample** contributions and directions for every feature, the mean of $|\phi|$ gives global importance, and interactions can be analyzed (SHAP interaction values).
- Note: correlated features still share SHAP values; to assess contribution to generalization, combine with permutation importance on the validation set.

Q27: ***What is GBDT + LR? Why does it work?***

- Origin: Facebook's 2014 ad CTR prediction paper (*Practical Lessons from Predicting Clicks on Ads at Facebook*).
- Method: first train a GBDT, take the **leaf index** each sample falls into in each tree as a categorical feature, one-hot encode them into a high-dimensional sparse vector (optionally alongside raw features), and feed it to **logistic regression**.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=3).fit(X_gbdt, y_gbdt)
leaves = gbdt.apply(X_lr)[:, :, 0]            # (n_samples, n_trees) leaf indices
enc = OneHotEncoder(handle_unknown="ignore").fit(leaves)
lr = LogisticRegression(max_iter=1000).fit(enc.transform(leaves), y_lr)
```

- Why it works: each root-to-leaf path is an **automatically learned feature combination** (multi-feature crossing + nonlinear discretization), solving LR's need for heavy manual feature engineering; the LR part can be updated frequently and suits large-scale sparse online learning.
- Caveats: GBDT and LR should preferably be trained on **different data**, to avoid LR overfitting on leaf features where the GBDT overfit; high-cardinality ID features are not suitable for direct use in the GBDT. This approach was later superseded by deep CTR models such as FM/FFM, Wide & Deep and DeepFM, but the idea (using tree models as feature transformers) is still common.

Q28: ***Why is GBDT usually preferred over deep learning for tabular data? How do you choose between random forest and GBDT?***

- **GBDT's advantages on tabular data** (see Grinsztajn et al. 2022, *Why do tree-based models still outperform deep learning on tabular data?*):
    - Tabular features are **heterogeneous** and include many uninformative features; trees perform feature selection naturally and are robust to irrelevant features; MLPs are rotation-invariant and struggle to exploit the "axis-aligned" meaning of features;
    - Target functions are often **non-smooth with abrupt changes**, which the piecewise-constant structure of trees fits more easily, while neural networks are biased toward smooth functions;
    - With small to medium data (thousands to hundreds of thousands of rows), trees are less prone to overfitting;
    - Insensitive to missing values, outliers and scale, so little preprocessing; fast training, low tuning cost, mature interpretability tools (SHAP), simple deployment.
- **When to consider deep learning**: very large data, unstructured features such as text/images/sequences, need for end-to-end representation learning (embeddings), multi-task/multi-modal setups, online incremental learning; or tabular foundation models (e.g. the TabPFN family is very strong on small datasets). In practice, **ensembles** of GBDT and NNs often work best.
- **Random forest vs GBDT**:

| | Random forest | GBDT |
|---|---|---|
| Ensemble method | Bagging, parallel | Boosting, sequential |
| Reduces | Variance | Bias |
| Base trees | Deep trees, unpruned | Shallow trees (depth 3–8) |
| Tuning | Little, default parameters are stable | More, sensitive to learning rate/tree depth |
| Overfitting | More trees means more stable, hard to overfit | Too many trees overfit, early stopping needed |
| Noise/outliers | Robust | Fairly sensitive |
| Accuracy | Good baseline | Usually higher once tuned |
| Other | OOB estimate, easy to parallelize | Supports any differentiable loss, ranking, etc. |

- Recommendation: use random forest for a fast, robust baseline, noisy data, or limited tuning resources; use GBDT (XGBoost / LightGBM / CatBoost, preferring CatBoost when there are many categorical features) when chasing maximum accuracy with time to tune.
