# Decision Tree

## Basics and Split Criteria

Q1: ***What is a decision tree and how does it make predictions?***

- A decision tree is a **non-parametric** supervised learning model for classification and regression. It **recursively partitions** the feature space into disjoint rectangular regions (axis-aligned splits), each region corresponding to a **leaf node**.
- Internal nodes are tests on a feature (e.g. $x_j \le t$ or $x_j \in \{a,b\}$), branches are test outcomes, and leaves store predictions:
    - Classification: the **majority class** of the samples in the leaf (or the class probability distribution);
    - Regression: the **mean** of the labels in the leaf.
- To predict, start at the root, follow the feature tests down to a leaf, and output that leaf's value. Essentially it is a set of **if-then rules**, and can also be seen as a piecewise-constant approximation of the conditional distribution $P(Y|X)$.

Q2: ***How is a decision tree built? When does splitting stop?***

- It uses **top-down, greedy** recursive partitioning (Hunt's algorithm):
    1. At the current node, iterate over all features and candidate split points and pick the **best split** according to the criterion (information gain, gain ratio, Gini index, MSE, etc.);
    2. Send the samples to child nodes according to that split;
    3. Recurse on each child until a stopping condition is met.
- Common stopping conditions: all samples in the node belong to one class; no features left or all samples have identical feature values; sample count below `min_samples_split`; `max_depth` reached; impurity decrease from the split below a threshold.
- Finding the globally optimal decision tree is **NP-complete**, so in practice a greedy, locally optimal strategy is used, which also means decision trees are sensitive to data perturbations.

Q3: ***How are entropy, conditional entropy and information gain defined?***

- Let dataset $D$ have $K$ classes with proportion $p_k$ for class $k$. **Entropy**:
$$H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k$$
Higher entropy means more uncertainty (impurity); $H=0$ when all samples are one class, and the maximum $\log_2 K$ is reached for a uniform distribution.
- Splitting $D$ by feature $A$ (values $a_1,\dots,a_V$) into $D_1,\dots,D_V$, the **conditional entropy** is:
$$H(D|A) = \sum_{v=1}^{V} \frac{|D_v|}{|D|} H(D_v)$$
- **Information gain** (i.e. the mutual information between $D$ and $A$):
$$\text{Gain}(D,A) = H(D) - H(D|A)$$
ID3 picks the feature with the largest information gain.

Q4: ***What is wrong with information gain? How does gain ratio fix it?***

- Information gain **favors features with many distinct values**. In the extreme, splitting on a "sample ID" puts one sample in each subset, giving zero conditional entropy and maximal information gain, but zero generalization.
- C4.5 uses the **gain ratio**, normalizing by the feature's own "intrinsic value" (split information):
$$\text{GainRatio}(D,A) = \frac{\text{Gain}(D,A)}{IV(A)},\quad IV(A) = -\sum_{v=1}^{V}\frac{|D_v|}{|D|}\log_2\frac{|D_v|}{|D|}$$
The more values a feature has, the larger $IV(A)$, which penalizes many-valued features.
- But gain ratio in turn **favors features with few values**, so C4.5 uses a heuristic: first keep candidate features whose **information gain is above average**, then among them pick the one with the **highest gain ratio**.

Q5: ***What is the Gini index and how does it relate to entropy?***

- The **Gini index** is the probability that two samples drawn at random from the dataset have different classes:
$$\text{Gini}(D) = \sum_{k=1}^{K} p_k(1-p_k) = 1 - \sum_{k=1}^{K} p_k^2$$
- After a split point on feature $A$ divides $D$ into $D_1, D_2$:
$$\text{Gini}(D,A) = \frac{|D_1|}{|D|}\text{Gini}(D_1) + \frac{|D_2|}{|D|}\text{Gini}(D_2)$$
CART picks the feature and split point that **minimize** the post-split Gini index.
- Relation to entropy: a first-order Taylor expansion of $-\ln p$ at $p=1$ gives $-\ln p \approx 1-p$; substituting into entropy yields $\sum p_k(1-p_k)$, so **Gini can be viewed as a first-order approximation of entropy**. The curves have similar shapes and in practice the difference is usually small; Gini needs no logarithm, so it is **faster to compute**; entropy is slightly more sensitive to changes in the class distribution.
- For binary classification: $\text{Gini}=2p(1-p)$, misclassification rate $=\min(p,1-p)$, entropy $=-p\log p-(1-p)\log(1-p)$; all three peak at $p=0.5$. Misclassification rate is not differentiable and insensitive to changes in node purity, so it is generally not used to grow trees, but is often used to evaluate pruning.

## The Three Classic Algorithms

Q6: ***What are the differences between ID3, C4.5 and CART?***

| | ID3 | C4.5 | CART |
|---|---|---|---|
| Split criterion | Information gain | Gain ratio (heuristic) | Classification: Gini index; regression: squared error |
| Task | Classification | Classification | Classification + regression |
| Tree structure | Multi-way | Multi-way | **Binary** |
| Continuous features | Not supported | Supported (binary discretization) | Supported |
| Missing values | Not supported | Supported (sample weight distribution) | Supported (surrogate splits) |
| Pruning | None | Post-pruning such as Pessimistic Error Pruning (PEP) | Cost-complexity pruning (CCP) |
| Feature reuse | A discrete feature is not reused once used | Discrete features not reused, continuous features can be | Features can be reused at different nodes |

- Note: sklearn's `DecisionTreeClassifier/Regressor` implements an **optimized version of CART** (binary trees, `criterion` can be `gini`/`entropy`/`log_loss`), not ID3/C4.5.

Q7: ***How do decision trees handle continuous features?***

- **Binary discretization** (C4.5 / CART): sort the values of feature $a$ at the current node as $a^1<a^2<\dots<a^n$, take midpoints of adjacent values $t_i = \frac{a^i + a^{i+1}}{2}$ as candidate split points ($n-1$ of them), evaluate the criterion for each and pick the best.
- Optimization: the optimal split point can only lie **between adjacent points where the class changes** (true for entropy/Gini), which greatly reduces the candidates; for large data, **histogram / quantile binning** approximations are used (common in GBDT frameworks).
- Unlike discrete features, continuous features **can be reused** in later nodes (e.g. first $x\le 5$, then $x \le 2$).
- Complexity: sorting per node per feature is $O(n\log n)$, which can be amortized by pre-sorting.

Q8: ***How do decision trees handle missing values?***

Two problems must be solved:

1. **How to choose the split feature when values are missing?** (C4.5) Compute information gain only on the **subset $\tilde D$ without missing values** for that feature, then multiply by the fraction of non-missing samples $\rho$: $\text{Gain}(D,a) = \rho \cdot \text{Gain}(\tilde D, a)$.
2. **Once the feature is chosen, which branch does a sample with a missing value go to?**
    - **C4.5**: the sample **enters all child nodes simultaneously with different weights**, the weights being the fraction $\tilde r_v$ of non-missing samples in each child; at prediction time it also goes down all branches and the probabilities are combined by weight.
    - **CART**: uses **surrogate splits**, i.e. other features whose split most closely mimics the primary split, to decide where missing samples go.
    - **XGBoost / LightGBM**: during training, learn a **default direction** for each split by putting missing values left and right and keeping the direction with higher gain.
    - **sklearn** (since 1.3): tree models support NaN natively; at each split they likewise try sending missing samples to the left/right child and keep the better side.
- In practice simple imputation also works (median, mode, or a sentinel like -999); trees can usually learn the meaning of a sentinel value.

Q9: ***How is a regression tree built?***

- A CART regression tree uses **squared-error minimization**. For feature $j$ and split point $s$, split into $R_1(j,s)=\{x|x_j\le s\}$ and $R_2(j,s)=\{x|x_j>s\}$ and solve:
$$\min_{j,s}\left[\min_{c_1}\sum_{x_i\in R_1}(y_i-c_1)^2 + \min_{c_2}\sum_{x_i\in R_2}(y_i-c_2)^2\right]$$
The inner optimum is $c_m = \text{mean}(y_i \mid x_i \in R_m)$, i.e. each leaf outputs the mean of its samples.
- Iterate over all $(j,s)$ to find the best split and recurse; the resulting model is a **piecewise-constant function**: $f(x)=\sum_{m=1}^{M} c_m \mathbb{I}(x\in R_m)$.
- MAE (leaf outputs the median, more robust to outliers) or Poisson deviance criteria can also be used.
- Limitations: predictions can only be leaf means from the training set, so trees **cannot extrapolate** beyond the training label range, and the output is discontinuous and not smooth.

## Pruning and Overfitting

Q10: ***Why do decision trees overfit easily? How do you counter it?***

- Reason: a decision tree is a low-bias, **high-variance** model. Unconstrained, it can keep splitting until each leaf contains one sample, reaching zero training error and fitting the noise completely; also, greedy splitting is very sensitive to data perturbations, and changing a small part of the data can yield a completely different tree.
- Countermeasures:
    - **Pre-pruning**: limit `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `min_impurity_decrease`;
    - **Post-pruning**: REP, PEP, CCP, etc.;
    - **Ensembles**: Bagging / random forests reduce variance; Boosting uses shallow trees to reduce bias;
    - Feature selection, more data.

Q11: ***What is the difference between pre-pruning and post-pruning? Pros and cons?***

- **Pre-pruning**: during growth, before each split, evaluate whether the split improves generalization (e.g. validation accuracy, impurity-decrease threshold, depth/sample-count limits); if not, stop and mark the current node as a leaf.
    - Pros: low training and test cost, simple to implement.
    - Cons: **"short-sighted"**, risk of underfitting. A split may look unhelpful now while its subsequent splits would bring a large improvement (the classic example is XOR: splitting on either feature alone gives no gain).
- **Post-pruning**: first grow a full tree, then examine each internal node bottom-up; if replacing its subtree with a leaf improves generalization, prune it.
    - Pros: keeps more branches, **lower underfitting risk, usually better generalization than pre-pruning**.
    - Cons: a full tree must be grown first, so training is more expensive.
- Common post-pruning methods: **REP** (Reduced Error Pruning, uses a separate validation set), **PEP** (Pessimistic Error Pruning, C4.5, applies a continuity correction to training error, no validation set needed), **CCP** (Cost-Complexity Pruning, CART).

Q12: ***Explain CART's cost-complexity pruning (CCP) in detail.***

- Define the cost-complexity loss of subtree $T$:
$$C_\alpha(T) = C(T) + \alpha |T|$$
where $C(T)$ is the training error (e.g. weighted sum of leaf Gini / misclassification / squared error), $|T|$ is the number of leaves, and $\alpha\ge0$ trades off fit against complexity. At $\alpha=0$ the full tree is optimal; as $\alpha\to\infty$ only the root remains.
- For internal node $t$, compare "prune to a single leaf" with "keep subtree $T_t$": the two losses are equal when
$$\alpha = g(t) = \frac{C(t) - C(T_t)}{|T_t| - 1}$$
$g(t)$ measures **how much error increases per leaf removed** by pruning that subtree.
- Algorithm (**weakest link pruning**):
    1. For the full tree $T_0$, compute $g(t)$ for every internal node, prune the node with the smallest $g(t)$ to get $T_1$, and record $\alpha_1 = \min g(t)$;
    2. Repeat until only the root remains, giving a nested sequence of subtrees $T_0 \supset T_1 \supset \dots \supset T_n$ with corresponding $\alpha$ intervals $[\alpha_i, \alpha_{i+1})$;
    3. Use **cross-validation** or a held-out validation set to pick the best subtree from the sequence (i.e. the best $\alpha$).
- In sklearn this is the `ccp_alpha` parameter; `cost_complexity_pruning_path()` returns the $\alpha$ sequence:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_train, y_train)
scores = [cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=0),
                          X_train, y_train, cv=5).mean() for a in path.ccp_alphas]
best_alpha = path.ccp_alphas[max(range(len(scores)), key=scores.__getitem__)]
```

## Feature Importance and Interpretability

Q13: ***How is feature importance computed in decision trees? What biases does it have?***

- sklearn's default `feature_importances_` is **MDI (Mean Decrease in Impurity, Gini importance)**: the importance of feature $j$ is the sum of **weighted impurity decreases** over all nodes that split on $j$, then normalized:
$$\text{Imp}(j) = \sum_{t:\,v(t)=j} \frac{N_t}{N}\left[I(t) - \frac{N_{t_L}}{N_t} I(t_L) - \frac{N_{t_R}}{N_t} I(t_R)\right]$$
- Biases and issues:
    - **Biased toward high-cardinality features** (continuous features, categorical features with many levels, ID-like features): they have many candidate split points and are more likely to "accidentally" find a split that reduces impurity, so even pure-noise features can get non-trivial importance;
    - **Computed on the training set**, so it reflects how the model fits training data rather than true contribution to generalization; an overfit tree amplifies this;
    - **Correlated features** share/steal importance, so individual features' importance gets underestimated;
    - It only gives global magnitude, not direction or per-sample explanations.
- Alternatives: **permutation importance** (shuffle a feature on the validation set and measure the performance drop; also has issues with correlated features), **drop-column importance**, **SHAP** (TreeSHAP, with consistency guarantees, giving per-sample contributions and direction).

Q14: ***Why are decision trees considered interpretable? What are the limits of that interpretability?***

- Strengths:
    - The model is a set of **human-readable if-then rules**, can be visualized, and each prediction can be traced back to a root-to-leaf path;
    - It performs **feature selection** naturally, showing which features drive decisions;
    - It is a "white-box" model, easy to explain to business stakeholders and regulators (e.g. credit approval).
- Limits:
    - Only **shallow trees** are truly interpretable; deep trees have hundreds or thousands of rules that humans cannot grasp;
    - **Instability**: small data perturbations can change the tree structure drastically, changing the explanation and reducing trust;
    - Rules reflect correlation, not causation;
    - Once ensembled into a random forest/GBDT, interpretability drops sharply and tools such as SHAP are needed.

Q15: ***Why don't decision trees need feature normalization/standardization?***

- Tree splits depend only on the **ordering of feature values** ($x_j \le t$ splits samples into two groups). Any **monotonic transformation** of a feature (scaling, shifting, taking logs, etc.) does not change the ordering of samples, so it does not change the partition or the impurity; only the threshold $t$ is transformed accordingly.
- Decision trees involve no distance computations and no gradient-descent parameter optimization, so there is no issue of "large-scale features dominating distances" or "poor conditioning slowing convergence".
- Likewise, trees are fairly robust to **outliers in features** (an outlier only affects its rank position). Note, however, that **label** outliers in regression trees still affect the MSE criterion and leaf means.

Q16: ***How do decision trees handle categorical features?***

- ID3/C4.5: multi-way splits, one branch per value (prone to fragmentation and biased toward many-valued features).
- CART: binary splits, which requires partitioning $K$ categories into two subsets; brute force has $2^{K-1}-1$ options. For binary classification (or regression) there is a classic result: after **sorting categories by their positive-class rate (or mean label)**, the optimal partition is always a cut point in that sorted sequence, so only $K-1$ options need checking (Fisher 1958 / Breiman et al.). LightGBM's categorical handling borrows this idea.
- sklearn's decision trees (as of now) **do not natively support categorical features**, requiring one-hot, ordinal or target encoding. With one-hot, high-cardinality features make the tree very unbalanced and each split can only peel off one category, which often performs worse than native categorical splits.

## Pros, Cons and Comparisons

Q17: ***What are the pros and cons of decision trees?***

Pros:

- Highly interpretable and intuitive to visualize;
- Almost no preprocessing needed: **no normalization**, handles numeric and categorical features, handles missing values (implementation-dependent);
- Automatically captures **nonlinear relationships and feature interactions**;
- Fast prediction, complexity $O(\text{depth})$;
- Handles multi-class and multi-output problems.

Cons:

- Prone to **overfitting**, needing pruning or ensembling;
- **Unstable** (high variance): tiny data changes can yield a completely different tree;
- **Greedy** construction only reaches a local optimum;
- Axis-aligned splits struggle to represent oblique/diagonal decision boundaries efficiently (requiring many staircase splits);
- Regression output is piecewise constant and **cannot extrapolate**;
- Biased toward the majority class under class imbalance;
- Hard to represent concepts like XOR or parity that require joint use of multiple features (no single feature has gain under greedy search).

Q18: ***How do decision trees differ from logistic regression? When is each appropriate?***

- **Decision boundary**: logistic regression has a linear boundary (unless features are hand-crafted); a tree has axis-aligned piecewise boundaries and can express nonlinearity.
- **Feature interactions**: trees learn interactions automatically; LR needs manually constructed cross features.
- **Preprocessing**: LR needs standardization, missing-value handling, one-hot for categoricals, and is sensitive to outliers; trees mostly need none of this.
- **Output**: LR gives smooth, reasonably calibrated probabilities; a tree outputs leaf frequencies, which are coarse probabilities.
- **Stability and variance**: LR has low variance and is stable; a single tree has high variance.
- **Extrapolation**: LR can extrapolate linear trends; trees cannot.
- Use cases: LR when the feature-target relation is roughly linear, stable probability outputs are needed, or features are high-dimensional and sparse (e.g. text, ad ID features); trees when there are complex nonlinearities and interactions or rule-style explanations are needed (in practice, tree ensembles are usually used).

Q19: ***What is the time complexity of a decision tree?***

- Training: with $n$ samples and $d$ features, each level scans all samples for all features; with pre-sorting each level costs about $O(nd)$, a balanced tree has depth $O(\log n)$, so the total is about $O(nd\log n)$ (about $O(nd\log^2 n)$ if sorting is redone at each node). The sklearn documentation gives a typical cost on the order of $O(n_{\text{features}}\, n_{\text{samples}} \log n_{\text{samples}})$.
- Prediction: $O(\text{depth})$ per sample, about $O(\log n)$ for a balanced tree, $O(n)$ in the worst case.
- Speedups: histogram/binning approximations (discretizing continuous features into $B$ bins reduces the per-level cost to $O(nd + Bd)$), feature subsampling, parallelization.

Q20: ***How do decision trees relate to random forests and GBDT?***

- Decision trees are the **base learners** of both ensemble families:
    - **Random forest** (Bagging): trains many **deep trees** (low bias, high variance) on bootstrap samples with random feature subsets, and averages them to **reduce variance**;
    - **GBDT** (Boosting): trains many **shallow trees** (high bias, low variance) sequentially, each fitting the negative gradient of the current model, to **reduce bias**.
- Reasons CART is chosen as the base learner: handles both classification and regression, binary structure is easy to implement, insensitive to feature scale, models interactions automatically, fast to train.
- The trees in GBDT are always **regression trees**, even for classification tasks, because they fit continuous negative gradients. See the random forest and gradient boosting cards for details.
