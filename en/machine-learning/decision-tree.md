# Decision Tree

## Basics and Split Criteria

Q1: ***What is a decision tree and how does it make predictions?***

**A decision tree is a non-parametric supervised learning model. It recursively cuts the feature space into rectangular regions, and each region is a leaf. To predict, you start at the root, follow the feature test at each node down to a leaf, and output that leaf's value.**

- Internal nodes are feature tests (e.g. $x_j \le t$), and leaves store the predictions.
- Classification outputs the majority class in the leaf (or the class proportions). Regression outputs the mean label in the leaf.
- It is basically a set of if-then rules, a piecewise-constant approximation of $P(Y|X)$.

Q2: ***How is a decision tree built? When does splitting stop?***

**The tree is built top-down by greedy recursive splitting. At the current node, try every feature and split point and pick the split that reduces impurity the most. Send the samples to the child nodes and repeat on each child until a stopping condition is met.**

- Stopping conditions: all samples in the node are one class, no usable features remain or all feature values are identical, too few samples, maximum depth reached, or the impurity decrease is below a threshold.
- Finding the globally optimal tree is NP-complete, so only greedy local choices are practical. This is also why trees are sensitive to small changes in the data.

Q3: ***How are entropy, conditional entropy and information gain defined?***

**Entropy measures how uncertain the class labels in a dataset are. Conditional entropy is the weighted average entropy of the subsets after splitting on a feature. Information gain is how much the entropy drops from the split, which is the mutual information between the feature and the label. ID3 picks the feature with the highest information gain.**

- Entropy is 0 when all samples are one class and is maximal ($\log_2 K$) when the classes are uniform.
- The weights in conditional entropy are the subset sample proportions.
- Key formula: $H(D) = -\sum_k p_k \log_2 p_k$

Q4: ***What is wrong with information gain? How does gain ratio fix it?***

**Information gain favors features with many values. In the extreme, splitting on "sample ID" gives the highest gain but does not generalize at all. C4.5's gain ratio divides the information gain by the feature's intrinsic value, which grows with the number of values, so many-valued features are penalized.**

- But gain ratio in turn favors features with few values.
- So C4.5 uses a heuristic: first keep the features with above-average information gain, then pick the one with the highest gain ratio.
- Key formula: $\text{GainRatio}(D,A) = \text{Gain}(D,A) / IV(A)$

Q5: ***What is the Gini index and how does it relate to entropy?***

**The Gini index is the probability that two samples drawn at random from the dataset have different classes, so lower means purer. CART picks the split with the lowest weighted Gini after splitting. Gini can be seen as a first-order Taylor approximation of entropy. The two usually perform about the same, but Gini needs no logarithm and is faster.**

- Entropy is slightly more sensitive to changes in the class distribution.
- For binary classification, Gini, entropy and misclassification rate all peak at p=0.5.
- Misclassification rate is insensitive to changes in purity, so it is rarely used to grow trees, but it is common for evaluating pruning.
- Key formula: $\text{Gini}(D) = 1 - \sum_k p_k^2$

## The Three Classic Algorithms

Q6: ***What are the differences between ID3, C4.5 and CART?***

**ID3 uses information gain, does classification only and supports only discrete features. C4.5 switches to gain ratio and supports continuous features, missing values and pruning. CART builds binary trees, using Gini for classification and squared error for regression, so it handles both tasks.**

| | ID3 | C4.5 | CART |
|---|---|---|---|
| Split criterion | Information gain | Gain ratio | Gini / squared error |
| Task and tree shape | Classification, multiway | Classification, multiway | Classification + regression, binary |
| Continuous features / missing values | Neither | Both | Both (surrogate splits) |
| Pruning | None | Pessimistic error pruning (PEP) | Cost-complexity pruning (CCP) |

- sklearn's decision trees implement an optimized CART, not ID3/C4.5.

Q7: ***How do decision trees handle continuous features?***

**They use binary splits. Sort the feature's values at the current node, take the midpoints between adjacent values as candidate split points, evaluate the criterion at each one and keep the best.**

- The best split point can only lie between adjacent values where the class changes, which cuts the candidates a lot.
- On large data, histogram or quantile bucketing approximates this (common in GBDT).
- Unlike discrete features, a continuous feature can be used again at later nodes.

Q8: ***How do decision trees handle missing values?***

**There are two problems to solve. When choosing a feature, C4.5 computes the information gain only on samples where that feature is present, then multiplies it by the fraction of such samples. When routing samples, different algorithms do different things.**

- C4.5: a sample with a missing value goes down every branch at once, weighted by each branch's share of samples.
- CART: surrogate splits, i.e. use another feature whose split best mimics the main split to decide where the sample goes.
- XGBoost / LightGBM / sklearn (since 1.3): try sending missing values left and right, and keep the direction with higher gain as the default.
- In practice you can also just impute (median, or a special value like -999), and trees usually learn what it means.

Q9: ***How is a regression tree built?***

**A CART regression tree splits to minimize squared error. For each feature and split point, divide the samples in two, predict each part with its mean, keep the split with the lowest total squared error, and recurse. The result is a piecewise-constant function where each leaf outputs the mean of its samples.**

- You can also use MAE (leaf takes the median, more robust to outliers) or Poisson deviance.
- Limits: it cannot extrapolate beyond the training label range, and the output is discontinuous and not smooth.
- Key formula: $\min_{j,s}\left[\sum_{x_i\in R_1}(y_i-c_1)^2 + \sum_{x_i\in R_2}(y_i-c_2)^2\right]$, where $c_m$ is the region mean

## Pruning and Overfitting

Q10: ***Why do decision trees overfit easily? How do you counter it?***

**A decision tree is a low-bias, high-variance model. Without limits it can keep splitting until every leaf holds a single sample, learning the noise too. Greedy splitting is also very sensitive to data perturbations. The fixes are pruning and ensembling.**

- Pre-pruning: limit `max_depth`, `min_samples_leaf`, `max_leaf_nodes`, `min_impurity_decrease`, etc.
- Post-pruning: REP, PEP, CCP.
- Ensembles: random forests reduce variance, and boosting uses shallow trees to reduce bias.
- Feature selection and more data also help.

Q11: ***What is the difference between pre-pruning and post-pruning? Pros and cons?***

**Pre-pruning checks before each split, during growth, whether the split improves generalization, and stops if it does not. Post-pruning first grows the full tree, then works bottom-up and replaces subtrees that are not worth keeping with leaves. Pre-pruning is fast but can underfit. Post-pruning usually generalizes better but is slower.**

- Pre-pruning is "short-sighted": a split with little immediate gain may enable later splits with large gain (e.g. the XOR problem).
- Post-pruning keeps more branches and has less underfitting risk, but it must grow the full tree first, which is expensive.
- Common post-pruning methods: REP (uses a validation set), PEP (C4.5, no validation set needed), CCP (CART).

Q12: ***Explain CART's cost-complexity pruning (CCP) in detail.***

**CCP adds a penalty of "α × number of leaves" to the training error. Starting from the full tree, it repeatedly prunes the subtree whose removal adds the least error per leaf removed (the weakest link). This gives a sequence of nested, ever-smaller subtrees, each matching a range of α. Cross-validation then picks the best one.**

- With α=0 the full tree is best. With a very large α only the root remains.
- For each internal node, compute how much error goes up per leaf removed if its subtree is pruned. The smallest one is pruned first.
- In sklearn, `cost_complexity_pruning_path()` gives the α sequence, and cross-validation then picks `ccp_alpha`.
- Key formula: $C_\alpha(T) = C(T) + \alpha |T|$

## Feature Importance and Interpretability

Q13: ***How is feature importance computed in decision trees? What biases does it have?***

**sklearn uses MDI (Mean Decrease in Impurity) by default. For each feature it sums the weighted impurity decrease over all nodes that split on it, then normalizes. It is biased toward high-cardinality features, it is computed on the training set, and correlated features split importance between them.**

- High-cardinality features (continuous, many categories, ID-like) have many candidate splits and can reduce impurity "by chance". Even a random noise feature can score fairly high.
- It is based on the training set, so it shows how the model fits the data, not how it generalizes. Overfit trees make this worse.
- It gives only a global magnitude, with no direction and no per-sample explanation.
- Alternatives: permutation importance on a validation set, drop-column importance, TreeSHAP.

Q14: ***Why are decision trees considered interpretable? What are the limits of that interpretability?***

**Because the model is a set of human-readable if-then rules that can be drawn, and every prediction traces back to one root-to-leaf path. But this only holds for shallow trees, and trees are very unstable.**

- Pros: built-in feature selection, and as a white-box model it is easy to explain to business and regulators (e.g. credit approval).
- Limit: deep trees have hundreds or thousands of rules that no one can follow.
- Small changes in the data can change the tree a lot, and the explanation changes with it.
- Rules show correlation, not causation. Once trees are ensembled into a random forest or GBDT, tools like SHAP are needed.

Q15: ***Why don't decision trees need feature normalization/standardization?***

**Because splits only depend on the order of feature values. Monotonic transforms such as scaling, shifting or taking logs do not change the order, so the partition and impurity stay the same and only the threshold moves. Trees also compute no distances and do not use gradient descent, so feature scale has no effect.**

- For the same reason, outliers in a feature only change sort positions, so trees are fairly robust to them.
- But outliers in the label of a regression tree still affect the squared error and the leaf means.

Q16: ***How do decision trees handle categorical features?***

**ID3/C4.5 make one branch per value. CART splits in two, so it must divide the categories into two groups. For binary classification or regression, sort the categories by their positive rate (or mean label); the best split is always at some cut point in that order, so only K−1 splits need checking instead of exponentially many.**

- Multiway splits fragment the data and favor many-valued features.
- The sorting trick comes from Fisher 1958 / Breiman et al., and LightGBM borrows it.
- sklearn's decision trees do not support categorical features natively, so you need one-hot, ordinal or target encoding. One-hot on high-cardinality features makes trees deep and unbalanced.

## Pros, Cons and Comparisons

Q17: ***What are the pros and cons of decision trees?***

**Pros: interpretable, almost no preprocessing, captures non-linearity and feature interactions automatically, and fast to predict. Cons: overfits easily, unstable, greedy so only locally optimal, axis-aligned splits struggle with diagonal boundaries, and regression cannot extrapolate.**

- More pros: handles numeric and categorical features and missing values (depending on the implementation), supports multiclass and multi-output, and prediction costs O(depth).
- More cons: biased toward the majority class under imbalance, and struggles with concepts like XOR or parity that need several features jointly.
- Overfitting and instability are usually handled with pruning and ensembles.

Q18: ***How do decision trees differ from logistic regression? When is each appropriate?***

**Logistic regression has a linear boundary and is stable with well-calibrated probabilities, but it needs hand-built interaction features and more preprocessing. A decision tree has axis-aligned piecewise boundaries and learns non-linearity and interactions automatically with almost no preprocessing, but it has high variance, coarse probabilities and cannot extrapolate.**

- LR needs standardization, missing-value handling and one-hot encoding of categories, and is sensitive to outliers. Trees need almost none of this.
- LR can extrapolate linear trends, and trees cannot.
- Use LR for roughly linear relationships, when stable probabilities matter, and for high-dimensional sparse features (text, ad IDs).
- Use trees for complex non-linearity and interactions, or when rule-style explanations are needed (in practice, tree ensembles).

Q19: ***What is the time complexity of a decision tree?***

**Training is roughly O(n d log n): each level scans n samples over d features, and a balanced tree has about log n levels. Predicting one sample is O(depth), about O(log n) for a balanced tree and O(n) in the worst case.**

- Re-sorting at every node is slower. Pre-sorting lets the order be reused.
- Speedups: histogram bucketing, feature sampling, parallelism.

Q20: ***How do decision trees relate to random forests and GBDT?***

**Decision trees are the base learner of both. A random forest trains many deep trees on bootstrap samples and random feature subsets, then averages them to reduce variance. GBDT trains many shallow trees in sequence, each fitting the negative gradient of the current model, to reduce bias.**

- CART is the usual base learner because it does both classification and regression, its binary structure is easy to implement, it ignores feature scale, it models interactions automatically and it trains fast.
- All trees in GBDT are regression trees, even for classification, because they fit a continuous negative gradient. See the random forest and gradient boosting cards for details.
