# Gradient Boosted Trees (GBDT / XGBoost / LightGBM / CatBoost)

## Boosting Basics

Q1: ***What is the difference between Boosting and Bagging?***

**Bagging trains many independent strong learners in parallel and averages them, which mainly reduces variance. Boosting trains weak learners one after another, each correcting the previous errors, which mainly reduces bias.**

| | Bagging (e.g. random forest) | Boosting (e.g. AdaBoost, GBDT) |
|---|---|---|
| Training | Parallel, independent | Sequential, depends on previous results |
| Base learner | Deep trees (low bias, high variance) | Shallow trees (high bias, low variance) |
| Main effect | Reduces variance | Reduces bias |
| Noise | Robust | Fairly sensitive, can fit noise |

- Intuition: Bagging is "several experts vote independently"; Boosting is "fix the mistakes one step at a time".

Q2: ***Briefly explain how AdaBoost works.***

**AdaBoost trains weak classifiers in sequence. Each round it raises the weights of misclassified samples so the next classifier focuses on the hard ones. The final prediction is a weighted vote, where more accurate classifiers get more say.**

- The lower a classifier's error rate, the larger its voting weight.
- Misclassified samples get heavier, correctly classified ones lighter, then the weights are renormalized.
- Theoretically it is a forward stagewise additive model with **exponential loss**. That loss punishes misclassified samples very hard, so AdaBoost is sensitive to noise and outliers.
- Key formula (classifier weight): $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$

## GBDT

Q3: ***What is the core idea of GBDT? Why is it called "gradient descent in function space"?***

**GBDT is an additive model that sums many regression trees. Each new tree fits the negative gradient of the loss for the current model. You can treat the whole prediction function as the parameter and take a small step along the negative gradient each round. That is why it is called gradient descent in function space.**

- The negative gradient can only be computed at training points. Fitting a tree to it is what lets the model generalize to new samples.
- Under squared loss the negative gradient is exactly the residual, so "fitting residuals" is just a special case.
- Key formula (pseudo-residual): $r_{i} = -\partial L(y_i, F(x_i)) / \partial F(x_i)$

Q4: ***Write out the GBDT algorithm.***

**Start from a constant model. Each round, compute every sample's negative gradient, fit a regression tree to it, find the best output value for each leaf, then scale by the learning rate and add the tree to the model.**

- Initialize with the constant that minimizes the loss (the mean for squared loss, the log-odds for log loss).
- The tree's **structure** comes from fitting the negative gradient. Its **leaf values** are re-optimized on the original loss (line search).
- After M rounds, the output is the sum of all trees.

Q5: ***How does GBDT do classification?***

**The base learners are always regression trees. For classification GBDT boosts in log-odds (logit) space, then turns the score into a probability with a sigmoid or softmax.**

- Binary classification uses log loss, and the negative gradient is "true label minus predicted probability", $y - p$.
- Leaf values have no closed form, so one Newton step is usually used to approximate them.
- Multiclass (K classes) trains K trees per round, one per class, and a softmax over their outputs gives the probabilities.

Q6: ***What do shrinkage (learning rate) and subsampling do in GBDT?***

**Both are regularization. The learning rate makes each tree take only a small step, which leaves room for later trees to correct it. Subsampling adds randomness, which lowers variance, reduces overfitting and speeds up training.**

- In practice "small learning rate + many trees" generalizes better but is slower. Tune the learning rate and the number of trees together, with early stopping.
- Row subsampling: each round trains on only part of the samples (Stochastic Gradient Boosting).
- Column subsampling: each tree, level or node uses only some of the features. XGBoost borrowed this from random forests.

Q7: ***Which loss functions are commonly used with GBDT?***

**Any loss with a first derivative works with GBDT. Regression commonly uses squared, absolute, Huber and quantile loss. Classification uses log loss, and ranking uses LambdaMART.**

- Squared loss is sensitive to outliers. Absolute loss and Huber loss are more robust.
- Quantile loss gives prediction intervals. Poisson, Gamma and Tweedie suit counts, positive values and similar targets.
- Exponential loss corresponds to AdaBoost.
- XGBoost also needs the second derivative, so a custom loss must provide both grad and hess.

## XGBoost

Q8: ***What is XGBoost's objective function? How is it derived with a second-order Taylor expansion?***

**The objective is "training loss + a complexity penalty on the tree", where the penalty covers the number of leaves and the size of the leaf weights. A second-order Taylor expansion of the loss around the previous round's prediction turns each leaf's objective into a quadratic in its leaf weight, which can be solved directly.**

- Each sample needs only its first derivative $g_i$ and second derivative $h_i$, summed per leaf into $G_j$ and $H_j$.
- After dropping constants the leaves are independent, so each can be optimized on its own.
- Key formula: $\Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_j w_j^2$

Q9: ***Derive XGBoost's optimal leaf weight and split gain formula.***

**Minimizing each leaf's quadratic gives the optimal leaf weight: minus the gradient sum divided by (Hessian sum + λ). Plugging it back gives the tree's structure score. The split gain is the score of the left and right children minus the score of the parent, minus γ.**

- Each node's score has the form $G^2/(H+\lambda)$. Gain = left score + right score − parent score (times 1/2). The node splits only if the gain is still above 0 after subtracting γ.
- Exact greedy: sort each feature and accumulate $G_L, H_L$ from left to right, so one scan gives the gain of every split point.
- The optimal weight is a Newton step with L2 regularization, so XGBoost can be seen as Newton's method in function space.
- Key formula: $w_j^* = -\frac{G_j}{H_j+\lambda}$

Q10: ***What roles do $\gamma$, $\lambda$ and `min_child_weight` play in XGBoost?***

**γ is the cost of adding one more leaf, so a split with too little gain is skipped. λ is L2 regularization on leaf weights, which makes predictions smoother. `min_child_weight` requires each child's Hessian sum to stay above a threshold, so the tree does not split off a tiny handful of samples.**

- γ acts as built-in pruning. XGBoost first grows to `max_depth`, then prunes negative-gain splits bottom-up, which avoids the short-sightedness of pre-pruning.
- λ also appears in the gain denominator, so it holds back leaves with few samples. α is L1 regularization and pushes small weights to 0.
- Under squared loss $h_i=1$, so `min_child_weight` is just a minimum sample count. Under log loss, samples the model is already confident about have very small $h_i$.

Q11: ***Why does XGBoost use second-order derivatives? What are the benefits?***

**The second derivative turns gradient descent into Newton's method. That approximates the loss more accurately, so training usually converges faster and needs fewer trees. It also lets any twice-differentiable loss plug into the same split and leaf-weight formulas.**

- Unified framework: a custom loss only needs grad and hess. There is no separate line-search derivation for each loss.
- The regularizers λ and γ show up in closed form in the leaf weight and the gain.
- $h_i$ can be read as a sample weight. This is the basis of the weighted quantile sketch.
- The loss must be twice differentiable with $h_i>0$. For example, MAE has a zero second derivative and needs special handling.

Q12: ***What engineering optimizations does XGBoost have? (sparsity awareness, weighted quantile sketch, block structure)***

**XGBoost mainly speeds up finding split points. A weighted quantile sketch cuts down the candidate splits, sparsity-aware splitting handles missing values efficiently, and a pre-sorted column block structure allows feature-level parallelism. Cache optimizations and out-of-core computation are added on top.**

- Weighted quantile sketch: propose candidate splits at quantiles weighted by $h_i$, instead of scanning every value.
- Sparsity awareness: each node learns a default direction for missing values (try all-left and all-right, keep the higher gain), and only non-missing values are scanned.
- Column blocks: data is sorted by column once before training, stored in blocks and reused. Different features can be processed in parallel, but trees are still built sequentially.
- Cache-aware prefetching and out-of-core computation (blocks on disk, compressed and sharded) support large data.
- The histogram method (`tree_method="hist"`) is now the default and GPU training is supported, so exact greedy is rarely used.

Q13: ***What improvements does XGBoost make over traditional GBDT?***

**On the algorithm side, XGBoost uses a second-order Taylor expansion and puts explicit regularization in the objective. On the engineering side, it adds a lot of speedups. The result is a GBDT implementation that is more accurate, faster and less prone to overfitting.**

- Algorithm: second derivatives (Newton's method), regularization on leaf count and leaf weights, column sampling, post-pruning with γ.
- Missing values: sparsity awareness that learns a default direction automatically.
- Split finding: exact, weighted-quantile approximate and histogram methods.
- Engineering: feature-level parallelism, pre-sorted column blocks, cache awareness, out-of-core, distributed and GPU training.
- Usability: linear base learners and DART, plus built-in cross-validation, early stopping and custom losses.

## LightGBM

Q14: ***What is LightGBM's histogram algorithm? What are its advantages?***

**Before training, each continuous feature is bucketed into at most 255 bins, and each sample stores only its bin index. When splitting, gradients are summed per bin to build a histogram, and split points are searched only at bin boundaries. This saves memory and is fast.**

- Small memory: feature values are stored as 8-bit integers, and no pre-sorted results are kept.
- Fast: the number of candidate splits drops from the number of samples to the number of bins.
- Histogram subtraction: subtracting one child's histogram from the parent's gives the other child's, so only the child with fewer samples has to be built.
- Coarse splits act as a regularizer. The cost is less precise split points, but the accuracy loss is usually tiny.

Q15: ***What is the difference between level-wise and leaf-wise tree growth?***

**Level-wise growth splits every leaf on a level at once. The tree stays balanced and is less prone to overfitting, but many splits are useless. Leaf-wise growth splits only the leaf with the highest gain. It is more accurate for the same number of leaves, but it can grow very deep trees and overfit on small data.**

- XGBoost defaults to level-wise and LightGBM to leaf-wise. XGBoost can also use `grow_policy="lossguide"`.
- Leaf-wise growth must be constrained with `num_leaves` (the main complexity parameter), `max_depth` and `min_data_in_leaf`.
- `num_leaves` should be below $2^{\text{max\_depth}}$, otherwise the depth limit does nothing.

Q16: ***What is GOSS (Gradient-based One-Side Sampling)?***

**GOSS keeps all large-gradient samples, randomly samples only from the small-gradient ones, and up-weights the sampled small-gradient samples to compensate. This estimates split gains fairly accurately from a small fraction of the data, which speeds up training.**

- Motivation: small-gradient samples are already well fit and contribute little to the gain, but simply dropping them would change the data distribution.
- Method: keep the top $a$ fraction of large-gradient samples, sample a $b$ fraction of the rest, and multiply those by the weight $\frac{1-a}{b}$.
- In newer versions enable it with `data_sample_strategy="goss"`. The parameters are `top_rate` (a) and `other_rate` (b).

Q17: ***What is EFB (Exclusive Feature Bundling)?***

**In high-dimensional sparse data, many features are almost never non-zero at the same time. EFB bundles these mutually exclusive features into a single feature. This cuts the feature count and speeds up histogram building, with almost no information loss.**

- Choosing which features to bundle can be modeled as graph coloring. It is solved greedily, and a few conflicts are allowed.
- When merging, each feature's values are shifted by an offset so they fall into non-overlapping bin ranges.
- The cost of building histograms then scales with the number of bundles instead of the number of features.

Q18: ***How does LightGBM handle categorical features?***

**LightGBM supports categorical features natively, with no one-hot needed. At each node it sorts the categories by "gradient sum / Hessian sum" and then searches for the best split as if the feature were continuous, so the categories are divided into two groups in one go.**

- Sorting and then splitting costs only $O(k\log k)$, instead of enumerating $2^k$ partitions.
- With very few categories (≤4 by default) it falls back to one-vs-rest.
- Why not one-hot: one-hot can only split off one category per split, so gains are small and trees become deep and unbalanced.
- High-cardinality categories can still overfit. Tune `cat_smooth`, `min_data_per_group` and similar parameters, or switch to target encoding or CatBoost.

Q19: ***How do LightGBM and XGBoost differ?***

**LightGBM uses histograms, leaf-wise growth, GOSS and EFB, so it is faster and uses less memory. Classic XGBoost uses pre-sorting and level-wise growth, which gives more balanced and more robust trees. When both are well tuned, their accuracy is usually similar.**

| | XGBoost | LightGBM |
|---|---|---|
| Split finding | Pre-sorted / quantile approximation (hist is now default) | Histogram + subtraction |
| Growth strategy | Level-wise | Leaf-wise |
| Sampling and sparsity | Row/column sampling, sparsity awareness | GOSS, EFB |
| Categorical features | Newer versions support `enable_categorical` | Native many-vs-many splits |

- Since XGBoost 2.x made hist the default and added GPU and categorical support, the speed gap has narrowed a lot.

## CatBoost

Q20: ***What are CatBoost's Ordered Target Statistics? What problem do they solve?***

**They are a target encoding that avoids target leakage. The samples are put in a random order, and each sample's category encoding uses only the label mean of the samples before it. So a sample's encoding never includes its own label.**

- Plain target encoding uses the sample's own label. The encoding is then strongly tied to the label during training but not at test time, which causes conditional shift and overfitting.
- A prior and a smoothing term keep the encoding stable for rare categories.
- Several random orderings are used to reduce variance. At test time the statistics are computed on all training data.
- CatBoost also greedily builds combinations of categorical features and applies ordered TS to them too.

Q21: ***What is Ordered Boosting? What is prediction shift?***

**Prediction shift happens because standard GBDT computes a sample's gradient with a model that has already seen that sample. Training residuals come out smaller than test residuals, so the gradient estimates are biased and the model overfits. Ordered Boosting computes each sample's residual with a model trained only on the samples before it in a random order, which removes this bias.**

- It is basically the same kind of problem as target leakage.
- In practice CatBoost does not keep n models. It keeps approximate models only for prefix lengths that grow exponentially, and they share tree structures.
- The gain is clear on small datasets. On large data, `boosting_type="Plain"` is faster.

Q22: ***Why does CatBoost use symmetric (oblivious) trees?***

**In a symmetric tree every node on a level uses the same feature and threshold. Prediction takes only d comparisons, which form a d-bit index into a lookup table, so inference is very fast. The strong structural constraint also acts as regularization and makes overfitting less likely.**

- Inference vectorizes easily and is clearly faster than ordinary trees on CPU.
- It is not very sensitive to hyperparameters, and the defaults already work well.
- The downside is limited expressiveness: the same depth needs more trees. CatBoost also supports Depthwise and Lossguide growth.

## Tuning and Practice

Q23: ***What are the key hyperparameters of GBDT-style models? What is the tuning order?***

**The key parameters fall into four groups: number of trees and learning rate, tree complexity, sampling ratios, and regularization. A common order is: use a fairly large learning rate with early stopping to fix the number of trees, then tune tree structure, then sampling and regularization, and finally lower the learning rate and add more trees.**

- Tree complexity (the biggest effect): `max_depth`, `num_leaves`, `min_child_weight` / `min_data_in_leaf`.
- Sampling: `subsample` and `colsample_bytree`, usually 0.6–0.9.
- Regularization: `reg_lambda`, `reg_alpha`, `gamma`.
- In practice, Bayesian optimizers such as Optuna search these jointly and are more efficient than grid search.

Q24: ***How is early stopping used in GBDT? What should you watch out for?***

**Monitor a validation metric during training. Stop once it has not improved for a number of rounds (e.g. 50–200), and keep the model from the best round.**

- The early-stopping validation set is itself "fit", so the final evaluation needs a separate test set or nested cross-validation.
- In cross-validation each fold has a different best round. When retraining on all data, use the average (or scale it up a little for the larger data).
- The monitored metric should match the business goal, and the validation split should match production (split time series by time).
- In LightGBM, use `callbacks=[lgb.early_stopping(100)]` and then read `best_iteration_`.

Q25: ***How does GBDT handle class imbalance?***

**The most common fix is to up-weight the minority class (e.g. `scale_pos_weight` = negatives / positives). Resampling and custom losses are the next options. You also need suitable evaluation metrics and a tuned decision threshold.**

- Weighting enlarges the minority class's gradients and Hessians, which affects both splits and leaf values.
- You can also downsample the majority class or use Focal Loss (which needs first and second derivatives).
- Use PR-AUC, F1 and similar metrics instead of accuracy, and tune the threshold on the validation set instead of using the default 0.5.
- Weighting or resampling distorts probabilities. When accurate probabilities matter (e.g. CTR), calibrate them or correct for the sampling.

Q26: ***What kinds of feature importance does GBDT offer? What are their issues? What are SHAP's advantages?***

**The built-in importances count splits (split/weight), sum gains (gain) or measure the samples covered (cover). All of them are computed on the training set, they can rank features very differently, and they are not consistent. SHAP is based on Shapley values. It is guaranteed to be consistent and gives each feature's contribution and direction for every sample.**

- Split counts favor high-cardinality and continuous features. Gain reflects real contribution better but is still based on the training set.
- Inconsistency: when the model relies more on a feature, that feature's importance does not necessarily go up.
- TreeSHAP computes exact values on tree models in polynomial time and can also analyze interactions.
- Correlated features still share SHAP value. To judge contribution to generalization, add permutation importance on a validation set.

Q27: ***What is GBDT + LR? Why does it work?***

**This is Facebook's 2014 CTR prediction approach. First train a GBDT, one-hot the index of the leaf each sample lands in for every tree to get sparse features, then feed those to logistic regression. Each root-to-leaf path acts as an automatically learned feature combination, which saves a lot of manual feature engineering.**

- GBDT handles non-linear discretization and feature crosses. LR handles large-scale sparse features and can be updated online at high frequency.
- Train GBDT and LR on different data, so LR does not overfit to leaf features from an overfit GBDT.
- High-cardinality ID features are a poor fit for direct use in GBDT.
- It was later replaced by deep models such as FM, Wide & Deep and DeepFM, but the idea of using trees to transform features is still common.

Q28: ***Why is GBDT usually preferred over deep learning for tabular data? How do you choose between random forest and GBDT?***

**Tabular features are heterogeneous, there are many irrelevant features, and targets often have sharp jumps. Trees select features naturally, fit piecewise functions well, overfit less on small and medium data, need little preprocessing and train fast, so they usually beat neural networks. Random forest makes a robust baseline, while a well-tuned GBDT is usually more accurate.**

- Neural networks prefer smooth functions and are rotation-invariant, so they struggle to use the "axis-aligned" meaning of features (Grinsztajn et al. 2022).
- Consider deep learning for very large data, unstructured inputs like text or images, or end-to-end representation learning. TabPFN is worth trying on small data, and a GBDT + NN ensemble is often best.

| | Random forest | GBDT |
|---|---|---|
| Ensemble type | Bagging, parallel, reduces variance | Boosting, sequential, reduces bias |
| Base trees | Deep trees | Shallow trees (depth 3–8) |
| Tuning and overfitting | Few parameters, more trees only helps | Sensitive to learning rate/depth, needs early stopping |
| Noise | Robust | Fairly sensitive |

- How to choose: use random forest for noisy data or a small tuning budget. Use GBDT for top accuracy, and prefer CatBoost when there are many categorical features.
