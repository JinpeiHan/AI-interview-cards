# Machine Learning Fundamentals

## Learning Paradigms and Theory

Q1: ***What is the difference between supervised, unsupervised and self-supervised learning?***

- **The difference is where the supervision signal comes from: supervised learning learns $x\to y$ from human labels; unsupervised learning has no labels and only looks for structure in the data; self-supervised learning builds "pseudo-labels" from the data itself and trains on them, with no human annotation.**
- Supervised: classification, regression; the cost is expensive labeling.
- Unsupervised: clustering (K-means), dimensionality reduction (PCA), density estimation, anomaly detection.
- Self-supervised: masked prediction (BERT, MAE), next-token prediction (GPT), contrastive learning (SimCLR, CLIP).
- The modern large-model recipe is usually "self-supervised pretraining + supervised fine-tuning + preference alignment (RLHF/DPO)"; there are also semi-supervised and reinforcement learning.

Q2: ***Derive the bias-variance decomposition***

- **A model's expected squared error splits into three parts: squared bias (how far the model's average prediction is from the true function), variance (how much the prediction jitters if you change the training set) and irreducible noise. The derivation inserts "the model's average prediction" into the error and expands the square; because the noise has zero mean and is independent of the training set, all cross terms vanish.**
- High bias: the model is too simple and systematically off → underfitting.
- High variance: the model is too sensitive to randomness in the training data → overfitting.
- More complex models usually have lower bias and higher variance, so there is a trade-off.
- Double descent in deep learning shows the classic U-shaped curve does not always hold: past the interpolation threshold, over-parameterized models' test error falls again, thanks to implicit regularization.
- Key formula: $E[(y-\hat f)^2]=\text{Bias}^2+\text{Variance}+\sigma^2$

Q3: ***How do you diagnose underfitting and overfitting with learning curves?***

- **Plot training and validation error against sample size (or training epochs): if both are high and close together, it is underfitting (high bias); if training error is low, validation error is high and there is a big gap, it is overfitting (high variance).**
- When underfitting, more data does not help; use a bigger model, more features, weaker regularization or longer training.
- When overfitting, more data helps; so do stronger regularization, a simpler model, data augmentation and early stopping.
- Compare with a baseline error (human level): the gap between training error and baseline ≈ avoidable bias; the gap between validation and training error ≈ variance.
- A validation curve (hyperparameter on the x-axis) finds the hyperparameter with the lowest validation error.

Q4: ***What is the difference between parametric and non-parametric models?***

- **A parametric model has a fixed number of parameters, independent of sample size, and makes strong assumptions about the function's form; a non-parametric model grows in complexity with the data and assumes less. "Non-parametric" does not mean no parameters; it means the number of parameters is not fixed.**
- Parametric: linear/logistic regression, naive Bayes, fixed-architecture neural networks; fast, data-efficient and interpretable, but high bias if the assumption is wrong.
- Non-parametric: KNN, decision trees, kernel SVM, Gaussian processes; flexible, but need more data, may predict slowly, overfit easily and suffer from the curse of dimensionality.

Q5: ***What is the difference between generative and discriminative models?***

- **A discriminative model learns $P(y\mid x)$ or the decision boundary directly; a generative model learns how the data are produced, i.e. the joint distribution $P(x,y)$, and then gets $P(y\mid x)$ via Bayes' rule.**
- Discriminative: logistic regression, SVM, CRF, most neural network classifiers. Generative: naive Bayes, Gaussian mixtures, HMMs, plus VAEs, GANs, diffusion models and language models.
- With plenty of data, discriminative models usually classify more accurately; with little data, generative models may converge faster (Ng & Jordan 2001).
- Generative models can generate samples, handle missing features and detect anomalies, but need stronger distributional assumptions.
- Classic pairs: naive Bayes ↔ logistic regression, HMM ↔ CRF.

Q6: ***What is the "No Free Lunch" theorem?***

- **The NFL theorem says that averaged over all possible problems, every learning algorithm performs the same; no algorithm is best on every problem. Whether an algorithm is good depends on whether its inductive bias matches the structure of the specific problem.**
- So model choice should rely on domain knowledge and data properties, compared with validation experiments.
- Real problems are not uniformly spread over "all possible functions"; they have regularities such as smoothness and hierarchy, which is why GBDT works well on tabular data and Transformers on sequences. This does not contradict NFL.

Q7: ***What is the difference between MLE and MAP? How do they relate to regularization?***

- **MLE only finds the parameters that make the data most likely; MAP additionally multiplies in a prior on the parameters, which amounts to adding a regularization term to the log-likelihood. A Gaussian prior corresponds to L2 regularization and a Laplace prior to L1.**
- The "narrower" the prior, the stronger the corresponding regularization.
- As data grow to infinity the likelihood dominates and MAP → MLE; with a uniform prior they are equal.
- MAP is still a point estimate; full Bayesian inference keeps the entire posterior.
- Key formula: $\hat\theta_{MAP}=\arg\max_\theta\left[\sum_i\log p(x_i\mid\theta)+\log p(\theta)\right]$

## Regularization

Q8: ***What is the difference between L1 and L2 regularization? Why does L1 produce sparse solutions?***

- **L1 penalizes the absolute value of weights and pushes many of them to exactly 0, giving a sparse solution that does feature selection; L2 penalizes squared weights and only shrinks them overall, almost never to 0.**
- Geometric view: the L1 constraint region is a diamond with corners on the axes, so the loss contours tend to touch it at a corner, setting some weights to 0; the L2 region is a circle and the tangent point is usually off the axes.
- Gradient view: the L2 gradient shrinks as the weight shrinks, so it never reaches 0; L1 keeps a constant-size "push" near 0 that drives small weights to exactly 0 (soft thresholding).
- Bayesian view: L1 corresponds to a Laplace prior with a sharp peak at 0, L2 to a smooth Gaussian prior.

| | L1 | L2 |
|---|---|---|
| Solution | sparse | dense, uniformly smaller |
| Correlated features | picks one arbitrarily | spreads weight evenly |
| Optimization | non-differentiable at 0; coordinate descent / proximal gradient | smooth, closed-form solution |

Q9: ***What is Elastic Net?***

- **Elastic Net adds both L1 and L2 penalties, keeping L1's sparsity while using L2 to fix two weaknesses of the Lasso.**
- Correlated features: Lasso picks one of them arbitrarily, while Elastic Net has a "grouping effect" and selects or drops them together.
- When there are more features than samples, Lasso can select at most $n$ features; Elastic Net has no such limit.
- It has two hyperparameters to tune (overall strength $\lambda$ and L1 ratio $\alpha$), usually by cross-validation.

## Model Evaluation and Data

Q10: ***What are the variants of cross-validation? When is each appropriate?***

- **The most common is k-fold cross-validation (each fold takes a turn as validation while the rest train, then average); then pick a variant for the data: stratified k-fold for class imbalance, group k-fold when one entity has many samples, time-ordered splits for time series, and nested CV when you both tune and evaluate.**
- k is usually 5 or 10; $k=n$ is leave-one-out, low bias but high variance and cost.
- Group k-fold keeps each user/patient on only one side of the split, preventing leakage.
- Never shuffle time series: training data must come strictly before validation data, with a gap in between if needed.
- Nested CV tunes in the inner loop and evaluates in the outer loop, avoiding the optimistic bias of choosing hyperparameters and reporting performance on the same data.

Q11: ***What is data leakage? What are its common forms?***

- **Data leakage is using information during training that will not be available at prediction time, which inflates offline metrics and leads to poor production performance.**
- Target leakage: features derived from the label or produced after it (e.g. using "account closure date" to predict churn).
- Preprocessing leakage: scaling, imputing, feature selection or oversampling on the full data before splitting; target encoding without out-of-fold.
- Temporal leakage and duplicate entities: randomly splitting time series; the same user or near-duplicate samples in both train and test; for large models, test sets leaking into pretraining corpora.
- Prevention: split first, then preprocess, and put preprocessing inside cross-validation with a Pipeline; split by time/group; be suspicious of features that look too good.

Q12: ***How do you handle class imbalance?***

- **First change the metric (not accuracy, but PR-AUC, F1, per-class recall), then act at the data, algorithm and decision levels; the simplest and often most effective step is adjusting the classification threshold.**
- Data level: undersample the majority class, oversample the minority, SMOTE interpolation; only inside the training folds, otherwise it leaks.
- Algorithm level: class weights / cost-sensitive learning, Focal Loss (down-weights easy examples), balanced ensemble methods.
- Decision level: choose the threshold on the PR curve based on business costs.
- Resampling or weighting distorts predicted probabilities, so recalibrate if you need probabilities; with extreme imbalance, treat it as anomaly detection.

Q13: ***Which models need feature scaling, and why?***

- **Models that rely on distances, gradient descent, regularization penalties or variance need scaling, because large-scale features dominate the result or slow down optimization; tree models do not, because splits only depend on the ordering of feature values.**
- Distance-based: KNN, K-means, RBF-kernel SVM.
- Gradient descent: linear/logistic regression, neural networks (uneven scales make the loss surface elongated and slow to converge).
- Regularization: Ridge/Lasso penalize all coefficients equally, which is unfair under different scales; in PCA, high-variance features dominate the components.
- Methods: standardization (most common), min-max (sensitive to outliers), robust scaling (median and IQR), log transforms; fit scaling parameters on the training set only.

Q14: ***How do you handle missing values?***

- **First identify the missingness mechanism, then pick a method: data missing completely at random can be dropped or simply imputed; missingness that depends on other variables can be imputed by modeling with other features; missingness that depends on the missing value itself (e.g. high earners not reporting income) is hardest, and "is missing" is itself information, so add a missing indicator.**
- Deletion: drop rows when little is missing, drop columns when most is missing.
- Simple imputation: mean/median, mode or an "Unknown" category; it underestimates variance.
- Model-based imputation: KNN imputation, iterative regression imputation (MICE); for time series, forward fill or interpolation.
- XGBoost/LightGBM learn which way missing values should go at each split; compute imputation statistics on the training set only.

Q15: ***What is model calibration? How do you calibrate a model?***

- **Calibration means predicted probabilities match real frequencies, e.g. about 80% of samples predicted at 0.8 are actually positive. Any setting that makes decisions from probabilities (risk control, ad bidding, model blending) needs it. The approach is to learn a monotonic function mapping scores to probabilities on a separate calibration set.**
- Evaluation: reliability diagrams, ECE (expected calibration error), Brier score, log loss.
- Logistic regression is usually well calibrated; naive Bayes pushes toward 0/1; random forests stay away from 0/1; deep networks are often overconfident.
- Methods: Platt scaling (fit a logistic regression, good with little data), isotonic regression (more flexible but needs more data), temperature scaling (learns a single temperature for neural networks and does not change predicted classes).
- Calibration is a monotonic transform, so it does not change AUC.

## KNN

Q16: ***How does KNN work? What distance metrics are commonly used?***

- **For a new sample, KNN finds the K nearest neighbors in the training set and takes a majority vote for classification or an average for regression (optionally distance-weighted). It has no training phase and computes only at prediction time, so it is a lazy learner.**
- Euclidean distance is the most common; Manhattan distance is more robust to outliers and discriminates better in high dimensions; both are special cases of the Minkowski distance.
- Cosine distance: for text and embeddings, looks only at direction, not magnitude.
- Mahalanobis distance accounts for feature correlation and scale; Hamming distance is for binary/categorical features, Jaccard for sets.
- Features must be scaled first, otherwise large-scale features dominate the distance.

Q17: ***How do you choose K in KNN?***

- **A small K makes the model noise-sensitive and overfit (low bias, high variance); a large K makes the boundary too smooth and underfit (high bias, low variance). In practice, use cross-validation over a range of K and pick the one with the lowest validation error.**
- A rule-of-thumb starting point is $K\approx\sqrt n$.
- Use an odd K for binary classification to avoid ties; with class imbalance, use distance-weighted voting.
- The asymptotic error of 1-NN is at most twice the Bayes error (Cover & Hart).

Q18: ***What is the curse of dimensionality? Why does it affect KNN?***

- **The curse of dimensionality means that as dimensions grow, data become extremely sparse and distances between all points become nearly equal, so "nearest neighbor" loses its meaning and KNN predictions degrade noticeably.**
- The number of samples needed to keep the same sampling density grows exponentially with dimension.
- Neighborhoods stop being local: in 10 dimensions, covering 10% of the data requires about 80% of the range in each dimension.
- Irrelevant features badly distort distances.
- Remedies: dimensionality reduction, feature selection, metric learning, KNN in a learned embedding space; real data usually lie on a low-dimensional manifold, which is why KNN still works in embedding spaces.

Q19: ***What is a KD-tree? How can KNN search be accelerated?***

- **A KD-tree is a binary space-partitioning tree: each level picks one dimension and splits the data in half at the median. Search first descends to the query's leaf to get a candidate, then backtracks, exploring the other side only if the ball centered at the query with radius equal to the current best distance intersects that region; otherwise it prunes. In low dimensions the average cost is close to $O(\log n)$.**
- In higher dimensions (roughly above 20) pruning stops working and it degrades to near brute force.
- Ball trees partition with hyperspheres and do better in higher dimensions and with non-Euclidean distances.
- Large-scale high-dimensional vectors use approximate nearest neighbor (ANN) methods: LSH, HNSW, IVF + product quantization (e.g. FAISS), which also underpin vector databases and RAG retrieval.

## Ensemble Learning

Q20: ***What is the difference between bagging, boosting and stacking?***

- **Bagging trains many independent strong models in parallel and averages them, mainly reducing variance (e.g. random forest); boosting trains weak models sequentially, each fixing the previous one's errors, mainly reducing bias (e.g. GBDT, XGBoost); stacking uses a meta-model to combine the predictions of several heterogeneous models.**

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| Training | parallel, independent | sequential, error-correcting | base models + meta-model |
| Mainly reduces | variance | bias | both |
| Base learners | strong (deep trees) | weak (shallow trees) | heterogeneous models |
| Examples | random forest | AdaBoost, GBDT, XGBoost | Kaggle blends |

- The benefit of averaging is limited by correlation between models; random forests reduce it by random feature subsampling.
- AdaBoost up-weights misclassified samples; GBDT fits the negative gradient of the loss each round; boosting is noise-sensitive and needs a learning rate and early stopping.
- Stacking must train the meta-model on out-of-fold predictions, otherwise it leaks; the meta-model is usually a simple logistic regression. Blending is a simplified version that uses a holdout set instead of K folds.
