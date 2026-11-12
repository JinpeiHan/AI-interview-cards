# Machine Learning Fundamentals

## Learning Paradigms and Theory

Q1: ***What is the difference between supervised, unsupervised and self-supervised learning?***

- **Supervised learning**: training data has labels $(x,y)$; learn a mapping $f:x\to y$; classification and regression. The cost is expensive labeling.
- **Unsupervised learning**: only $x$ is available; discover structure in the data; clustering (K-means), dimensionality reduction (PCA), density estimation, anomaly detection.
- **Self-supervised learning**: construct "pseudo-labels" from the data itself and train in a supervised way, with no human annotation needed:
    - Masked prediction: BERT's MLM, MAE's masked image reconstruction;
    - Next-token prediction: pre-training of GPT-style language models;
    - Contrastive learning: SimCLR, MoCo, CLIP (pull together different views of the same sample / matching image-text pairs, push apart different samples).
- Others: **semi-supervised** (a few labels + lots of unlabeled data, e.g. pseudo-labeling, consistency regularization), **reinforcement learning** (reward signals from interacting with an environment).
- Follow-up: the modern large-model paradigm is usually "self-supervised pre-training + supervised fine-tuning + preference alignment (RLHF/DPO)".

Q2: ***Derive the bias-variance decomposition***

- Let $y=f(x)+\epsilon$ with $E[\epsilon]=0$ and $\mathrm{Var}(\epsilon)=\sigma^2$; $\hat f$ is the model learned on a random training set $D$. The expected squared error at a point $x$:
  $$E_{D,\epsilon}\left[(y-\hat f(x))^2\right]=\underbrace{\left(E_D[\hat f(x)]-f(x)\right)^2}_{\text{Bias}^2}+\underbrace{E_D\left[(\hat f(x)-E_D[\hat f(x)])^2\right]}_{\text{Variance}}+\underbrace{\sigma^2}_{\text{irreducible error}}$$
- Derivation sketch: let $\bar f=E_D[\hat f(x)]$ and expand $(f+\epsilon-\hat f)^2=((f-\bar f)+(\bar f-\hat f)+\epsilon)^2$. Since $\epsilon$ is independent of $D$ with zero mean, and $E_D[\bar f-\hat f]=0$, all cross terms vanish.
- Meaning:
    - **Bias**: the model's assumptions are too simple and it systematically deviates from the true function → underfitting;
    - **Variance**: the model is too sensitive to the randomness of the training set → overfitting;
    - Increasing model complexity usually lowers bias and raises variance -- a trade-off.
- Follow-up: the **double descent** phenomenon in deep learning -- the test error of over-parameterized models falls again beyond the interpolation threshold, showing that the classic U-shaped curve does not always hold and that implicit regularization (SGD, architecture) plays a role.

Q3: ***How do you diagnose underfitting and overfitting with learning curves?***

- **Learning curve**: x-axis is the number of training samples (or training epochs), y-axis is training error and validation error.
- **High bias (underfitting)**: both training and validation error are high and quickly converge to similar values → more data **does not help**; increase model complexity, add features, reduce regularization, train longer.
- **High variance (overfitting)**: training error is very low and validation error is clearly higher, with a large **gap** that shrinks as the sample size grows → more data **helps**; also strengthen regularization, reduce features, simplify the model, use data augmentation, early stopping.
- **Validation curve**: x-axis is a hyperparameter (e.g. tree depth, $\lambda$); find the point with the lowest validation error.
- Also compare against a **baseline error** (human-level / Bayes error): the gap between training error and baseline ≈ avoidable bias, and the gap between validation and training error ≈ variance.

Q4: ***What is the difference between parametric and non-parametric models?***

- **Parametric models**: a fixed number of parameters independent of sample size, with strong assumptions about the functional form; e.g. linear regression, logistic regression, naive Bayes, neural networks with a fixed architecture.
    - Pros: fast training/prediction, need less data, interpretable; cons: large bias when the assumptions are wrong.
- **Non-parametric models**: model complexity grows with the amount of data, few assumptions about the functional form; e.g. KNN, decision trees, kernel SVMs (the number of support vectors grows with the data), Gaussian processes, kernel density estimation.
    - Pros: flexible, can fit complex functions; cons: need more data, prediction can be slow (KNN stores all the data), prone to overfitting, heavily affected by the curse of dimensionality.
- Note: "non-parametric" does not mean "no parameters" but that the number of parameters is not fixed.

Q5: ***What is the difference between generative and discriminative models?***

- **Discriminative models**: directly model $P(y\mid x)$ or the decision boundary; logistic regression, SVM, CRF, most neural network classifiers.
- **Generative models**: model the joint distribution $P(x,y)=P(x\mid y)P(y)$ and obtain $P(y\mid x)$ via Bayes' rule; naive Bayes, LDA (linear discriminant analysis), Gaussian mixture models, HMMs; more broadly also VAEs, GANs, diffusion models, autoregressive language models (modeling $P(x)$).
- Comparison:
    - Classification accuracy: with ample data, discriminative models are usually better (they solve only the problem needed);
    - With little data, generative models may converge faster (Ng & Jordan 2001 comparing naive Bayes and logistic regression);
    - Generative models can handle missing features, generate samples and do anomaly detection, but require stronger distributional assumptions.
- Classic pairs: naive Bayes ↔ logistic regression; HMM ↔ CRF.

Q6: ***What is the "No Free Lunch" theorem?***

- **NFL theorem** (Wolpert 1996): averaged over all possible problems (target functions uniformly distributed), every learning algorithm has the same expected generalization performance -- no algorithm beats all others on every problem.
- Implications:
    - How good an algorithm is depends on whether its **inductive bias** matches the true structure of the problem (e.g. the translation equivariance of CNNs, the linearity assumption of linear models);
    - Models must be chosen based on domain knowledge and data characteristics and compared through validation experiments.
- Follow-up: real-world problems are not uniformly distributed over "all possible functions"; they have regularities like smoothness and hierarchical structure, so some methods (GBDT on tabular data, Transformers on sequences) generally perform well in practice -- this does not contradict NFL.

Q7: ***What is the difference between MLE and MAP? How do they relate to regularization?***

- **MLE**: $\hat\theta_{MLE}=\arg\max_\theta \sum_i\log p(x_i\mid\theta)$, depends only on the data.
- **MAP**: $\hat\theta_{MAP}=\arg\max_\theta\left[\sum_i\log p(x_i\mid\theta)+\log p(\theta)\right]$, adds a prior, which amounts to adding a **regularization term** to the log-likelihood.
- Correspondence with regularization (for linear regression, where MLE under Gaussian noise ⇔ least squares):
    - Gaussian prior $\theta\sim\mathcal N(0,\tau^2)$ ⇒ $-\log p(\theta)\propto\frac{1}{2\tau^2}\|\theta\|_2^2$ ⇒ **L2 / ridge regression**;
    - Laplace prior $p(\theta)\propto e^{-|\theta|/b}$ ⇒ $-\log p(\theta)\propto\frac1b\|\theta\|_1$ ⇒ **L1 / Lasso**;
    - Regularization strength $\lambda$ is inversely proportional to the prior variance: the "narrower" the prior, the stronger the regularization.
- As data → ∞ the likelihood dominates and MAP → MLE; with a uniform prior, MAP = MLE.
- MAP is still a **point estimate**; full Bayesian inference keeps the entire posterior distribution.

## Regularization

Q8: ***What is the difference between L1 and L2 regularization? Why does L1 produce sparse solutions?***

- Objective: $\min_w L(w)+\lambda\|w\|_1$ (L1/Lasso) or $\min_w L(w)+\lambda\|w\|_2^2$ (L2/Ridge).
- **Geometric view**: equivalent to the constrained problem $\min L(w)$ s.t. $\|w\|_1\le t$ or $\|w\|_2\le t$. The L1 constraint region is a diamond (a "polytope" in high dimensions) with corners on the coordinate axes; the loss contours are more likely to first touch it at a **corner**, making some $w_j=0$. The L2 region is a circle, and the tangent point is generally not on an axis.
- **Gradient/subgradient view**: the L2 gradient $2\lambda w$ goes to 0 as $w\to0$, so it shrinks weights but does not reach 0; the L1 subgradient is a constant $\lambda\,\mathrm{sign}(w)$, so it keeps a constant "push" near 0. Under an orthogonal design the solution is **soft thresholding**: $w_j=\mathrm{sign}(z_j)\max(|z_j|-\lambda,0)$, whereas L2 gives proportional shrinkage $w_j=z_j/(1+\lambda)$.
- **Bayesian view**: L1 ↔ Laplace prior (a sharp peak at 0), L2 ↔ Gaussian prior (flat near 0).
- Other comparisons:

| | L1 | L2 |
|---|---|---|
| Solution | Sparse, performs feature selection | Dense, all weights shrink |
| Correlated features | Picks one arbitrarily | Spreads weight evenly |
| Optimization | Non-differentiable at 0; needs coordinate descent / proximal gradient | Smooth, closed form $(X^TX+\lambda I)^{-1}X^Ty$ |
| Robustness | Unstable feature selection | More stable, mitigates collinearity |

Q9: ***What is Elastic Net?***

- Combines L1 and L2: $\min_w L(w)+\lambda\left[\alpha\|w\|_1+\frac{1-\alpha}{2}\|w\|_2^2\right]$ (`l1_ratio` in sklearn is $\alpha$).
- Fixes two problems of Lasso:
    - **Groups of correlated features**: Lasso picks one arbitrarily, while Elastic Net, thanks to its L2 part, has a "grouping effect" and selects or drops correlated features together;
    - **$p>n$**: Lasso selects at most $n$ features; Elastic Net has no such limit.
- It keeps sparsity. Two hyperparameters ($\lambda$, $\alpha$) must be tuned, usually by cross-validated grid search.

## Model Evaluation and Data

Q10: ***What are the variants of cross-validation? When is each appropriate?***

- **k-fold cross-validation**: split the data into $k$ parts; in turn use 1 part for validation and $k-1$ for training; average the $k$ results; commonly $k=5$ or 10. Larger $k$ means lower bias but higher variance and cost; $k=n$ is **leave-one-out (LOOCV)**.
- **Stratified k-fold**: each fold keeps the same class proportions as the whole dataset; used for classification, especially with **class imbalance**.
- **Group k-fold**: samples from the same group (same user, patient, session) appear only in training or only in validation, preventing "same entity" leakage that inflates scores.
- **Time-series cross-validation**: strictly chronological; the training set only contains data before the validation set (expanding or sliding window, `TimeSeriesSplit`); leave a **gap/embargo** between training and validation if needed to avoid leakage from overlapping labels. Never shuffle.
- **Nested cross-validation**: the inner loop tunes hyperparameters and the outer loop evaluates, avoiding the optimistic bias of both selecting hyperparameters and reporting performance on the same validation data.
- Repeated k-fold: average over several random splits to reduce variance due to the split.

Q11: ***What is data leakage? What are its common forms?***

- **Data leakage**: training uses information that would not be available at prediction time, leading to inflated offline metrics and poor online performance.
- Common forms:
    - **Target leakage**: features contain information derived from the label or produced after it (e.g. using "account closure date" to predict churn, or "medication records" to predict a disease);
    - **Train/test contamination**: standardizing, imputing missing values, selecting features, running PCA or oversampling (SMOTE) on the full dataset **before** splitting;
    - **Temporal leakage**: randomly splitting time-series data, using the future to predict the past;
    - **Duplicate entities**: the same user/patient/near-duplicate samples appear in both training and test sets;
    - Target encoding not done out-of-fold;
    - In the LLM era: test sets included in the pre-training corpus (**benchmark contamination**).
- Prevention: split first, then preprocess; put preprocessing inside cross-validation with a `Pipeline`; split by time/group; check when each feature is generated; be suspicious of features that are "too good to be true".

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
pipe = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(pipe, X, y, cv=5)  # scaler is fit inside each training fold
```

Q12: ***How do you handle class imbalance?***

- **Evaluation metrics**: do not use accuracy; use Precision/Recall/F1, PR-AUC (more informative than ROC-AUC when positives are rare), per-class recall.
- **Data level**:
    - Random undersampling of the majority class (loses information), random oversampling of the minority class (prone to overfitting);
    - **SMOTE**: synthesize new samples by interpolating between minority samples and their neighbors; variants Borderline-SMOTE, ADASYN;
    - Resampling must only be done on the **training set** (inside cross-validation folds), otherwise it leaks.
- **Algorithm level**:
    - **Class weights / cost-sensitive learning**: give the minority class a larger weight in the loss (`class_weight='balanced'`);
    - **Focal Loss**: $-(1-p_t)^\gamma\log p_t$, down-weights easy examples;
    - Ensemble methods: EasyEnsemble, BalancedRandomForest.
- **Decision level**: **adjust the classification threshold** (often the simplest and most effective), picking the operating point on the PR curve according to business costs.
- Note: resampling or weighting changes the distribution of predicted probabilities; if calibrated probabilities are needed, recalibrate afterwards or correct for the prior.
- Under extreme imbalance, consider reframing it as **anomaly detection**.

Q13: ***Which models need feature scaling, and why?***

- **Need scaling**:
    - **Distance**-based: KNN, K-means, SVM (RBF kernel) -- features with large magnitudes dominate the distance;
    - Optimized by **gradient descent**: linear/logistic regression, neural networks -- different feature scales make the loss surface an elongated ellipse, slowing convergence;
    - **Regularized** models: Ridge/Lasso -- the penalty treats all coefficients equally, so different scales make the penalty unfair;
    - **Variance**-based: PCA -- high-variance features dominate the principal components.
- **Do not need scaling**: **tree models** (decision trees, random forests, GBDT/XGBoost/LightGBM) -- splits depend only on the ordering of feature values, and monotonic transforms do not change the result; naive Bayes usually does not either.
- Methods:
    - **Standardization** (Z-score): $x'=\frac{x-\mu}{\sigma}$, the most common;
    - **Normalization** (Min-Max): $x'=\frac{x-x_{min}}{x_{max}-x_{min}}$, sensitive to outliers;
    - **Robust scaling**: uses the median and IQR, resistant to outliers;
    - Log transform for skewed distributions.
- Scaling parameters must be fit on the training set only, then applied to the validation/test sets.

Q14: ***How do you handle missing values?***

- First analyze the **missingness mechanism**:
    - **MCAR** (missing completely at random): missingness is unrelated to any variable; deletion introduces no bias;
    - **MAR** (missing at random): missingness depends on other observed variables; impute by modeling from other features;
    - **MNAR** (missing not at random): missingness depends on the missing value itself (e.g. high earners not reporting income); hardest to handle, and "being missing" is itself information.
- Methods:
    - **Deletion**: drop rows (few missing and MCAR) or drop columns (very high missing ratio);
    - **Simple imputation**: mean/median (numeric), mode or an "Unknown" category (categorical); underestimates variance;
    - **Add a missing-indicator variable**: used together with imputation to keep the "missing" signal; especially useful for MNAR;
    - **Model-based imputation**: KNN imputation, iterative regression imputation (MICE / `IterativeImputer`);
    - Time series: forward fill, interpolation;
    - **Native model support**: XGBoost/LightGBM learn a default direction for missing values at each split, no imputation needed.
- Imputation statistics must likewise be computed on the training set only.

Q15: ***What is model calibration? How do you calibrate a model?***

- **Calibration**: predicted probabilities match observed frequencies, i.e. among samples predicted at 0.8, about 80% are actually positive. Crucial wherever probabilities are used (risk decisions, expected value calculations, ad bidding, model blending).
- Evaluation: **reliability diagram** (binned predicted probability vs. actual positive rate), **ECE** (expected calibration error) $=\sum_b\frac{|B_b|}{n}|\text{acc}(B_b)-\text{conf}(B_b)|$, Brier score, log loss.
- Calibration behavior of different models:
    - Logistic regression is usually well calibrated (it directly optimizes log loss);
    - Naive Bayes tends to push probabilities toward 0/1 (independence assumption);
    - Random forests tend to stay away from 0/1; SVMs output margins, not probabilities;
    - Modern deep neural networks are often **overconfident** (Guo et al. 2017).
- Calibration methods (fit on a separate calibration set):
    - **Platt scaling**: fit a logistic regression $\sigma(As+B)$ on the model scores; good for small data and S-shaped distortions;
    - **Isotonic regression**: a non-parametric monotonic function; more flexible but needs more data and can overfit;
    - **Temperature scaling**: $\text{softmax}(z/T)$ for neural networks; learns a single parameter $T$ and does not change the predicted class (accuracy unchanged).
- Note: AUC only measures ranking, so calibration does not change AUC (a monotonic transform).

## KNN

Q16: ***How does KNN work? What distance metrics are commonly used?***

- **Principle**: for a new sample, find the $K$ nearest neighbors in the training set; for classification take a **majority vote**, for regression take the **average** (optionally weighted by inverse distance). It is **lazy learning**: no training phase, all computation happens at prediction time, with prediction complexity $O(nd)$.
- Distance metrics:
    - **Minkowski distance**: $d(x,y)=\left(\sum_i|x_i-y_i|^p\right)^{1/p}$;
    - $p=2$ **Euclidean distance** (most common); $p=1$ **Manhattan distance** (more robust to outliers, more discriminative than Euclidean in high dimensions); $p\to\infty$ **Chebyshev distance** $\max_i|x_i-y_i|$;
    - **Cosine distance** $1-\frac{x\cdot y}{\|x\|\|y\|}$: text and embedding vectors; focuses on direction rather than magnitude;
    - **Mahalanobis distance** $\sqrt{(x-y)^T\Sigma^{-1}(x-y)}$: accounts for feature correlation and scale;
    - **Hamming distance**: categorical/binary features; **Jaccard distance**: sets.
- **Feature scaling** is required first, otherwise large-scale features dominate the distance.

Q17: ***How do you choose K in KNN?***

- **K too small** (e.g. $K=1$): complex decision boundary, sensitive to noise, **low bias, high variance**, overfitting;
- **K too large**: overly smooth decision boundary; in the extreme $K=n$ it always predicts the majority class, **high bias, low variance**, underfitting.
- Selection: **cross-validation** over a range of K values, picking the one with the lowest validation error; a rule-of-thumb starting point is $K\approx\sqrt n$.
- For binary classification use an **odd** K to avoid ties; with class imbalance, use distance-weighted voting.
- The asymptotic error rate of 1-NN is at most twice the Bayes error rate (Cover & Hart 1967).

Q18: ***What is the curse of dimensionality? Why does it affect KNN?***

- In high-dimensional spaces:
    - **Data become sparse**: the number of samples needed to keep the same sampling density grows exponentially with dimension;
    - **Distance concentration**: the ratio between the farthest and nearest distances among random points tends to 1, $\frac{d_{max}-d_{min}}{d_{min}}\to0$, so "nearest neighbor" becomes meaningless;
    - **"Neighborhoods" are no longer local**: in a unit hypercube, covering 10% of the data requires an edge length of $0.1^{1/d}$ per dimension -- about 0.8 for $d=10$, almost the entire range;
    - The volume of a hypersphere concentrates **near its surface**.
- Effect on KNN: neighbors are no longer "close", prediction quality drops, and irrelevant features severely distort distances.
- Mitigation: **dimensionality reduction** (PCA, autoencoders), feature selection, metric learning, learned embedding spaces; real data usually lie on a low-dimensional **manifold**, which is why KNN still works in embedding spaces.

Q19: ***What is a KD-tree? How can KNN search be accelerated?***

- **KD-tree**: a binary space-partitioning tree. At each level choose a dimension (usually the one with the largest variance, or cycling through dimensions), split the data in half at the **median**, and recurse; construction is $O(n\log n)$.
- **Search**:
    1. Descend from the root to the leaf containing the query point, taking it as the current nearest-neighbor candidate;
    2. Backtrack: if the hypersphere centered at the query with radius equal to the current best distance **intersects** the region of the other child node, search that subtree; otherwise **prune** it;
    3. In low dimensions the average complexity is about $O(\log n)$.
- Limitation: in higher dimensions (empirically around $d>20$) pruning fails and search degenerates to nearly brute force.
- Other methods: **Ball Tree** (partitions with hyperspheres; better in high dimensions and for non-Euclidean metrics); for large-scale / high-dimensional vectors use **approximate nearest neighbor (ANN)** search: LSH (locality-sensitive hashing), HNSW (hierarchical navigable small world graphs), IVF + product quantization (PQ, e.g. FAISS) -- which is also the foundation of vector databases and RAG retrieval.

## Ensemble Learning

Q20: ***What is the difference between bagging, boosting and stacking?***

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| Training | **Parallel**, base learners independent | **Sequential**, each corrects the previous one's errors | Two levels: base learners + meta-learner |
| Samples | Bootstrap resampling | Full data, reweight samples or fit residuals/negative gradients | Out-of-fold predictions of base learners as meta-features |
| Mainly reduces | **Variance** | **Bias** (can also reduce variance) | Both; combines strengths of heterogeneous models |
| Base learners | Strong (low bias, high variance, e.g. deep trees) | Weak (e.g. shallow trees) | Usually heterogeneous (GBDT, NN, LR, etc.) |
| Examples | Random forest | AdaBoost, GBDT, XGBoost, LightGBM, CatBoost | Common Kaggle blending setups |
| Overfitting | Resistant to overfitting | Sensitive to noise; needs learning rate / early stopping | Meta-learner must use OOF predictions, otherwise leakage |

- Why bagging reduces variance: averaging $B$ models with variance $\sigma^2$ and pairwise correlation $\rho$ gives variance $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$; random forests lower $\rho$ through **random feature subsampling**.
- Boosting: AdaBoost increases the weights of misclassified samples (equivalent to forward stagewise additive modeling with exponential loss); GBDT fits the **negative gradient** of the loss at each round.
- Stacking: the meta-learner must be trained on K-fold **out-of-fold** predictions; it is usually a simple model (e.g. logistic regression) to avoid overfitting. **Blending** is a simplified version that uses a single holdout set instead of K folds.
