# SVM

Q1.***What is SVM***?

- **An SVM (Support Vector Machine) is a supervised learning method whose core idea is to find a hyperplane in feature space that separates the two classes while making the margin between them as large as possible.**
- It only cares about the hardest-to-separate points closest to the boundary (the support vectors), not about all samples.

Q2:***What is a hyperplane in SVM***?

- **The hyperplane is the SVM's decision boundary: points on either side of it are assigned to different classes. It has one dimension fewer than the feature space: with 2 features it is a line, with 3 features it is a plane.**
- Many hyperplanes can separate the data; SVM picks the one with the largest margin.

![https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.png](https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.png)

Q3:***What are support vectors in SVM***?

- **Support vectors are the samples closest to the hyperplane, lying exactly on the margin boundary (or inside the margin). They determine where the hyperplane is: remove them and the hyperplane moves; remove any other point and it stays the same.**
- Prediction also uses only the support vectors, so SVM is a sparse model.

![https://miro.medium.com/max/1400/0*ecA4Ls8kBYSM5nza.jpg](https://miro.medium.com/max/1400/0*ecA4Ls8kBYSM5nza.jpg)

Q4:***What are hard-margin and soft-margin SVMs***?

- **A hard-margin SVM requires every point to be correctly separated and stay outside the margin, so it only works on linearly separable data; a soft-margin SVM lets a few points fall inside the margin or even be misclassified, at a penalty, so it works on noisy, not perfectly separable data.**

Q5:***What types of SVM kernels do you know***?

- **The common ones are the linear kernel, the polynomial kernel and the Gaussian (RBF) kernel; there is also the sigmoid kernel.**

Q6:***Why use the kernel trick***?

- **When data is not linearly separable, it has to be mapped into a higher-dimensional space to become separable, but computing there directly is too expensive. The kernel trick computes the high-dimensional inner product directly in the original low-dimensional space, giving the high-dimensional classification effect without ever doing the mapping.**
- The SVM dual depends only on inner products between samples, so the inner product can simply be replaced by a kernel function.

Q7:***What are the applications of SVM***?

- **SVM is mainly used for classification. Typical applications are face detection, text classification, image classification, bioinformatics (protein and cancer classification) and handwriting recognition; it suits tasks with few samples and many dimensions especially well.**

Q8:***Name some advantages of SVM***?

- **SVM is a convex optimization problem, so it is guaranteed to reach the global optimum; the kernel trick lets it handle non-linear problems; and because it depends only on support vectors, it generalizes well on small, high-dimensional datasets.**
- The soft margin makes it work on both linearly separable and non-separable data.

Q9:***For an `N`-dimensional dataset, what is the minimum possible number of support vectors? (Assume the data is perfectly linearly separable)***

- **At least 2 (one per class), regardless of the dimension or the number of samples.**

Q10:***What happens when there is no clear hyperplane in SVM?***

- **If no hyperplane can separate the data in the original space, the data is mapped into a higher-dimensional space and a linear hyperplane is found there; this is done through a kernel function and is called kernelization.**

![https://66.media.tumblr.com/9bffea56372d28d2a30f80557451e824/tumblr_inline_o9aabehtqP1u37g00_540.png](https://66.media.tumblr.com/9bffea56372d28d2a30f80557451e824/tumblr_inline_o9aabehtqP1u37g00_540.png)

Q11:***What is the hinge loss in SVM?***

- **The hinge loss is the loss used by the soft-margin SVM: when a sample is correctly classified and far enough from the boundary (functional margin ≥ 1) the loss is 0; otherwise the loss grows linearly with the distance. So even a correctly classified point is penalized if it is too close to the boundary.**
- The soft-margin SVM is equivalent to "hinge loss + L2 regularization": the regularizer makes $\|w\|$ small, i.e. the margin large.
- Key formula: $\max(0,\ 1-y_i f(x_i))$

![https://app.yinxiang.com/shard/s24/res/8d14cae8-d22f-499a-aeb8-7a8e535fc62b.gif](https://app.yinxiang.com/shard/s24/res/8d14cae8-d22f-499a-aeb8-7a8e535fc62b.gif)

![https://miro.medium.com/max/575/1*PGqpYm7o5GCbDXxXErr2JA.png](https://miro.medium.com/max/575/1*PGqpYm7o5GCbDXxXErr2JA.png)

Reference: [https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/](https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/)

Q12:***The role of the hyperparameter C in SVM***

- **C is the penalty coefficient for misclassified samples in the soft-margin SVM; it trades off "margin as wide as possible" against "as few errors as possible". A larger C tolerates fewer errors, a smaller C tolerates more.**

Q13: ***Describe the hyperparameters commonly used when designing an SVM classifier***

- **The most important ones are the penalty coefficient C and the RBF kernel's gamma, plus the kernel type (and the degree for a polynomial kernel).**
- C: larger means less tolerance for errors and a risk of overfitting; smaller risks underfitting.
- gamma: sets how far a single sample's influence reaches; larger means a smaller reach, a more curved boundary, usually more support vectors, and a risk of overfitting; smaller gives a smoother boundary.

Q14:***Compare the KNN algorithm with the SVM algorithm***

- **KNN has no training step and computes distances to all training samples at prediction time; SVM must be trained to find a hyperplane and uses only the support vectors at prediction time.**
- KNN is a non-sparse model with slow prediction and large storage; SVM is a sparse model.
- KNN essentially has only the parameter k; SVM requires tuning C, the kernel and its parameters.

Q15:***What is the difference between using SVM for classification and for regression?***

- **SVM classification makes the points closest to the hyperplane as far away as possible, i.e. separates the two classes as widely as possible; SVR regression tries to fit all samples inside a "tube" around the hyperplane, so that their deviation from the hyperplane is as small as possible.**

Reference: [https://www.cnblogs.com/belter/p/8975606.html](https://www.cnblogs.com/belter/p/8975606.html)

Reference: https://www.quora.com/What-is-the-main-difference-between-a-SVM-and-SVR

Q16:***What is a convex hull?***

- **The convex hull of a set of points is the smallest convex polygon (one with no inward-bending corners) that encloses all of them.**
- Link to SVM: for linearly separable data, the maximum-margin hyperplane is the perpendicular bisector of the segment joining the closest points of the two classes' convex hulls. In the figure below, the left is a convex polygon and the right is non-convex.

![https://miro.medium.com/max/560/1*xuqyWJ49_oNxrXGE6TehVw.png](https://miro.medium.com/max/560/1*xuqyWJ49_oNxrXGE6TehVw.png)

Q17:***How can SVM be used for anomaly detection?***

- **Use a One-class SVM: train only on normal data and learn a boundary (a hyperplane or hypersphere) that encloses it; a new sample inside the boundary is normal, one outside is an anomaly.**

Q18: ***Compare SVM and logistic regression***

- **Both are linear classifiers; the main difference is the loss: LR uses cross-entropy (log loss) and every sample contributes, while SVM uses the hinge loss and is determined only by the support vectors near the boundary.**
- Outliers: SVM only looks at points near the boundary, so it is insensitive to distant outliers; in LR every point contributes, so it is more easily affected.
- Non-linearity: SVM relies on the kernel trick; LR relies mainly on feature engineering (feature crosses, discretization).
- Efficiency: on massive data LR is faster; kernel SVM is very slow.

Q19:***Explain what the kernel trick is***

- **The kernel trick means finding a kernel function $K$ whose value, computed in the original low-dimensional space, equals the inner product after mapping to the high-dimensional space. This gives the high-dimensional classification effect without explicitly computing the mapping $\phi(x)$.**
- Only inner products are needed because the SVM dual problem and decision function contain only inner products between samples.
- Key formula: $K(x,y)=\langle\phi(x),\phi(y)\rangle$

![https://app.yinxiang.com/shard/s24/res/f3d68831-c5ee-4e9c-8737-49753082af57.gif](https://app.yinxiang.com/shard/s24/res/f3d68831-c5ee-4e9c-8737-49753082af57.gif)

![https://app.yinxiang.com/shard/s24/res/d8e584b8-ff97-4155-8df7-b0e131e2f32a.gif](https://app.yinxiang.com/shard/s24/res/d8e584b8-ff97-4155-8df7-b0e131e2f32a.gif)

![https://app.yinxiang.com/shard/s24/res/44a1a853-026a-40fe-8752-da1d7302a9de.gif](https://app.yinxiang.com/shard/s24/res/44a1a853-026a-40fe-8752-da1d7302a9de.gif)

![https://app.yinxiang.com/shard/s24/res/8be1a479-ee3c-45c8-aac4-aa5f44f02ed6.gif](https://app.yinxiang.com/shard/s24/res/8be1a479-ee3c-45c8-aac4-aa5f44f02ed6.gif)

Reference: [https://www.fanyeong.com/2017/11/13/the-kernel-trick/](https://www.fanyeong.com/2017/11/13/the-kernel-trick/)

Q20:***Compare random forest and SVM: in what situations would you prefer random forest (or SVM)?***

- **Prefer SVM for small, high-dimensional data with few outliers; prefer random forest when there is more data, mixed feature types or missing values, since it is usually more robust and needs less tuning.**
- SVM has no mechanism for missing values while decision trees do; SVM depends on distances in feature space, so missing features hurt it a lot.

Q21:***Compare SVM and deep learning***

- **SVM finds a maximum-margin boundary on fixed features: the optimization is convex, has a global optimum and the theory is clear; deep learning stacks layers to learn hierarchical feature representations automatically, which is far more expressive but needs lots of data and the optimization is non-convex.**
- Interpretability: SVM is backed by a complete mathematical derivation; deep networks are closer to a black box.
- Optimization: neural networks use gradient descent, are sensitive to initialization and can get stuck in local optima; SVM converges to the global optimum regardless of initialization.
- Multi-class: a single neural network suffices; SVM needs to combine several binary classifiers.

Q22:***In which situations is SVM not suitable?***

- **SVM is a poor fit for very large datasets (kernel SVM trains too slowly), data with many missing values, multi-class problems, and cases where the feature dimension is far larger than the number of samples.**

Q23:***Explain what support vectors are?***

- **Support vectors are the samples closest to the separating hyperplane, and only they determine its position: removing them changes the hyperplane, removing other points does not. Prediction also uses only the support vectors.**

Q24:***Provide an intuitive explanation of the soft-margin Support Vector Machine (SVM)***

- **Real data has noise and outliers, so a hard margin may have no solution, or may end up very narrow just to accommodate a few outliers. The soft margin lets a few points cross the line (fall inside the margin or even be misclassified), but every violation costs something; in essence it trades off "margin as wide as possible" against "as few errors as possible".**
- Each sample gets a slack variable $\xi_i$ for how much it violates the margin: 0 means outside the margin, between 0 and 1 means inside the margin but correctly classified, above 1 means misclassified.
- A larger C is closer to a hard margin (narrow margin, overfitting risk); a smaller C is more tolerant (wide margin, possible underfitting).
- It is equivalent to "hinge loss + L2 regularization" (see Q11); in the dual it only shows up as an upper bound on the multipliers, $0\le\alpha_i\le C$.
- Key formula: $\min\ \frac{1}{2}\|w\|^2 + C\sum_i\xi_i$, subject to $y_i(w^Tx_i+b)\ge 1-\xi_i,\ \xi_i\ge 0$

Q25:***Explain the convex optimization problem in SVM***

- **Convex optimization requires a convex objective, convex inequality constraints and affine equality constraints, so that any local optimum is the global optimum. The SVM primal minimizes $\frac{1}{2}\|w\|^2$ (a convex quadratic) under constraints that are linear inequalities in $w,b$, so it is a convex quadratic program.**
- In practice it is converted to the dual problem with the Lagrangian method and the dual is solved.

![https://app.yinxiang.com/shard/s24/res/bbd4d0b0-071c-4661-bf9a-ec42d7014c74.gif](https://app.yinxiang.com/shard/s24/res/bbd4d0b0-071c-4661-bf9a-ec42d7014c74.gif)

![https://app.yinxiang.com/shard/s24/res/cb08997e-1d24-4cf3-86f8-a62e813f3df4.gif](https://app.yinxiang.com/shard/s24/res/cb08997e-1d24-4cf3-86f8-a62e813f3df4.gif)

![https://app.yinxiang.com/shard/s24/res/bd9293e2-da82-46ef-9835-d2d11c982742.gif](https://app.yinxiang.com/shard/s24/res/bd9293e2-da82-46ef-9835-d2d11c982742.gif)

![https://app.yinxiang.com/shard/s24/res/aa4102c0-7780-4825-87d4-2089b018264f.gif](https://app.yinxiang.com/shard/s24/res/aa4102c0-7780-4825-87d4-2089b018264f.gif)

![https://app.yinxiang.com/shard/s24/res/4a63af41-25e5-41ff-a5eb-0be2213030a9.gif](https://app.yinxiang.com/shard/s24/res/4a63af41-25e5-41ff-a5eb-0be2213030a9.gif)

![https://app.yinxiang.com/shard/s24/res/6c214eb0-6937-4e6f-b1e0-96a12199fc25.gif](https://app.yinxiang.com/shard/s24/res/6c214eb0-6937-4e6f-b1e0-96a12199fc25.gif)

Q26:***What is the polynomial kernel?***

- **The polynomial kernel uses a polynomial function to implicitly map data into a higher-dimensional space, making originally non-separable data linearly separable.**
- It is a global kernel: even distant points affect the kernel value.
- A larger degree p means a higher-dimensional mapping and a more complex model, risking overfitting; too small a p may underfit.
- Key formula: $K(x,y)=(\gamma\, x^Ty+r)^p$

![https://app.yinxiang.com/shard/s24/res/ed13238c-98af-4a65-a531-beee76ef7da3.gif](https://app.yinxiang.com/shard/s24/res/ed13238c-98af-4a65-a531-beee76ef7da3.gif)

Reference: https://www.pycodemates.com/2022/10/svm-kernels-polynomial-kernel.html

Q27***:What is the Gaussian (RBF) kernel function?***

- **The Gaussian kernel measures similarity from the distance between two samples: close points give a value near 1, distant points a value near 0; it is equivalent to mapping data into an infinite-dimensional space.**
- It is a local kernel: only nearby points influence each other noticeably.
- It has few parameters and works well on both small and large datasets; **when you don't know which kernel to use, RBF is usually the first choice**.
- Key formula: $K(x,y)=\exp\left(-\gamma\|x-y\|^2\right)$, where $\gamma=\frac{1}{2\sigma^2}$

![https://app.yinxiang.com/shard/s24/res/19dcaf08-349c-44db-9b7e-6c45d59c3758.gif](https://app.yinxiang.com/shard/s24/res/19dcaf08-349c-44db-9b7e-6c45d59c3758.gif)

Q28:***What is the difference between a decision boundary and a hyperplane***?

- **A hyperplane is a geometric concept: a subspace one dimension lower than the surrounding space (a line in 2D, a plane in 3D); a decision boundary is a functional concept: the boundary that separates the classes.**
- In a linear SVM the decision boundary is exactly a hyperplane; with a kernel, the decision boundary is usually a curved surface in the original space.

Q29:***How does redundant data affect the performance of SVM***?

- **Redundant or irrelevant features degrade SVM classification: they add extra noise when mapped to high dimensions and distort the distance and inner-product computations, which hurts the choice of separating surface. That is why feature selection is usually done before using SVM.**

Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4709852/

Q30:***Explain the role of the Lagrangian method in the SVM algorithm***

- **The Lagrange multiplier method turns the SVM's constrained convex quadratic program into a Lagrangian, then uses strong duality (Slater's condition holds) to turn it into the dual problem, and finally recovers $w$ and $b$ from the multipliers via the KKT conditions.**

Q31:***Explain the role of the dual problem in SVM***

- **The dual eliminates $w$ and $b$, turning the optimization into a problem over the Lagrange multipliers only, which is easier to solve; more importantly, the dual contains only inner products between samples, so kernel functions can be introduced naturally to handle non-linear classification.**

Q32:***How does the value of the hyperparameter C affect SVM***?

- **A larger C penalizes misclassified samples more heavily, so the model tries not to give up on any outlier: the margin narrows, the boundary gets more complex, and it overfits more easily; a smaller C tolerates outliers more, giving a wider margin and a simpler model, but possibly underfitting.**
- In scikit-learn, C defaults to 1.

Q33:***How does the value of the hyperparameter gamma affect SVM***?

- **gamma controls how far a single sample's influence reaches in the RBF kernel. A small gamma means each sample influences a wide area and the decision boundary is smooth; a large gamma means a small area, a more curved boundary, usually more support vectors, and a tendency to overfit.**
- In scikit-learn the default is `gamma='scale'`, i.e. `1/(n_features * X.var())` (older versions defaulted to `1/n_features`).

Q34:***Is there a relationship between the number of support vectors and the performance of an SVM classifier***?

- **Yes: more support vectors usually means a more complex model that overfits more easily and predicts more slowly; very few support vectors means a simple model that may underfit.**

Q35:***What is a structured support vector machine***?

- **A structured SVM generalizes the SVM: a regular SVM outputs a single label or value, while a structured SVM can output structured objects such as sequences, trees or sets.**

Q36:***Explain the idea of the SMO algorithm***;

- **SMO is an efficient algorithm for solving the SVM dual: each step picks just two Lagrange multipliers to optimize while fixing the rest; this two-variable subproblem has a closed-form solution, and the process repeats until all multipliers satisfy the KKT conditions.**
- Why two: the dual has the equality constraint $\sum_i\alpha_i y_i=0$, so changing only one multiplier would break it.
- Every step decreases the objective, and the subproblems need no numerical optimization, so it is fast.

Q37:***State the respective advantages of using support vector machines versus logistic regression for classification***

- **SVM's advantages are that it only cares about points near the boundary, so it is insensitive to outliers, and that it handles non-linear problems efficiently with the kernel trick; logistic regression's advantages are fast training, suitability for large-scale data, and directly outputting probabilities.**

Q38***:When should you use SVM, and when should you use logistic regression***?

- **Use logistic regression for large datasets; use (kernel) SVM for small-to-medium data, data with many outliers, or data with non-linear structure.**
- When the feature dimension is far larger than the number of samples, the data is often already linearly separable, so use logistic regression or a linear SVM.
- With few features and a moderate number of samples, use an RBF-kernel SVM.
- With a very large number of samples, kernel SVM is inefficient; use logistic regression or a linear SVM.

Q39:***What are slack variables in SVM***?

- **A slack variable $\xi_i$ is the per-sample "tolerance" introduced by the soft-margin SVM, measuring how much that sample violates the margin constraint. The hard margin requires a functional margin of at least 1; the soft margin relaxes this to at least $1-\xi_i$, tolerating noise and outliers.**
- A larger $\xi_i$ means a worse violation, and above 1 means misclassified; the objective penalizes them with $C\sum_i\xi_i$.

Q40:***The meaning of the parameters C and gamma in SVM***

- **C is the penalty factor that controls tolerance for misclassified samples: the larger C, the more the model focuses on misclassified points; gamma is the RBF kernel parameter that determines how curved the decision boundary is: larger gamma gives a more curved boundary, smaller gamma a smoother one.**

Reference: [https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine](https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine)

Q41:***How would you use SVM to classify non-linearly separable data***

- **Combine two tools: a soft margin (slack variables plus the penalty factor C) to tolerate a few points that cannot be separated, and a kernel function to implicitly map the data into a high-dimensional space where it becomes linearly separable.**

Q42:***Can you give an intuitive explanation of PAC learning theory?***

- **PAC (Probably Approximately Correct) theory says that with only finite samples and an unknown true distribution, it is unrealistic to demand a model with zero error; we can only require the algorithm to learn, with high probability, a model with small error.**
- PAC learnable: an algorithm can, in polynomial time and with a reasonable number of samples, learn with probability at least $1-\delta$ a hypothesis whose error is at most $\epsilon$.
- It tells us how many samples we need: the larger the hypothesis space, the smaller the required error and the higher the confidence, the more samples are needed.

Q43:***Can support vector machines be used for outlier detection?***

- **Yes, with a One-class SVM. It trains only on normal samples and finds a hyperplane that maximizes the margin between the data and the origin, effectively drawing a "circle" around the normal data; a new sample inside the circle is normal, outside it is an outlier.**

Q44:***Does SVM suffer from the curse of dimensionality?***

- **Partly. The kernel trick means SVM never computes explicitly in the high-dimensional space, so computation does not explode with dimension; but in high dimensions the data becomes sparse and the model can still learn noise, so the overfitting risk remains.**

Q45:***What is Mercer's theorem, and how is it related to SVM?***

- **Mercer's theorem gives the condition for a function to be a valid kernel: for any set of samples, its kernel matrix must be symmetric positive semi-definite.**
- Significance: to check whether a function is a valid kernel, you don't need to find the mapping $\phi$; you only need to check that the kernel matrix is symmetric positive semi-definite.

Q46:***When using SVM, how can the RBF kernel be optimized to handle a large number of training samples?***

- **Use random features (Random Fourier Features): build a random explicit mapping into a finite-dimensional space whose inner products approximate the RBF kernel, then train a linear SVM there. Computation and storage then grow linearly with the number of samples, so it scales to large data.**
- The method comes from the NIPS 2007 paper "Random Features for Large-Scale Kernel Machines"; the mapping is built by Monte Carlo sampling.

Reference: [https://maelfabien.github.io/machinelearning/largescale/](https://maelfabien.github.io/machinelearning/largescale/)

[https://www.zhihu.com/question/19591450](https://www.zhihu.com/question/19591450)

Q47:***Why is SVM no longer popular?***

- **This is the big-data era, with millions of samples; kernel SVM training costs roughly quadratic to cubic time in the number of samples and scales poorly, while deep learning performs better on big data, images and text and learns features automatically, so SVM's popularity has declined.**

Q48:***When does SVM perform poorly?***

- **When the sample size is very large and the kernel mapping is very high-dimensional, the computation is too heavy; when the feature dimension is far larger than the number of samples, results are mediocre; and it also does poorly when the data has many missing values.**

Q49:***How do you choose a kernel for SVM?***

- **With high feature dimension or many samples, use a linear kernel (or logistic regression); with low feature dimension and a moderate number of samples, use the Gaussian (RBF) kernel.**
- With high feature dimension the data is usually already close to linearly separable, so there is no need to map it further.
- With many samples, the Gaussian kernel must compute pairwise inner products, which costs far more than the linear kernel; you can hand-add some features and then use a linear model.
