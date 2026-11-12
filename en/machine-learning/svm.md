# SVM

Q1.***What is SVM***?

- A Support Vector Machine (SVM) is a supervised learning method used for *classification*, *regression* and *outlier detection*. The goal of the SVM algorithm is to find a hyperplane in an N-dimensional space (N = number of features) that maximizes the margin between the two classes of data in the feature space.
- SVM only focuses on the points that are hardest to separate (the support vectors), whereas other classifiers pay attention to all points.

Q2:***What is a hyperplane in SVM***?

- A hyperplane is the decision boundary that helps classify data points. Data points falling on either side of the hyperplane can be attributed to different classes. The dimension of the hyperplane depends on the number of features. If the number of input features is 2, the hyperplane is just a line. If the number of input features is 3, the hyperplane becomes a two-dimensional plane.
- To separate two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find the plane with the maximum margin, i.e. the maximum distance between the data points of the two classes.

![https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.png](https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.png)

Q3:***What are support vectors in SVM***?

- Support vectors are the data points closest to the hyperplane; if these points of the dataset were removed, the position of the separating hyperplane would change. Using these support vectors, we maximize the margin of the classifier. *Only* the support vectors are used when computing predictions.

![https://miro.medium.com/max/1400/0*ecA4Ls8kBYSM5nza.jpg](https://miro.medium.com/max/1400/0*ecA4Ls8kBYSM5nza.jpg)

Q4:***What are hard-margin and soft-margin SVMs***?

- A Hard-Margin SVM has *linearly separable* training data. No data points are allowed inside the margin region. This type of linear classification is called hard-margin classification.
- A Soft-Margin SVM has training data that is *not linearly separable*. Margin violation means choosing a hyperplane that allows some data points to stay inside the margin region or on the wrong side of the hyperplane.

Q5:***What types of SVM kernels do you know***?

- Commonly: the linear kernel, the polynomial kernel, and the Gaussian kernel

Q6:***Why use the kernel trick***?

- When SVM handles data that is not linearly separable, it needs a kernel function to map the data from a low dimension to a higher dimension so that a boundary can be found and the data classified; but as the number of dimensions grows, the computation becomes very expensive. The kernel trick lets us compute on the original (low-dimensional) features while the actual classification effect (via the inner product) takes place in the high-dimensional space. This avoids complex computation directly in the high-dimensional space and truly solves the problem of SVM on linearly non-separable data.

Q7:***What are the applications of SVM***?

SVM relies on supervised learning algorithms. The goal of using SVM is to correctly *classify* unseen data. *Some common applications of SVM are:

- Face detection: SVMc classifies parts of an image as face and non-face and creates a square boundary around the face.
- Text and hypertext categorization: SVMs allow text and hypertext categorization for both inductive and transductive models. They use training data to classify documents into different categories. Classification is based on the generated score, which is then compared with a threshold.
- Image classification: using SVMs provides better search accuracy for image classification. It offers higher accuracy compared with traditional query-based search techniques.
- Bioinformatics: this includes protein classification and cancer classification. We use SVM to identify the classification of genes, patients based on genes, and other biological problems.
- Protein fold and remote homology detection: SVM algorithms are applied to protein remote homology detection.
- Handwriting recognition: we use SVMs to recognize widely used handwritten characters.
- Generalized Predictive Control (GPC): SVM-based GPC is used to control chaotic dynamics with useful parameters.

Q8:***Name some advantages of SVM***?

- Guaranteed optimality: owing to the nature of convex optimization, the solution will always be a global minimum rather than a local minimum.
- Abundant implementations: we can access it conveniently, whether from Python or Matlab.
- SVM can be used for both *linearly separable* and *non-linearly separable* data. Linearly separable data uses a hard margin, while non-linearly separable data uses a soft margin.
- SVM *fits* semi-supervised learning models. It can be used in settings where data is both labeled and unlabeled. It only requires a condition on the minimization problem, known as the Transductive SVM.
- *Feature Mapping* used to place a heavy burden on the computational complexity of the model's overall training performance. However, with the help of the Kernel Trick, SVM can perform feature mapping using simple dot products.

Q9:***For an `N`-dimensional dataset, what is the minimum possible number of support vectors? (Assume the data is perfectly linearly separable)***

- *Regardless* of the *dimensionality* or size of the dataset, the number of support vectors can be as few as `2`.

Q10:***What happens when there is no clear hyperplane in SVM?***

- Data is rarely as clean as a hyperplane, which is a line that linearly separates and classifies a set of data. To classify the dataset, it is necessary to move from a 2D view of the data to a 3D view.
- This "lifting" of the data points represents *mapping* the data into a *higher dimension*. This is called kernelling.

![https://66.media.tumblr.com/9bffea56372d28d2a30f80557451e824/tumblr_inline_o9aabehtqP1u37g00_540.png](https://66.media.tumblr.com/9bffea56372d28d2a30f80557451e824/tumblr_inline_o9aabehtqP1u37g00_540.png)

Q11:***What is the hinge loss in SVM?***

- The hinge loss function is mainly associated with soft-margin SVMs. It incorporates the margin, or distance from the classification boundary, into the cost calculation. Even if new observations are classified correctly, they are penalized if their distance from the decision boundary is not large enough. The hinge loss increases linearly.
- The main purpose of introducing the hinge loss is to penalize misclassified samples in soft-margin SVM.
- The hinge loss formula is: , where the first term is the misclassification loss and the second term is the margin distance.
    
    ![https://app.yinxiang.com/shard/s24/res/8d14cae8-d22f-499a-aeb8-7a8e535fc62b.gif](https://app.yinxiang.com/shard/s24/res/8d14cae8-d22f-499a-aeb8-7a8e535fc62b.gif)
    

![https://miro.medium.com/max/575/1*PGqpYm7o5GCbDXxXErr2JA.png](https://miro.medium.com/max/575/1*PGqpYm7o5GCbDXxXErr2JA.png)

Reference: [https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/](https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/)

Q12:***The role of the hyperparameter C in SVM***

- C: the penalty coefficient C of the objective function, used in soft-margin SVM to balance the classification margin and the loss from misclassified samples;

Q13: ***Describe the hyperparameters commonly used when designing an SVM classifier***

- An SVM model has two very important parameters, C and gamma. C is the penalty coefficient, i.e. the tolerance for errors. A higher C means errors are less tolerated, which easily leads to overfitting. A smaller C easily leads to underfitting. A C that is too large or too small will both worsen generalization ability; gamma is a parameter that comes with the RBF function once it is chosen as the kernel. It implicitly determines the distribution of the data after it is mapped into the new feature space: the larger gamma is, the fewer support vectors; the smaller gamma is, the more support vectors. The number of support vectors affects the speed of training and prediction.

Q14:***Compare the KNN algorithm with the SVM algorithm***

- KNN has no training process; it simply measures distances between the data and the training data to perform classification; SVM requires training to determine the parameters of the hyperplane;
- KNN uses all training sample points at prediction time and is a non-sparse model; SVM uses only the support vectors and is a sparse model;
- KNN has only one parameter, k, while SVM has more parameters, including the coefficient of the slack variables and the specific kernel function;

Q15:***What is the difference between using SVM for classification and for regression?***

- The difference between SVR regression and SVM classification is that in SVR the sample points ultimately belong to only one class, and the optimal hyperplane it seeks is not one that separates two or more classes of sample points "as far apart as possible" as in SVM, but one that minimizes the total deviation of all sample points from the hyperplane.
- SVM maximizes the "distance" of the sample points closest to the hyperplane; SVR minimizes the "distance" of the sample points farthest from the hyperplane. Reference: [https://www.cnblogs.com/belter/p/8975606.html](https://www.cnblogs.com/belter/p/8975606.html)

Reference: https://www.quora.com/What-is-the-main-difference-between-a-SVM-and-SVR

Q16:***What is a convex hull?***

- The convex hull of a set of points is defined as the smallest convex polygon that encloses all the points in the set. Convex means the polygon has no corners that bend inwards. In the figure below, the left is a convex polygon and the right is a non-convex polygon.
    
    ![https://miro.medium.com/max/560/1*xuqyWJ49_oNxrXGE6TehVw.png](https://miro.medium.com/max/560/1*xuqyWJ49_oNxrXGE6TehVw.png)
    

Q17:***How can SVM be used for anomaly detection?***

- Use a one-class SVM to detect anomalies in the data. The basic idea is to train a hypersphere with SVM that separates normal data from abnormal data: if a data point lies inside the hypersphere, it is judged to belong to the data distribution; otherwise, it is judged to be an outlier.

Q18: ***Compare SVM and logistic regression***

- Both are supervised classification models
- LR is sensitive to outliers; SVM is not sensitive to outliers. SVM only considers local points near the boundary, while logistic regression considers the global picture (points far away also contribute to determining the boundary, although their effect is relatively smaller). The hyperplane found by the LR model tries to keep all points as far from it as possible, whereas the hyperplane sought by SVM only keeps the points closest to the separating line as far away as possible, i.e. only the support vector samples are used
- Different loss functions: logistic regression uses cross-entropy loss, SVM uses hinge loss
- Different computational complexity: for massive data, SVM is less efficient and LR is more efficient
- Different ways of handling non-linear problems: LR relies mainly on feature construction and must combine cross features and discretize features, while SVM relies mainly on the kernel trick

Q19:***Explain what the kernel trick is***

- For a training set that is not linearly separable, a kernel function is needed to non-linearly map it into a high-dimensional space to turn it into a linearly separable dataset; when the dimension is very high, one has to compute the mapped vectors ,  separately and then compute their inner product (the inner product is needed because the SVM solution formula contains inner product operations), which is computationally very expensive;
    
    ![https://app.yinxiang.com/shard/s24/res/f3d68831-c5ee-4e9c-8737-49753082af57.gif](https://app.yinxiang.com/shard/s24/res/f3d68831-c5ee-4e9c-8737-49753082af57.gif)
    
    ![https://app.yinxiang.com/shard/s24/res/d8e584b8-ff97-4155-8df7-b0e131e2f32a.gif](https://app.yinxiang.com/shard/s24/res/d8e584b8-ff97-4155-8df7-b0e131e2f32a.gif)
    
    ![https://app.yinxiang.com/shard/s24/res/44a1a853-026a-40fe-8752-da1d7302a9de.gif](https://app.yinxiang.com/shard/s24/res/44a1a853-026a-40fe-8752-da1d7302a9de.gif)
    
- The kernel trick means finding a kernel function such that , so that one only needs to compute the value of the kernel function in the low-dimensional space, without first mapping the data into the high-dimensional space, saving a large amount of computation.
    
    ![https://app.yinxiang.com/shard/s24/res/8be1a479-ee3c-45c8-aac4-aa5f44f02ed6.gif](https://app.yinxiang.com/shard/s24/res/8be1a479-ee3c-45c8-aac4-aa5f44f02ed6.gif)
    

Reference: [https://www.fanyeong.com/2017/11/13/the-kernel-trick/](https://www.fanyeong.com/2017/11/13/the-kernel-trick/)

Q20:***Compare random forest and SVM: in what situations would you prefer random forest (or SVM)?***

- SVM has an advantage on small datasets with few extreme values, while random forest needs more data but can generally produce very good and robust models.
- SVM has no strategy for handling missing values, whereas decision trees do (missing data means some feature data is missing and the vector data is incomplete; SVM needs to find a separating plane in the feature space and is strongly affected by missing data)

Q21:***Compare SVM and deep learning***

- Intuitive difference: deep learning stacks many neural network layers to learn increasingly abstract representations of the data in a hierarchical manner; SVM finds a separating boundary among the data points (a hyperplane in the linearly separable case) that is as far as possible from the support vectors of either class;
- Deep learning is relatively uninterpretable, while SVM is backed by strong mathematical derivation;
- Sensitivity to weight initialization: because neural networks use gradient descent, they are sensitive to the initial randomization of their weight matrices. This is because if the initial randomization places the network close to a local minimum of the optimization function, the accuracy will never rise above a certain threshold. In contrast, SVMs are more reliable: regardless of their initial configuration, they are guaranteed to converge to the global minimum.
- On multi-class problems, deep learning is more convenient; SVM needs multiple SVM classifiers to handle multi-class problems, while deep learning needs only a single neural network.
- When the number of features exceeds the number of samples, SVM performs poorly; SVM requires more work on feature engineering;

Q22:***In which situations is SVM not suitable?***

- If the feature dimension is far greater than the number of samples, SVM performs only moderately.
- When the sample size is very large and the kernel mapping dimension is very high, SVM's computational cost is too large, so it is not very suitable.
- When there is a large amount of missing data (some feature data is missing and the vector data is incomplete): SVM wants the samples to be linearly separable in the feature space, so the quality of the feature space is important for SVM's performance. Missing feature data will affect the quality of the training results.
- Multi-class problems

Q23:***Explain what support vectors are?***

- Support vectors are the data points closest to the separating hyperplane; if these points of the dataset were removed, the position of the separating hyperplane would change. Using these support vectors, we maximize the margin of the classifier. *Only* the support vectors are used when computing predictions.

Q24:***Provide an intuitive explanation of the soft-margin Support Vector Machine (SVM)***

Q25:***Explain the convex optimization problem in SVM***

- The goal of convex optimization is to solve extremum problems of functions with constraints. The general model is , and the following three conditions must all be satisfied:  is a convex function,  is a convex function, and  is an affine function;
    
    ![https://app.yinxiang.com/shard/s24/res/bbd4d0b0-071c-4661-bf9a-ec42d7014c74.gif](https://app.yinxiang.com/shard/s24/res/bbd4d0b0-071c-4661-bf9a-ec42d7014c74.gif)
    
    ![https://app.yinxiang.com/shard/s24/res/cb08997e-1d24-4cf3-86f8-a62e813f3df4.gif](https://app.yinxiang.com/shard/s24/res/cb08997e-1d24-4cf3-86f8-a62e813f3df4.gif)
    
    ![https://app.yinxiang.com/shard/s24/res/bd9293e2-da82-46ef-9835-d2d11c982742.gif](https://app.yinxiang.com/shard/s24/res/bd9293e2-da82-46ef-9835-d2d11c982742.gif)
    
    ![https://app.yinxiang.com/shard/s24/res/aa4102c0-7780-4825-87d4-2089b018264f.gif](https://app.yinxiang.com/shard/s24/res/aa4102c0-7780-4825-87d4-2089b018264f.gif)
    
- The primal problem of SVM is , where  is a typical convex function, and the highest order of the constraint is only first order, so it is indeed an affine function; therefore the primal problem is a convex optimization problem, which can be converted into an unconstrained problem via the Lagrangian method and then solved through its dual problem.
    
    ![https://app.yinxiang.com/shard/s24/res/4a63af41-25e5-41ff-a5eb-0be2213030a9.gif](https://app.yinxiang.com/shard/s24/res/4a63af41-25e5-41ff-a5eb-0be2213030a9.gif)
    
    ![https://app.yinxiang.com/shard/s24/res/6c214eb0-6937-4e6f-b1e0-96a12199fc25.gif](https://app.yinxiang.com/shard/s24/res/6c214eb0-6937-4e6f-b1e0-96a12199fc25.gif)
    

Q26:***What is the polynomial kernel?***

- The polynomial kernel relies on a polynomial function to map data into a high-dimensional space, making originally linearly non-separable data linearly separable. Its mathematical form is
    
    ![https://app.yinxiang.com/shard/s24/res/ed13238c-98af-4a65-a531-beee76ef7da3.gif](https://app.yinxiang.com/shard/s24/res/ed13238c-98af-4a65-a531-beee76ef7da3.gif)
    
- The polynomial kernel function can map a low-dimensional input space into a high-dimensional feature space. It is suitable for orthonormalized data (vectors that are orthogonal with norm 1) and is a global kernel function, allowing data points that are far apart to influence the value of the kernel function. The larger the parameter p, the higher the mapped dimension, leading to a more complex model that may overfit. Conversely, a parameter p that is too small leads to underfitting.

Reference: https://www.pycodemates.com/2022/10/svm-kernels-polynomial-kernel.html

Q27***:What is the Gaussian (RBF) kernel function?***

- The Gaussian kernel function maps the original space into an infinite-dimensional space. Its mathematical form is ; because the shape of the function is close to a Gaussian distribution, it is called the Gaussian kernel function;
    
    ![https://app.yinxiang.com/shard/s24/res/19dcaf08-349c-44db-9b7e-6c45d59c3758.gif](https://app.yinxiang.com/shard/s24/res/19dcaf08-349c-44db-9b7e-6c45d59c3758.gif)
    
- The Gaussian radial basis function is a strongly local kernel function that can map a sample into a higher-dimensional space. It is the most widely used kernel, performing well for both large and small samples, and it has fewer parameters than the polynomial kernel. Therefore, **in most cases, when you do not know which kernel function to use, the Gaussian kernel is the first choice**.

Q28:***What is the difference between a decision boundary and a hyperplane***?

- A hyperplane is a subspace of a space whose dimension is one less than that of the space it lives in. If the data space itself is three-dimensional, its hyperplane is a two-dimensional plane; if the data space itself is two-dimensional, its hyperplane is a one-dimensional line.
- A decision boundary is a subset of hyperplanes. In a binary classification problem, if a hyperplane can divide the data into two sets, each containing a single class, we say that this hyperplane is the "decision boundary" of the data.

Q29:***How does redundant data affect the performance of SVM***?

- Redundant data/features in a dataset can worsen SVM's classification performance. Redundant features introduce a lot of unnecessary accumulated noise when the original data is mapped to a high dimension, adversely affecting SVM's choice of the separating surface.

Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4709852/

Q30:***Explain the role of the Lagrangian method in the SVM algorithm***

- The Lagrange multiplier method converts the constrained convex quadratic programming problem in SVM into an unconstrained problem, then, based on Slater's condition, converts it into the Lagrangian dual problem, and the parameters w, b of the primal problem are solved via the KKT conditions;

Q31:***Explain the role of the dual problem in SVM***

- The important role of the Lagrangian dual is to bring the computation of w, b forward and eliminate w, b, so that the optimization function becomes a single-parameter optimization problem over the Lagrange multipliers, making the problem easier to solve
- The dual problem naturally introduces kernel functions, thereby extending to non-linear classification problems

Q32:***How does the value of the hyperparameter C affect SVM***?

- When the penalty coefficient C is relatively large, our `loss function will also be larger`, which means we are `unwilling to give up outliers that are relatively far away`. As a result we will `have more support vectors`, which means `the model of support vectors and hyperplane will become more complex and more prone to overfitting`.
- When the penalty coefficient C is relatively small, it means we `do not want to bother with those outliers`, and will choose `fewer samples as support vectors`, so the final model of support vectors and hyperplane will also be simple. The default value of C in `scikit-learn` is 1

Q33:***How does the value of the hyperparameter gamma affect SVM***?

- When `γ` is relatively small, `a single sample has a relatively small influence on the whole separating hyperplane and is less likely to be selected as a support vector`
- When `γ` is relatively large, `a single sample has a relatively large influence on the whole separating hyperplane and is more likely to be selected as a support vector**`, or in other words the model will have more support vectors overall. The default value in `scikit-learn` is `1/n_features`

Q34:***Is there a relationship between the number of support vectors and the performance of an SVM classifier***?

- The more support vectors, the more complex the model and the more prone SVM is to overfitting
- The fewer support vectors, the simpler the model and the more prone SVM is to underfitting

Q35:***What is a structured support vector machine***?

- Structured SVM is a generalization of SVM. Unlike a regular SVM, which only considers univariate predictions (such as classification and regression), a structured SVM allows structured outputs (e.g. trees, sequences, sets).

Q36:***Explain the idea of the SMO algorithm***;

- Simple version: the SMO algorithm decomposes a large optimization problem into multiple small optimization problems. These small problems are usually easy to solve, and solving them sequentially gives exactly the same result as solving them as a whole
- Detailed version: the SMO algorithm is a heuristic algorithm. Its basic idea is: **if the solutions of all variables satisfy the KKT conditions (Karush-Kuhn-Tucker conditions) of this optimization problem, then the solution of the optimization problem has been obtained, because the KKT conditions are necessary and sufficient conditions for this optimization problem**. Otherwise, choose two variables, fix the other variables, and build a quadratic programming problem with respect to these two variables. The solution of this quadratic programming problem with respect to the two variables should be closer to the solution of the original quadratic programming problem, because it makes the objective function value of the original quadratic programming problem smaller. Importantly, the subproblem can then be solved analytically, which greatly improves the computational speed of the whole algorithm.

Q37:***State the respective advantages of using support vector machines versus logistic regression for classification***

- SVM is less affected by outliers in the data, because it only cares about the points closest to the decision boundary; logistic regression attends to all data and is easily affected by outliers
- Logistic regression trains fast and is suitable for large batches of training data
- For non-linear data spaces, SVM is faster (the kernel trick can be used to optimize the computation)

Q38***:When should you use SVM, and when should you use logistic regression***?

- When the data contains many anomalies/outliers, use SVM
- When the feature dimension is far greater than the number of data samples, use logistic regression; when the feature dimension is relatively large, use SVM
- When processing large batches of data, SVM is not efficient

Q39:***What are slack variables in SVM***?

- The constraint of hard-margin SVM is $y_{i}(wx_{i}+b)\geq 1$. To handle data that is not strictly linearly separable, soft-margin SVM introduces slack variables $\xi _{i}$, and the constraint after introducing them is $y_{i}(wx_{i}+b)\geq\xi _{i}$
- The slack variable $\xi _{i}$ is the functional distance of a data point to its separating boundary. Originally the functional distance from a data point to the separating surface was required to be greater than or equal to 1; now this is relaxed to the functional distance from a data point to the separating surface being greater than or equal to $1-\xi _{i}$, which is a form of tolerance for anomalous outliers.

Q40:***The meaning of the parameters C and gamma in SVM***

- C, also called the penalty factor, is the parameter in SVM that controls errors. The optimization objective of soft-margin SVM is $min\frac{1}{2}w^{T}w+C\sum_{i=1}^{n}\xi _{i}$; the larger C is, the more the model pays attention to misclassified points;
- The gamma parameter is a parameter of the Gaussian kernel. Gamma determines how much curvature the SVM needs in the decision boundary: a larger gamma means more curvature, a lower gamma means less curvature.

Reference: [https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine](https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine)

Q41:***How would you use SVM to classify non-linearly separable data***

- Add slack variables and a penalty factor to find the "relatively best" hyperplane that classifies the data as correctly as possible;
- Map the low-dimensional (linearly non-separable) data into a high-dimensional space so that the data is linearly separable there, and introduce kernel functions to solve the problem of complex high-dimensional computation

Q42:***Can you give an intuitive explanation of PAC learning theory?***

- PAC (Probably Approximately Correct) learning theory. Its main idea is: since we do not know the true data distribution, nor the true target function, it is unrealistic to expect to learn a function with an expected error of 0 from a finite number of training samples; we need to lower our expectations of the learning algorithm's ability and only require that the learning algorithm can learn an approximately correct hypothesis with a certain probability;
- A PAC-learnable algorithm is a learning algorithm that can learn an approximately correct probability distribution from a reasonable amount of training data in polynomial time
- PAC learning theory can help analyze under what conditions a machine learning method can learn an approximately correct classifier; if the model's hypothesis space is to be larger and the generalization error smaller, more samples are needed.

Q43:***Can support vector machines be used for outlier detection?***

- Yes, using one-class SVM; in anomaly detection scenarios, one-class SVM finds a hyperplane, with the learning objective of maximizing the distance from the separating hyperplane to the origin; the hyperplane encircles the positive samples, and this circle is used to determine whether a new data point is a positive sample. If it is inside the circle, it is a positive sample; if it is outside the circle, it is an outlier

Q44:***Does SVM suffer from the curse of dimensionality?***

- The curse of dimensionality: to obtain better classification results, we need to add more features. As the number of features increases, although the classifier's fit becomes more accurate, the amount of computation increases significantly and the density of data in the space decreases sharply. The classifier then also learns the noise, so overfitting occurs;
- Through the kernel trick, SVM replaces high-dimensional operations with an equivalent low-dimensional kernel function, avoiding the increase in computation; but it can hardly avoid the overfitting problem of the curse of dimensionality

Q45:***What is Mercer's theorem, and how is it related to SVM?***

- Mercer's theorem defines a necessary condition for kernel functions: **K is a valid kernel function ==> the kernel matrix K is symmetric positive semi-definite**
- Mercer's theorem shows that to prove $K$ is a valid kernel function, we do not need to find the mapping function $\phi$; we only need to compute each $K_{ij}$ on the training set and then check whether the matrix $K$ is positive semi-definite (using methods such as the leading principal minors being greater than or equal to zero).

Q46:***When using SVM, how can the RBF kernel be optimized to handle a large number of training samples?***

- By using Random Features, kernel methods can be scaled to large datasets; this method comes from "Random Features for Large-Scale Kernel Machines" at NIPS 2017. Its basic idea is to construct a "random" mapping $z:R^{d}\rightarrow R^{D}$ that maps the data directly into a high-dimensional space, such that the inner product in this space approximates the kernel function $K(x,y)$; the feature mapping can be approximated by MC sampling; doing linear SVM in the $R^{D}$ space scales effectively to large datasets, with computation and storage linear in the data.

Reference: [https://maelfabien.github.io/machinelearning/largescale/](https://maelfabien.github.io/machinelearning/largescale/)

[https://www.zhihu.com/question/19591450](https://www.zhihu.com/question/19591450)

Q47:***Why is SVM no longer popular?***

- We are now in the era of big data, and training samples are generally massive in scale. Because of SVM's extremely large computational cost on large samples, its popularity has declined

Q48:***When does SVM perform poorly?***

- When the sample size is very large and the kernel mapping dimension is very high, SVM's computational cost is too large, so it is not very suitable
- If the feature dimension is far greater than the number of samples, SVM performs only moderately
- When there is a large amount of missing data

Q49:***How do you choose a kernel for SVM?***

- If the feature dimension is very high, the data is often linearly separable (the idea of SVM for solving non-linear classification problems is to map samples into a higher-dimensional feature space), so LR or a linear-kernel SVM can be used;
- If the number of samples is very large, since the objective function involves computing inner products between every pair of samples when solving the optimization problem, a Gaussian kernel clearly requires more computation than a linear kernel, so manually add some features to make the data linearly separable, and then use LR or a linear-kernel SVM;
- If neither of the above holds, i.e. the feature dimension is small and the number of samples is normal, a Gaussian-kernel SVM can be used.
