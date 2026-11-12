# Random Forest

Q1: **How do you define a random forest?**

- A random forest is an ensemble learning method used for classification, regression or other tasks.
- A random forest works by constructing many decision trees at training time. It works by averaging multiple decision trees trained on different parts of the same training set.

Q2: ****Explain how a random forest produces outputs for classification and regression problems?****

- Classification: the output of the random forest is the class selected by the most trees.
- Regression: the output of the random forest is the mean or average prediction of the individual trees.

Q3: **What ensemble learning methods are there?**

- Ensemble learning methods combine multiple machine learning sub-models as a technique to reduce bias or variance and improve overall model performance. Methods of *ensemble learning* include:
- **Bagging** (short for *bootstrap aggregating*): if there are K sub-models, each sub-model of the ensemble is trained on one of K separate datasets. Each separate dataset is constructed by random sampling (sampling with replacement) from the original training dataset. There may be duplicates across the datasets.
    
    ![https://miro.medium.com/max/1400/1*-PXzSlXtFEGTxgcmCyMkjQ.png](https://miro.medium.com/max/1400/1*-PXzSlXtFEGTxgcmCyMkjQ.png)
    
- Boosting: builds a collection of models iteratively, with each subsequent model focusing on learning the examples that the previous model got wrong. In short, boosting iteratively improves a sequence of weak learners and uses a weighted average to produce the final strong learner.
    
    ![https://miro.medium.com/max/1400/1*trWLNnfyqdRklFmLGcF_Zw.png](https://miro.medium.com/max/1400/1*trWLNnfyqdRklFmLGcF_Zw.png)
    
- Stacking: a hierarchical structure; in the first layer, the outputs of a set of models are combined to make predictions. Different model architectures are trained on the full training dataset. Then, the **model outputs of the first layer are used as features to train a second-level model**. The learning objective of the second-level model is how best to combine the results of the initial models to reduce the training error.
    
    ![https://miro.medium.com/max/1400/1*IaqmYUSmQXE8AJw7moqwkA.png](https://miro.medium.com/max/1400/1*IaqmYUSmQXE8AJw7moqwkA.png)
    

Q4: **Does the random forest algorithm need pruning?**

- Pruning is a model compression technique in machine learning and search algorithms that reduces the size of decision trees by removing non-critical and redundant parts of the tree used to classify instances.
- Random forests usually do not need pruning, because they do not overfit the way a single decision tree does; because the trees are bootstrapped and the multiple random trees use random features, each tree is strong while being uncorrelated with the others.

Q5: **What is the relationship between random forests and decision trees?**

- A random forest is an ensemble learning method that works by constructing a large number of decision trees. Random forests can be built for both classification and regression tasks.
- Random forests outperform decision trees, and they also do not have the habit of overfitting the data the way decision trees do.
- A decision tree trained on a particular dataset will become very deep and lead to overfitting. To create a random forest, decision trees can be trained on different subsets of the training dataset, and the different decision trees can then be averaged to reduce variance.

Q6: **What hyperparameters does a random forest have?**

- The number of decision trees in the forest.
- The number of features each tree considers when splitting a node.
- The maximum depth of an individual tree.
- The minimum number of samples required to split an internal node.
- The maximum number of leaf nodes.
- The number of random features.
- The size of the bootstrapped dataset.

Q7: **Is it necessary to perform *cross-validation* in a random forest?**

- Random forest is a bagging ensemble algorithm that uses the Bootstrap; theory and practice show that each time, about 1/3 of the samples do not appear in the sample set drawn by the Bootstrap. Hence they do not take part in building the decision tree; these data are called out-of-bag (OOB) data
- When training a random forest, because the forest is built using the Bootstrap, one way to evaluate accuracy is to use the out-of-bag error estimate (OOB) instead of cross-validation.

Q8: ****How does a random forest handle missing values?****

- For classification problems, a random forest fills missing values with the mean or median of each feature.
- For regression problems, a random forest fills missing values with the mean or median of each feature, or uses a method similar to CART regression trees to fill them.

Q9: **How can the performance of a random forest be improved?**

- Increase the number of decision trees, although too many trees may lead to overfitting.
- Choose a better feature subset. A feature selection algorithm can be used to determine which features are most important, and then only the most important features are selected.
- Tune the hyperparameters. For example, the random forest can be tuned by increasing the maximum depth of each tree or decreasing the minimum number of leaf nodes of each tree.

Q10: **Introduce the steps for generating a random forest**

1. From the original training dataset, apply the bootstrap method to randomly draw, with replacement, k new bootstrap sample sets, and build k classification/regression trees from them; the samples not drawn each time form K out-of-bag (OOB) datasets.
2. Suppose there are n features; then at each node of each tree, randomly draw mtry features, and, by computing the amount of information contained in each feature, select the feature with the strongest classification ability among them to split the node.
3. Each tree is grown to its maximum extent without any pruning
4. The multiple generated trees form the random forest, which is used to classify new data; the classification result is determined by the votes of the tree classifiers.

Q11: **Why random forests do not overfit**

- In the process of building each decision tree, two points need attention: sampling and complete splitting. First come the two random sampling processes: random forest samples both the rows and the columns of the input data. For row sampling, sampling with replacement is used, meaning the sampled set may contain duplicate samples.
- For row sampling, sampling with replacement is used, meaning the sampled set may contain duplicate samples. Assuming there are N input samples, N samples are also drawn. As a result, during training the input samples of each tree are not the full set of samples, which makes over-fitting relatively unlikely. Then column sampling is performed: from M features, m are selected (m << M). After that, decision trees are built from the sampled data using complete splitting, so that each leaf node of a decision tree either cannot be split further or contains samples that all belong to the same class. *Many decision tree algorithms have an important step, pruning, but that is not done here: because the two preceding random sampling processes guarantee randomness, over-fitting does not occur even without pruning.

Q12: **Comparison of random forest and SVM**

(1) There is no need to tune many parameters, because a random forest only needs the number of trees tuned, and generally the more trees the better, whereas other machine learning algorithms, such as SVM, have very many hyperparameters to tune, e.g. choosing the most suitable kernel function, the regularization penalty, etc.
(2) Classification is relatively simple and direct. Random forests and support vector machines are both non-parametric models (their complexity grows as the number of training samples increases). Compared with general linear models, in terms of computational cost, training non-parametric models is therefore more time-consuming and labor-intensive. The more classification trees there are, the more time is needed to build the random forest model. Likewise, the support vector machine we train has many support vectors; in the worst case, there are as many support vectors as there are instances in the training set. Although we can use multi-class support vector machines, traditional multi-class classification is generally implemented as one-vs-all (one-vs-all means applying a binary classification method to multi-class classification: for example, if I want K classes, one class is treated as positive), so we still need to train one support vector machine for each class. In contrast, decision trees and random forests can handle multi-class problems with no difficulty.
(3) It is easier to get started with in practice. Random forests are simpler in terms of model training. You can easily get a good and robust model. The complexity of a random forest model is proportional to the training samples and the trees. Support vector machines require some work on hyperparameter tuning; in addition, the computational cost grows linearly as the number of classes increases.
(4) On small data, SVM excels, while random forests require more data. From experience, I would rather think that support vector machines have an advantage on small datasets with few extreme values. Random forests need more data but can generally produce very good and robust models.

Reference: [https://zhuanlan.zhihu.com/p/45040640](https://zhuanlan.zhihu.com/p/45040640)

Q13: ****What are the advantages and disadvantages of random forests****

Advantages:

- On many current datasets, it has a large advantage over other algorithms and performs well.
- It can handle very high-dimensional data (many features) without feature selection (because feature subsets are chosen randomly).
- After training, it can indicate which features are relatively important.
- Training is fast and easily parallelized (trees are independent of each other during training).
- During training, it can detect interactions between features.
- For imbalanced datasets, it can balance the errors.
- If a large portion of the features is missing, it can still maintain accuracy.

Disadvantages:

- Random forests overfit on some classification or regression problems with high noise.
- For data with attributes that take different numbers of values, attributes with more value splits have a greater influence on the random forest, so the attribute weights produced by a random forest on such data are not reliable.
