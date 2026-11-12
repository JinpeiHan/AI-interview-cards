# PCA

Q1: ***What is PCA***?

- ***PCA*** is commonly used for *dimensionality reduction*; for a high-dimensional dataset, it finds as few orthogonal vectors (principal components) as possible to represent the informational features of the data.
- The principal components are a series of unit vectors, and the direction of each principal component is the direction that maximizes the variance of the projected data (representing the largest amount of information).

Q2: ****Can *PCA* be used for feature selection?**  

- Feature selection means selecting a subset of features from the full feature set; PCA, by defining principal components, defines a set of new features that reflect most of the information in the dataset, and the new features are not equivalent to the original features. Therefore PCA is not a feature selection method.

Q3: *****How are the first and second principal component axes chosen in *PCA*?**  

- The simplest way is to find the projection that maximizes variance. The first principal component is **the direction in space along which the projection has the largest variance**. The second principal component is the direction that maximizes variance among all directions orthogonal to the first.


Q4: ***Why is it important to standardize the data before performing PCA?***  

- Standardizing the data puts the features on the same order of magnitude; features with values of different orders of magnitude will affect PCA's computation of the best principal components, because PCA is very sensitive to the variance of the variables.

Q5: ***What is the difference between PCA and the Random Projection method?***  

- Both are dimensionality reduction operations that map the data space from a high-dimensional space to a low-dimensional space, i.e., they choose a small set of basis vectors in the high-dimensional space, and the low-dimensional data is obtained by projecting the data onto the basis vectors. The main difference is that PCA chooses the best basis vectors by finding the directions of maximum variance in the original data, whereas random projection simply chooses directions at random.
- **The Random Projection method is more computationally efficient on high-dimensional data**

Reference: [https://www.quora.com/Is-there-any-difference-between-PCA-Principal-component-analysis-and-random-projection-when-preprocessing-data](https://www.quora.com/Is-there-any-difference-between-PCA-Principal-component-analysis-and-random-projection-when-preprocessing-data)

Q6: *****What is the difference between PCA and LDA?*****

- LDA is a supervised dimensionality reduction method, while PCA is an unsupervised one; LDA can reduce the dimensionality to at most k-1 dimensions (k being the number of classes), while PCA has no such limit; LDA chooses the projection direction with the best classification performance, while PCA chooses the direction along which the projected sample points have the largest variance.

Q7: ****How do you perform *Principal Component Analysis (PCA)*?**  

- 1.**Standardize the features**. We standardize each feature so that it has mean 0 and variance 1; features with values of different orders of magnitude prevent PCA from computing the best principal components, because PCA is very sensitive to the variance of the variables;
- 2.**Compute the covariance matrix**; *the covariance matrix is a $d$*-dimensional square matrix that shows the pairwise correlations between each pair of features;
- 3.**Compute the eigendecomposition of the covariance matrix**. We compute the eigenvectors (unit vectors) of the covariance matrix and their associated eigenvalues (the scalars that multiply the eigenvectors);
- 4.**Sort the eigenvectors from the highest eigenvalue to the lowest**. The eigenvector with the highest eigenvalue is the first principal component. Higher eigenvalues correspond to a larger amount of shared variance explained;
- 5.**Choose the number of principal components**. Select the top N eigenvectors (based on their eigenvalues) to become the N principal components;
- 6.****Recast the data along the principal component axes:**** using the feature vector formed from the eigenvectors of the covariance matrix, reorient the data from the original axes to the axes represented by the principal components; this can be done by multiplying the transpose of the original dataset by the transpose of the feature vector.


*Q9*: ****Would you choose to use PCA on a large dataset? Or is there a better option?****  

- When the numbers of observations and variables are very large, the classical PCA method requires considerable memory and huge computation time because it needs to compute the singular value decomposition of the matrix.
- On large datasets, better choices for dimensionality reduction include random projection, flash PCA, etc. Flash PCA comes from the 2014 paper "Fast Principal Component Analysis of Large-Scale Genome-Wide Data" and performs principal component analysis on a smaller subset of the data that represents most of the data's variance.

Reference: [https://www.elucidata.io/blog/principal-component-analysis-on-big-data](https://www.elucidata.io/blog/principal-component-analysis-on-big-data)


*Q10*: *****What is the difference between *PCA* and *t-SNE*?**  

- PCA is a linear dimensionality reduction method, while t-SNE (t-distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction method.
- PCA considers the directions of maximum variance in the data and preserves the global structure of the data; t-SNE preserves the local cluster structure of the data by considering the distances between points.
- PCA involves no hyperparameters, while t-SNE involves hyperparameters such as perplexity, learning rate, and number of steps.
- PCA is strongly affected by outliers, while t-SNE is less affected by outliers.
- Compared with PCA, t-SNE works better and is one of the best dimensionality reduction algorithms.


Reference: [https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/](https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/)

*Q11*: ****Does PCA check which features are redundant and discard them?****  

- PCA does not select some features and discard others. Instead, it constructs some new features (principal components) that summarize our list of data points well. Of course, these new features are built from the old features;

*Q12*: ****What is *Sparse PCA*?**  

- Sparse PCA is a technique specialized for statistical analysis and the analysis of multivariate datasets. It reduces the dimensionality of the data by introducing a sparsity structure on the input variables.
- The difference between classical PCA and sparse PCA is that classical PCA usually produces linear combinations of all input variables, while sparse PCA finds linear combinations that contain only a few variables.
- Because sparse PCA finds linear combinations containing only a small number of input variables, it reduces the original feature space to some extent, but not as compactly as ordinary PCA (in other words, it creates a sparse feature vector).



*Q13*: *****How can *PCA* be used for *anomaly detection*?**  

- After PCA maps the original data into a low-dimensional space, the original data can also be reconstructed from the data's coordinates in the low-dimensional space; suppose the coordinates of data sample $x_{i}$ mapped into the $k$-dimensional space are ${y_{1},y_{2},...,y_{k}}$; the data reconstructed from these $k$-dimensional coordinates is $x_{ik}^{'}=\sum_{i=1}^{k}y_{i}e_{i}$, where $e_{i}$ is the unit vector in the corresponding direction.
- PCA-based anomaly detection relies on the reconstruction error. Intuitively, PCA extracts the main features of the data; if a data sample is hard to reconstruct, it means the features of this sample are inconsistent with the features of the overall data, so it is clearly an anomalous sample.



*Q14*: *****What is the difference between *Principal Component Analysis* and *Independent Component Analysis*?**  

- Principal Component Analysis (PCA) is commonly used to compress information, i.e., for dimensionality reduction. It is mathematically equivalent to performing an eigenvector decomposition of the covariance matrix of the original variable matrix, selecting the few dimensions with the largest eigenvalues (representing the most information) to form the principal component space, and projecting the original data onto the principal component space*.*
- Independent Component Analysis (ICA) aims to separate information by transforming the input space into a maximally independent basis. To this end, ICA assumes that the independent bases must be
    - Statistically independent: the independent bases are mutually uncorrelated
    - Non-Gaussian, i.e., the independent components have non-Gaussian distributions

*Q15*: *****What is the relationship between the *k-Means algorithm* and *PCA*?** 

- The intuition is that PCA tries to represent all data vectors as linear combinations of a small number of eigenvectors, and does so to minimize the mean squared reconstruction error. K-means tries to represent the original data vectors with linear combinations of a small number of centroid vectors, where the linear combination weights must all be zero except for a single 1, and this is also done to minimize the mean squared reconstruction error. Therefore, the K-means algorithm can be viewed as a super-sparse PCA.

Reference: [https://stats.stackexchange.com/questions/183236/what-is-the-relation-between-k-means-clustering-and-pca](https://stats.stackexchange.com/questions/183236/what-is-the-relation-between-k-means-clustering-and-pca)
