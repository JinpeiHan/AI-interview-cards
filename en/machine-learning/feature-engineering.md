# Feature Engineering

Q1: **What is the difference between feature engineering and feature selection?**

- Feature engineering allows us to create new features from existing ones to help a machine learning model make more effective and accurate predictions. Some typical feature engineering tasks include:
    - Imputing missing values in variables.
    - Encoding categorical variables as numbers.
    - Variable transformation.
    - Creating or extracting new features from the features available in the dataset.
- Feature selection allows us to select features from the feature pool (including any newly engineered features) that will help the machine learning model predict the target variable more effectively. Feature selection methods require supervised learning and are evaluated according to the performance of the trained model on a validation dataset.
- In a typical machine learning pipeline, feature selection is performed after feature engineering is complete

Q2: **Which common machine learning problems can unsupervised learning help solve?**

- Data scarcity: supervised learning needs a large amount of labeled data for a model to perform well. Unsupervised learning can automatically label unlabeled examples through methods such as clustering.
- Outliers: data quality is very important, and outliers in the data will seriously affect the performance of machine learning algorithms. Unsupervised learning can use dimensionality reduction to perform outlier detection and remove outliers, creating a solution for the normal data.
- Feature engineering: feature engineering is a labor-intensive task that requires humans to creatively design features. Unsupervised learning methods, such as representation learning, can automatically learn the right kind of features to build feature engineering.

Q3: ***Feature selection*: name some of its benefits**

- High-dimensional features and a low ratio of samples to features introduce noise into the dataset, and algorithms are more likely to overfit
- Reducing the number of features reduces the program's running time. At the same time, a smaller number of features allows you to use more complex algorithms, search more hyperparameters, or perform more evaluations.
- A smaller feature set is easier for humans to understand. This lets you focus on the main sources of predictive power and do more precise feature engineering.

Q4: **Explain One-Hot encoding and label encoding. Does the dimensionality of the dataset increase or decrease after applying these techniques?**

- Label encoding assigns an *integer value* to each *unique category value*. For example, in the "*color*" variable, "*red*" is `0`, "*green*" is `1`, "*blue*" is `2`, and so on. These integer values have a natural ordinal relationship with one another, which machine learning algorithms can understand and exploit. This encoding does not affect the dimensionality of the dataset.

![Untitled](feature-engineering/Untitled.png)

- One-Hot Encoding converts each unique integer value into a new vector containing only 0s and 1s, with the integer value determining the position in the vector, so the dimensionality of the dataset increases.

![Untitled](feature-engineering/Untitled%201.png)

**Q5: Describe the main steps that a machine learning workflow should include?**

- A machine learning pipeline should contain the series of steps involved in training a machine learning model, covering preprocessing, feature selection, classification/regression and post-processing. For example, a basic pipeline for a classification model contains:
    - Scaler: used to preprocess the data, transforming it to zero mean and unit variance
    - Feature selector: discards strongly correlated features to reduce redundancy
    - Model

Q6: **Can PCA be used for feature selection?**

- PCA is not a feature selection technique; feature selection means selecting a subset of features from the full feature set, whereas what PCA produces are Principal Components, which are linear combinations of the whole set of original feature variables

Q7: **How do you use the F-test to select features?**

- The F-test is a statistical test used to compare models and check whether the difference between models is significant
- The F-Test performs a hypothesis test on models X and Y, where X is a model created from a constant only and Y is a model created from the constant and a feature. The least-squares errors of the two models are compared, and it is checked whether the difference in error between models X and Y is significant

Reference: [https://towardsdatascience.com](https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2)

Q8: **Which feature selection methods do you know?**

- Feature selection methods are generally divided into unsupervised and supervised methods
- ****Unsupervised feature selection methods:****
    - Discard features with near-zero variance (zero variance means little information), discard features with many missing values, discard highly correlated features
- ****Supervised feature selection methods:****
    - Wrappers: use a model to score different feature subsets and finally select the best features. Each new subset is used to train a model, whose performance is then evaluated on a hold-out set. The feature subset that yields the best model performance is selected
    - Filters: to evaluate the usefulness of each feature, simply analyze the statistical relationship between the feature and the model's target, using measures such as correlation or mutual information as a proxy for model performance metrics
    - Embedded:
        - LASSO regression is essentially just a regularization method that shrinks feature weights towards zero during model training. As a result, many features end up with a weight of zero, which means they are discarded from the model;
        - The AE Bottleneck method uses a neural network for feature selection;

![Untitled](feature-engineering/Untitled%202.png)

Reference: [https://towardsdatascience.com/](https://towardsdatascience.com/)

Q9: **How can the performance of a random forest be improved?**

- By changing the **nodesize and tree size**: if you want to increase predictive power, choose a smaller node size and more trees. The trees will be larger and grow all the way to the end, and the number of trees will also increase, so their overall performance will be better

Q10: **How do you do feature engineering on unknown features (when you do not understand the real meaning of the features)?**

- When you do not have much domain knowledge but have a large number of features, manual feature engineering is relatively pointless; this is when automated feature engineering is most suitable
- Automated feature engineering includes:
    - Principal Component Analysis (PCA) and Independent Component Analysis (ICA), which map the existing data into another feature space
    - • Deep feature synthesis, which allows intermediate learning to be transferred from the intermediate layers of a neural network

Reference: [https://datascience.stackexchange.com/questions/10640/how-to-perform-feature-engineering-on-unknown-features](https://datascience.stackexchange.com/questions/10640/how-to-perform-feature-engineering-on-unknown-features)

Q11: **Describe the concrete procedure of PCA?**

- 1.**Feature standardization**. We standardize each feature so that it has mean 0 and variance 1; features whose values have different orders of magnitude prevent PCA from computing the best principal components, because PCA is very sensitive to the variance of variables;
- 2.**Compute the covariance matrix**; *the covariance matrix is a $d$*-dimensional square matrix that shows the pairwise feature correlations between every pair of features;
- 3.**Compute the eigendecomposition of the covariance matrix**. We compute the eigenvectors (unit vectors) of the covariance matrix and their associated eigenvalues (the scalars by which the eigenvectors are multiplied);
- 4.**Sort the eigenvectors from the highest eigenvalue to the lowest**. The eigenvector with the highest eigenvalue is the first principal component. Higher eigenvalues correspond to a larger amount of shared variance explained;
- 5.**Choose the number of principal components**. Select the top N eigenvectors (based on their eigenvalues) to become the N principal components;
- 6.****Recast the data along the principal component axes:**** using the feature vector formed by the eigenvectors of the covariance matrix, reorient the data from the original axes to the axes represented by the principal components. This can be done by multiplying the transpose of the original dataset by the transpose of the feature vector.

Q12: **Explain the difference between PCA (Principal Component Analysis) and ICA (Independent Component Analysis)?**

- Although the two methods may seem related, they perform different tasks. Specifically, PCA is often used to compress information, i.e. for dimensionality reduction, while ICA aims to separate information by transforming the input space into a maximally independent basis; ICA focuses more on independence, i.e. independent components

Q13: **How do you transform a skewed distribution into a normal distribution?**

- In machine learning problems, if the predictor variables and the target variable are normally distributed, the model tends to make more reliable predictions
- Common methods for transforming a skewed data distribution into a normal distribution are:
    - ****Log transformation: apply a log() transformation to all the data****
    - **Square root transformation**

Reference: [https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45](https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45)

Q14: **Which data imputation methods do you know?**

- For continuous data:
    - If the data is normally distributed, use the mean.
    - If the data is skewed or has many outliers, use the median.
- For categorical data:
    - If the data is ordinal (sortable), use the median.
    - If the data is not sortable, use the mode.
- For boolean features:
    - Use frequency imputation

![Untitled](feature-engineering/Untitled%203.png)

Q15: **In multiple linear regression, should highly correlated features be removed?**

- If two features are highly correlated, it is recommended that we keep only one of them in the dataset; this is because of the least-squares solution of linear regression: $w=(X^{T}X)^{-1}X^{T}y$. Assuming the dependent variable $y$ follows a normal distribution with variance $\delta$, the variance of the above weight vector is $\delta(X^{T}X)^{-1}$. For the model to be sufficiently stable, this variance should be low.
- From the singular value decomposition of the matrix, when there are highly correlated features, the variance of $w$ will be large, affecting the stability of the model

Reference: [https://datascience.stackexchange.com/questions/36404/when-to-remove-correlated-variables](https://datascience.stackexchange.com/questions/36404/when-to-remove-correlated-variables)

Q16: **How do you apply feature engineering in transfer learning?**

- If a model is trained on a sufficiently large and general dataset, it will effectively serve as a general-purpose model for a range of tasks. You can then use the feature maps learned by this large model to accomplish feature engineering;
- Simply add a new classifier, trained from scratch, on top of the pretrained model, so that the feature maps previously learned on the dataset can be reused

,

Q17: **Similarities and differences between standardization and normalization**

- Both are linear transformations that rescale the original data. Normalization generally rescales to [0,1], while standardization transforms the data to follow a standard normal distribution; the order of the data values does not change. Normalization is mainly affected by the two extreme values, whereas in standardization every data point has an influence, because the mean and standard deviation are computed.

Q18: **How do you understand combined features?**

- Combined features are really feature crosses. For example, in news browsing, if news is recommended to users based on the gender feature or the interest feature alone, the effect may not be particularly good; but observation shows that gender and interest are strongly related, e.g. women may prefer entertainment content while men prefer sports and finance. Combining these two features and adding them to the model works much better than single features.

Q19: **How do you handle high-dimensional combined features, such as user ID and content ID?**

- Use matrix factorization. Common matrix factorizations include
    - QR decomposition: Q is an orthogonal matrix, R is an upper triangular matrix
    - LU decomposition
    - SVD
    - Jordan decomposition: decomposes into multiple Jordan blocks, where each Jordan block has the same eigenvalue on its diagonal and 1s on its superdiagonal.
    

Q20: **What models are there for text feature representation? What are their pros and cons?**

- Bag-of-words model: the vocabulary needs careful design, especially to manage document size, which affects the sparsity of the document representation; it is simple but loses the contextual structure of words; for computational reasons (space and time complexity) and informational reasons, sparse representations are harder to model, because the model faces a huge challenge in exploiting so little information in such a vast representation space
- N-gram: groups n consecutive words into a phrase that is put into the vector representation as a single feature; the drawbacks are an overly large parameter space and data sparsity
- Topic models: can compute the topic distribution of each document
- Word embedding models: map words into K-dimensional vectors; can represent semantic relationships between words;

Q21: **What are the differences and connections between Word2vec and LDA?**

- LDA uses the co-occurrence relationships of words in documents to cluster words by topic, which can also be understood as factorizing the "document-word" matrix to obtain two probability distributions, "document-topic" and "topic-word". Word2Vec, on the other hand, actually learns the "context-word" matrix, where the context consists of several surrounding words, so the resulting word vector representations incorporate more of the contextual co-occurrence features. That is, if the Word2Vec vectors of two words are highly similar, they likely often appear in the same contexts.
- The biggest difference between topic models and word embeddings actually lies in the models themselves. A topic model is a generative model based on probabilistic graphical models, whose likelihood function can be written as a product of several conditional probabilities, including latent variables that need to be inferred (i.e. topics); a word embedding model is generally expressed in the form of a neural network, with the likelihood function defined on the network's output, and the dense vector representations of words are obtained by learning the network's weights.

Q22: **How do you detect outliers in features?**

- Outlier detection based on the Gaussian distribution: the 3σ rule is one kind of Gaussian-distribution-based method. Here an outlier is defined as a value whose deviation from the mean exceeds three standard deviations. Under the normality assumption, the region u+3σ contains 99.7% of the data; if a value is more than 3σ away from the mean of the distribution, it can simply be marked as an outlier: P(|x−μ|>3σ)≤0.003
- Quartiles
- In addition, there are methods that use various statistics to detect multivariate outliers, such as the x^2 test, t-test, etc.
- Matrix factorization methods based on principal component analysis: decompose with PCA, then reconstruct, and judge whether a point is anomalous by the fact that outliers have larger deviations on the principal components.
- Distance-based: using the idea of clustering, cluster the data and exclude the N points farthest from the centers; common methods include kmeans, knn, DBSCAN, etc.

Q23: **How do you handle outliers in features?**

- Delete the records containing outliers, or treat outliers as missing values and handle them with missing-value handling methods;
- Or correct them with the mean or the mode.

Reference: [https://blog.csdn.net/](https://blog.csdn.net/)

Q24: **How do you solve the problem of imbalanced data?**

- Sampling methods. Sampling is divided into oversampling and undersampling. Oversampling duplicates the minority class multiple times, while undersampling removes some samples from the majority class, or in other words selects only part of the majority-class samples. After oversampling, some samples appear repeatedly in the dataset, and the trained model will overfit to some extent; a solution is to add slight random perturbations each time a new data point is generated. The drawback of undersampling is also obvious: the final training set loses data and the model only learns part of it, which may lead to underfitting; a solution: undersample multiple times (sampling with replacement, so that the resulting training sets are independent of one another) to produce multiple different training sets, then train multiple different classifiers and use model ensembling to obtain the final result by combining the results of multiple classifiers.
- Data synthesis methods, which use the similarity of minority samples in the feature space to generate new samples; such methods have many success stories in small-data scenarios, such as medical image analysis.
- When the data is extremely imbalanced, you can observe the precision and recall of the training results of different algorithms on the same data. This has two benefits: first, you can understand how sensitive different algorithms are to the data; second, you can determine which evaluation metric is more appropriate. For data imbalance problems in machine learning, it is recommended to rely more on PR (Precision-Recall) curves rather than ROC curves; the specific reason becomes clear by plotting them: if the ROC curve is used as the evaluation metric, it is easy to overlook, due to a high AUC value, that the actual performance on the few minority samples is not ideal.
- Choose algorithms that are relatively insensitive to data skew, such as tree models.

Reference: [https://blog.csdn.net/](https://blog.csdn.net/)

Q25: **Briefly describe the principle of Linear Discriminant Analysis (LDA)**

- LDA is a supervised dimensionality reduction technique, meaning each sample in its dataset has a class output. This differs from PCA, which is an unsupervised dimensionality reduction technique that does not consider sample class outputs. The idea of LDA can be summarized in one sentence: "minimize the within-class variance and maximize the between-class variance after projection". We want to project the data onto a lower dimension such that, after projection, the projected points of each class are as close as possible, while the distances between the class centers of different classes are as large as possible.

## Signal Processing Feature Engineering (Physiological Signals & Audio)

Q26: **What are the sampling theorem (Nyquist) and aliasing? What effects does quantization have?**

- **Nyquist-Shannon sampling theorem**: if the highest frequency of a signal is $f_{max}$, the sampling rate must satisfy $f_s > 2f_{max}$ for distortion-free reconstruction; $f_s/2$ is called the **Nyquist frequency**.
- **Aliasing**: frequency components above $f_s/2$ "fold back" into lower frequencies as $f_{alias}=|f-kf_s|$, and after sampling they can no longer be distinguished or removed.
- **Anti-aliasing**: before sampling (ADC) or downsampling, a **low-pass filter (anti-aliasing filter)** must be applied, with a cutoff below the new Nyquist frequency; in practice the sampling rate is often chosen as 2.5–5 times the highest useful frequency to leave room for the filter's transition band.
- **Quantization**: mapping continuous amplitudes to discrete values with a finite number of bits ($N$ bits) introduces **quantization noise**; for a full-scale sine wave the theoretical signal-to-noise ratio is about $SNR\approx 6.02N+1.76\ \text{dB}$. Too few bits or an improper gain setting can bury weak signals (e.g. EEG at the µV level) or cause **clipping** of large signals.

Q27: **What is the difference between FIR and IIR filters? What are the characteristics of the Butterworth filter?**

- **FIR (finite impulse response)**: the output depends only on current and past inputs, $y[n]=\sum_{k=0}^{M}b_k x[n-k]$; it is always stable and can be designed to have **exactly linear phase** (no waveform distortion, only a constant delay). The drawback is that a steep transition band requires a high order, which means more computation and longer delay.
- **IIR (infinite impulse response)**: contains feedback terms, $y[n]=\sum_k b_k x[n-k]-\sum_{k\ge1} a_k y[n-k]$; a very low order can achieve a steep frequency response, but the phase is **nonlinear**, and high orders or coefficient quantization can make it **unstable** (in practice it is often implemented as second-order sections, SOS).
- **Butterworth**: **maximally flat** passband with no ripple and a monotonically decreasing magnitude response; an $n$-th order filter rolls off at about $20n\ \text{dB/decade}$ in the stopband. Its transition band is less steep than Chebyshev / elliptic filters, but it introduces no passband ripple, which makes it the most commonly used IIR filter for physiological signal preprocessing.

Q28: **In which scenarios are band-pass and notch filters used? Why is zero-phase filtering (filtfilt) commonly used?**

- **Band-pass filtering**: keeps the frequency band of interest while removing low-frequency drift and high-frequency noise, e.g. ECG 0.5–40 Hz, EEG 0.5–45 Hz, sEMG 20–450 Hz, PPG 0.5–8 Hz.
- **Notch filtering**: strong attenuation in a very narrow band, mainly used to remove **50/60 Hz powerline interference** (and its harmonics if necessary); the bandwidth is controlled by the quality factor $Q=f_0/\Delta f$.
- **Zero-phase filtering (filtfilt)**: filter forward once, then reverse the result and filter again, so that the phase distortions cancel out and the result has **zero phase**, avoiding time shifts of waveform features such as R-peaks or event-related potentials; the equivalent magnitude response is $|H(f)|^2$ (attenuation is doubled, equivalent to doubling the order).
- Limitation: filtfilt is **non-causal** and needs future samples, so it can only be used for offline analysis; real-time/online systems should use causal filters and compensate for their group delay.

Q29: **What is spectral leakage? Why does windowing mitigate it?**

- Taking the DFT of a finite-length signal implicitly applies a **rectangular window** and assumes the signal is periodically extended; if the segment length is not an integer multiple of the signal period, discontinuities appear at the boundaries and energy "leaks" from the true frequency into neighboring bins — this is **spectral leakage**.
- **Windowing**: multiplying the signal by a window function that tapers smoothly to zero at both ends (Hann, Hamming, Blackman, etc.) reduces the boundary discontinuity and thus **lowers the sidelobes**.
- Trade-off: **main-lobe width** (frequency resolution) and **sidelobe level** (amount of leakage) constrain each other. The rectangular window has the narrowest main lobe but a first sidelobe of about $-13\ \text{dB}$; Hann is about $-31\ \text{dB}$; Hamming about $-43\ \text{dB}$; Blackman has even lower sidelobes but a wider main lobe.
- Zero-padding only makes the spectrum curve "smoother" (interpolation); it does **not** improve the true frequency resolution.

Q30: **What is the difference between FFT and STFT? How should the time-frequency resolution trade-off be understood?**

- **FFT** is a fast algorithm for computing the DFT (complexity $O(N\log N)$). It gives the global spectrum of the whole signal and loses information about "when a frequency component occurs", so it is only suitable for stationary signals.
- **STFT (short-time Fourier transform)**: splits the signal into short frames (windowed, overlapping) and applies an FFT to each frame, producing a **time-frequency representation (spectrogram)**, suitable for non-stationary signals such as speech and EEG:
$$X(m,\omega)=\sum_n x[n]\,w[n-mH]\,e^{-j\omega n}$$
where $H$ is the hop size.
- **Time-frequency resolution trade-off**: frequency resolution is $\Delta f = f_s/N = 1/T_{win}$. A longer window gives better frequency resolution but worse time resolution, and a shorter window the opposite. Constrained by the uncertainty principle $\Delta t\cdot\Delta f\ge \frac{1}{4\pi}$, both cannot be made arbitrarily fine at the same time. STFT uses a fixed window length for all frequencies, so its resolution is uniform across the time-frequency plane.

Q31: **What advantages do wavelet transforms (CWT/DWT) have over STFT?**

- The wavelet transform uses basis functions obtained by scaling and shifting a **mother wavelet**: $W(a,b)=\frac{1}{\sqrt{a}}\int x(t)\,\psi^*\!\left(\frac{t-b}{a}\right)dt$, where $a$ is the scale and $b$ the translation.
- **Multi-resolution analysis**: short windows at high frequencies (good time resolution) and long windows at low frequencies (good frequency resolution). This naturally suits signals of the "slow low-frequency + high-frequency transient" type, such as the QRS complex in ECG, spikes in EEG and transients in speech.
- **CWT (continuous wavelet transform)**: scales take continuous values; it is highly redundant and computationally expensive, and is mainly used for time-frequency visualization (e.g. a Morlet wavelet scalogram, which can be used as input to a 2D CNN).
- **DWT (discrete wavelet transform)**: dyadic scales, implemented by a low-pass/high-pass filter bank + downsampling. Each level decomposes into **approximation coefficients** (low frequency) and **detail coefficients** (high frequency), with the band halved at each level; it is computationally efficient and non-redundant. It is commonly used for **denoising** (thresholding the detail coefficients), baseline wander removal and compression, and the energy, standard deviation, entropy, etc. of each sub-band can be used as features. Common mother wavelets include db4, sym and coif.
- Drawbacks: the mother wavelet and the number of decomposition levels must be chosen; DWT is not shift-invariant (this can be mitigated with the stationary wavelet transform, SWT).

Q32: **How is the power spectral density (PSD) estimated? What is the principle of Welch's method?**

- **Periodogram**: $P(f)=\frac{1}{N}|X(f)|^2$, i.e. take the FFT of the whole signal and compute the squared magnitude. It is an **inconsistent estimator**: even as the data length increases, its variance does not decrease, so the spectral estimate is very noisy.
- **Welch's method**: split the signal into several **overlapping** segments (commonly 50%), window each segment and compute its periodogram, then **average** over all segments. Averaging $K$ segments reduces the variance by roughly a factor of $K$, at the cost of lower frequency resolution because each segment is shorter — a classic **bias-variance trade-off**.
- Other methods: the multitaper method (uses multiple orthogonal DPSS tapers to reduce variance) and parametric AR-model spectral estimation (suitable for short data, gives a smoother spectrum).
- PSD is the basis for features such as EEG band power, HRV frequency-domain indices and EMG median frequency.

Q33: **What should be noted about resampling, detrending and normalization in signal preprocessing?**

- **Resampling**: when fusing data from multiple devices or datasets, the sampling rate needs to be unified. Before **downsampling**, anti-aliasing low-pass filtering is mandatory (e.g. `scipy.signal.decimate`); for rational-factor conversion, **polyphase filtering** (`resample_poly`) can be used, which is efficient with small boundary effects; the FFT-based `resample` assumes the signal is periodic by default, which may cause ringing at the boundaries.
- **Detrending**: removing linear trends or slow drifts (e.g. electrode polarization, body-temperature drift), using polynomial fitting and subtraction, high-pass filtering or median filtering; otherwise low-frequency energy will dominate the spectrum.
- **Normalization**:
    - z-score (per channel, per window or per subject) removes inter-individual amplitude differences (e.g. those caused by electrode impedance or skin thickness);
    - min-max is sensitive to outliers; robust scaling (median/IQR) is more robust;
    - **Key point**: statistics may only be estimated on the training set, or using only the subject's own calibration data; otherwise this causes **data leakage**.
- Note: normalization erases **absolute amplitude information**, which some tasks (e.g. EMG force estimation, absolute EEG power) actually need.

Q34: **How are long signals segmented (sliding windows)? Which statistical features are commonly extracted from windows?**

- **Segmentation**: cut the continuous signal into fixed-length windows. The window length should cover the time scale of the target phenomenon (e.g. 30 s for EEG sleep staging, 150–250 ms for EMG gesture recognition, 5 min for HRV analysis), and the step size usually corresponds to an **overlap ratio** such as 50%, to increase the number of samples and temporal smoothness. Windows that are too long increase latency and mix multiple states; windows that are too short give unstable feature estimates.
- **Label assignment**: majority vote within the window, the label at the end of the window, or discarding windows that cross state boundaries.
- **Time-domain statistical features**: mean, variance/standard deviation, RMS, max/min, peak-to-peak, percentiles, **skewness** (asymmetry of the distribution), **kurtosis** (tail heaviness/peakedness), zero-crossing rate, and the Hjorth parameters (Activity, Mobility, Complexity).
- **Frequency-domain features**: band power, relative power, spectral centroid, spectral entropy, dominant frequency.
- Note: overlapping windows are highly correlated. If you window first and then randomly split into training/test sets, adjacent windows end up in the training and test sets respectively, causing **severe leakage**.

Q35: **What are the common entropy features? How is sample entropy computed?**

- Entropy measures the **complexity/regularity** of a signal and is widely used in fatigue detection, depth-of-anesthesia monitoring, epilepsy detection and HRV analysis.
- **Shannon entropy / spectral entropy**: compute $H=-\sum p_i\log p_i$ on the normalized amplitude distribution or the PSD; the lower the spectral entropy, the more the energy is concentrated in a few frequencies.
- **Approximate entropy (ApEn)**: compares the template match rates for lengths $m$ and $m+1$, but counts **self-matches**, so it is biased and depends on data length.
- **Sample entropy (SampEn)**: $SampEn(m,r)=-\ln\frac{A}{B}$, where $B$ is the number of pairs of length-$m$ templates (**excluding self-matches**) that match within tolerance $r$ (Chebyshev distance), and $A$ is the number of matches for length $m+1$. Commonly $m=2$, $r=0.2\cdot\text{std}$. A larger value means a more irregular signal; compared with ApEn it is less biased and less sensitive to data length, but its complexity is $O(N^2)$.
- **Permutation entropy**: based on the distribution of ordinal patterns of the sequence; fast to compute and robust to noise. **Multiscale entropy**: computes sample entropy separately at multiple coarse-grained time scales.

Q36: **What are the common noise sources in ECG preprocessing? How are they removed?**

- **Baseline wander**: caused by respiration and body movement, usually below 0.5 Hz. It can be removed with a high-pass filter (0.5 Hz for monitoring; for diagnostic quality, to preserve the ST segment, use 0.05 Hz or a linear-phase filter), by estimating the baseline with a two-stage median filter and subtracting it, or by removing the low-frequency approximation component of a wavelet decomposition.
- **Powerline interference**: 50/60 Hz and its harmonics, removed with a notch filter or adaptive filtering.
- **Muscle (EMG) noise**: broadband high-frequency noise, which can be reduced with low-pass filtering (e.g. 40 Hz) or wavelet threshold denoising; since it overlaps with the QRS band, it is hard to remove completely.
- **Motion artifacts / poor electrode contact**: appear as large-amplitude transients or saturation; usually a **signal quality index (SQI)** is computed and low-quality segments are simply discarded rather than forcibly filtered.
- To preserve waveform morphology such as QRS/ST, offline analysis generally uses zero-phase filtering.

Q37: **Briefly describe the Pan-Tompkins R-peak detection algorithm**

- This is a classic real-time QRS detection algorithm. Its main steps are:
    - **Band-pass filtering** (about 5–15 Hz): emphasizes QRS energy and suppresses P/T waves, baseline wander and powerline noise;
    - **Differentiation**: obtains slope information of the QRS;
    - **Squaring**: makes all values positive and nonlinearly amplifies the high-slope parts;
    - **Moving-window integration** (window of about 150 ms): yields the energy envelope of the QRS;
    - **Adaptive dual-threshold decision**: maintains running estimates of signal peaks and noise peaks separately and updates the thresholds dynamically; a **200 ms refractory period** is used to avoid double detections; if no beat is detected for more than about 1.66 times the average RR interval, a **search-back** with a lower threshold is performed;
    - Finally, the R-peak is precisely located near the corresponding position in the raw (or filtered) signal.
- Modern alternatives include wavelet-based detection, the Hamilton algorithm and deep-learning detectors; in engineering practice, libraries such as NeuroKit2 and BioSPPy can be used directly.

Q38: **What are the time-domain and frequency-domain features of HRV? What should be noted when computing them?**

- The input is the sequence of intervals between adjacent R-peaks (RR/NN intervals).
- **Time-domain features**:
    - **SDNN**: standard deviation of NN intervals, reflecting overall variability;
    - **RMSSD**: $\sqrt{\frac{1}{N-1}\sum_{i}(RR_{i+1}-RR_i)^2}$, reflecting short-term (beat-to-beat) variability, mainly reflecting **parasympathetic (vagal)** activity;
    - **pNN50**: the proportion of successive interval differences greater than 50 ms; plus mean heart rate, etc.
- **Frequency-domain features**: VLF (0.003–0.04 Hz), **LF (0.04–0.15 Hz)**, **HF (0.15–0.4 Hz, related to respiratory sinus arrhythmia, reflecting vagal activity)**, and **LF/HF** (traditionally interpreted as sympathovagal balance, though this interpretation is controversial).
- **Caveats**:
    - The RR series is **unevenly sampled**, so it must first be interpolated and resampled (e.g. 4 Hz cubic spline) before applying Welch, or the Lomb-Scargle periodogram can be used directly;
    - **Ectopic beats** and false detections must be removed/corrected first; otherwise indices such as RMSSD will be severely overestimated;
    - The indices depend on recording length (the standard for short-term HRV is 5 min), so results from different lengths are not directly comparable; they are also affected by breathing rate and posture.
- Nonlinear indices: SD1/SD2 of the Poincaré plot, DFA $\alpha_1$, sample entropy, etc.

Q39: **How are EEG frequency bands divided? How are band power features extracted?**

- **Common bands** (boundaries vary slightly across the literature):
    - **delta** 0.5–4 Hz: deep sleep;
    - **theta** 4–8 Hz: drowsiness, memory-related;
    - **alpha** 8–13 Hz: relaxed with eyes closed, most prominent over the occipital region;
    - **beta** 13–30 Hz: alertness, movement-related (mu/beta desynchronization, ERD, occurs during motor imagery);
    - **gamma** >30 Hz: higher cognition, but easily contaminated by muscle activity.
- **Band power extraction**: estimate the PSD of each channel with Welch and integrate within the band to obtain **absolute power**; divide by total power to obtain **relative power** (which reduces inter-individual amplitude differences); usually take the $\log$ to make the distribution closer to normal.
- **Derived features**: band power ratios (e.g. theta/beta for attention assessment), inter-hemispheric **asymmetry** (e.g. frontal alpha asymmetry), **differential entropy** (for a Gaussian signal $DE=\frac{1}{2}\log(2\pi e\sigma^2)$, commonly used in emotion recognition), inter-channel coherence/phase-locking value (connectivity features), and **CSP (common spatial patterns)** spatial-filter features commonly used in motor imagery.

Q40: **What are the common artifacts in EEG? How is ICA used to remove them? What is epoching?**

- **Common artifacts**:
    - **EOG** (blinks, eye movements): low-frequency, large-amplitude, mainly over the frontal region;
    - **EMG** (jaw clenching, neck muscles): high-frequency (>20 Hz) broadband noise, concentrated over the temporal regions;
    - **ECG**, powerline interference, poor electrode contact (electrode pop), slow drifts caused by sweating.
- **ICA artifact removal workflow**:
    - First apply high-pass filtering (about 1 Hz, to improve ICA decomposition quality) and remove bad channels;
    - Decompose the multichannel EEG into statistically independent components $x=As$;
    - Identify artifact components from their scalp topography, time course and spectrum (or with the help of the ICLabel automatic classifier, or correlation with EOG reference channels);
    - Set the artifact components to zero and back-project to reconstruct the clean signal.
- ICA assumes the sources are statistically independent and linearly mixed, with enough channels; it may also remove part of the neural signal by mistake. Other methods include regressing out EOG reference channels and ASR (artifact subspace reconstruction).
- **Epoching**: cut fixed time windows anchored at events (stimuli, key presses), e.g. from $-200$ ms to $800$ ms, use the pre-event interval for **baseline correction**, and then reject bad trials by an amplitude threshold (e.g. ±100 µV); this is a standard step in ERP analysis and event-related classification. In addition, a reference scheme (e.g. average reference) usually needs to be chosen.

Q41: **What are the difficulties of PPG signal processing? How are heart rate and blood oxygen (SpO2) estimated from it?**

- **PPG** measures changes in blood volume optically: the **AC component** is the pulsation caused by heartbeats, and the **DC component** reflects baseline absorption by tissue, venous blood, etc.
- **Main difficulty — motion artifacts**: in wrist-worn devices, the frequency of artifacts caused by arm movement (e.g. step frequency 1–3 Hz) overlaps with the heart-rate band, so simple band-pass filtering cannot separate them. Common methods:
    - Adaptive filtering (LMS/RLS) using a synchronized **accelerometer** as the reference signal;
    - Spectral subtraction and spectral peak tracking (e.g. TROIKA);
    - Discarding poor segments by a signal quality index (SQI);
    - Deep learning models that fuse PPG and accelerometer information.
- **Heart rate estimation**: after band-pass filtering (about 0.5–4 Hz), obtain pulse intervals by peak detection, or find the dominant frequency in the PSD; one can also extract pulse rate variability (PRV), pulse waveform morphology features (rise time, dicrotic notch), and the **pulse transit time (PTT)** computed jointly with ECG (used for cuffless blood pressure estimation).
- **SpO2**: exploits the different absorption of red light (~660 nm) and infrared light (~940 nm) by oxygenated and deoxygenated hemoglobin, and computes the **ratio of ratios**:
$$R=\frac{AC_{red}/DC_{red}}{AC_{IR}/DC_{IR}}$$
which is then mapped to SpO2 through an empirical calibration curve (a common approximation is $SpO2\approx 110-25R$; in practice it must be calibrated per device). The measurement is affected by motion, low perfusion and skin tone.

Q42: **Which features are commonly used for surface EMG (sEMG)?**

- **Preprocessing**: band-pass 20–450 Hz (sampling rate usually ≥1000 Hz), notch filtering to remove powerline noise; envelope extraction by rectification followed by low-pass filtering.
- **Time-domain features** (the classic Hudgins feature set):
    - **MAV** (mean absolute value): $\frac{1}{N}\sum|x_i|$, related to the level of muscle activation;
    - **RMS**: $\sqrt{\frac{1}{N}\sum x_i^2}$, approximately monotonically related to muscle force;
    - **WL** (waveform length): $\sum|x_{i+1}-x_i|$, reflecting signal complexity;
    - **ZC** (zero crossings) and **SSC** (slope sign changes), usually used with a noise threshold;
    - In addition, IEMG (integrated EMG), AR model coefficients, etc.
- **Frequency-domain features**:
    - **Mean frequency** $MNF=\frac{\sum f_iP_i}{\sum P_i}$;
    - **Median frequency MDF**: the frequency that splits the power spectrum into two halves of equal energy, i.e. $\sum_{f\le MDF}P=\frac{1}{2}\sum P$.
    - When a muscle **fatigues**, conduction velocity decreases and MNF/MDF shift toward lower frequencies, which is the classic fatigue indicator.
- In gesture recognition, windows of 150–250 ms with 50% or more overlap are commonly used to meet real-time requirements (total latency <300 ms).

Q43: **How can data leakage be avoided in physiological signal modeling? How should class imbalance be handled?**

- **Inter-subject variability** is very large (physiology, electrode placement, skin impedance). If windows are split randomly, samples from the same subject (or even adjacent overlapping windows from the same recording) appear in both the training and test sets, so the model learns to "recognize the person" rather than the task itself, leading to severely inflated results.
- **Correct practice**:
    - **Subject-wise split**, using GroupKFold or **leave-one-subject-out cross-validation (LOSO)**;
    - For within-subject evaluation, split by time and leave a gap between the training and test segments;
    - Steps such as normalization, feature selection, ICA templates and oversampling must only be fitted within the training fold (put them in a Pipeline);
    - Clearly report the evaluation protocol (cross-subject vs within-subject); the two are not comparable.
- **Class imbalance** (e.g. arrhythmias, seizures and sleep stage N1 are rare):
    - Use **class-weighted loss**, Focal Loss, or **stratified/balanced sampling**;
    - Oversample/SMOTE only on the training set; data augmentation at the signal level is usually more sensible than SMOTE in feature space;
    - Evaluate with F1, balanced accuracy, Cohen's kappa, AUPRC, and event-level metrics (e.g. false alarms per hour), rather than plain accuracy;
    - The decision threshold may also need to be adjusted according to the prior class ratio at deployment.

Q44: **What are the pros and cons of handcrafted features + traditional models versus end-to-end 1D-CNNs (on raw signals)?**

- **Handcrafted features + traditional models** (e.g. HRV/band power + random forest/XGBoost/SVM):
    - Pros: robust in **small-data** scenarios, **interpretable** (clinically meaningful, easier for regulatory approval), low computational cost, easy to deploy on embedded devices;
    - Cons: depend on domain knowledge, features may miss useful information, and the whole pipeline is vulnerable to errors in one step (e.g. R-peak detection errors).
- **End-to-end deep models** (1D-CNN, ResNet, CNN+LSTM/Transformer, taking raw signals or time-frequency maps as input):
    - Pros: automatically learn morphological and temporal patterns, and usually perform better with large amounts of data (e.g. single-lead ECG arrhythmia classification, DeepSleepNet for sleep staging);
    - Cons: require large amounts of labeled data, easily overfit to subject or device characteristics, poor interpretability, and are sensitive to changes in sampling rate and lead configuration.
- **Compromise approaches**:
    - Use **time-frequency maps (STFT/CWT) + 2D CNN**;
    - Use convolutional layers with interpretable priors (e.g. SincNet with learnable band-pass filters, EEGNet combining depthwise separable convolutions and spatial filtering);
    - Concatenate and fuse handcrafted features with deep features;
    - Self-supervised pretraining on large-scale data first, then fine-tuning on the target task.
- The choice should be based on data size, interpretability requirements and deployment resources; whichever approach is used, subject-independent evaluation is required.

Q45: **Describe the MFCC extraction pipeline in detail**

- 1.**Pre-emphasis**: $y[n]=x[n]-\alpha x[n-1]$, $\alpha\approx0.95\sim0.97$, compensating for the attenuation of high-frequency components in speech (the glottal spectrum falls at about $-6$ dB/octave) and balancing the spectrum;
- 2.**Framing**: speech is short-time stationary; frame length is generally 20–40 ms (commonly 25 ms) with a 10 ms hop;
- 3.**Windowing**: each frame is multiplied by a Hamming window to reduce spectral leakage;
- 4.**FFT and power spectrum**: apply an $N$-point FFT (e.g. 512) and compute the power spectrum $P=\frac{1}{N}|X(k)|^2$;
- 5.**Mel filterbank**: weight and sum with 20–40 triangular filters equally spaced on the **Mel scale**, mimicking the human ear's high resolution at low frequencies and low resolution at high frequencies: $m=2595\log_{10}\left(1+\frac{f}{700}\right)$;
- 6.**Logarithm**: take the $\log$ of each filter's energy, compressing the dynamic range in line with the ear's logarithmic perception of loudness, and turning convolutional channel effects into additive ones;
- 7.**DCT (discrete cosine transform)**: apply DCT-II to the log-Mel energies to decorrelate them, keeping the first 12–13 coefficients (the 0th coefficient is often replaced by the log energy) to obtain the cepstral coefficients; the higher-order coefficients correspond to fine structure (pitch harmonics) and are discarded;
- 8.**Optional post-processing**: liftering, **cepstral mean and variance normalization (CMVN)** to remove channel effects, and concatenation of first/second-order deltas, forming the common 39-dimensional feature.

Q46: **What is the difference between the log-Mel spectrogram and MFCC? Why do deep models prefer log-Mel? What are delta features?**

- **Difference**: MFCC = log-Mel + DCT + truncation. The DCT makes the feature dimensions **approximately decorrelated** and compresses the dimensionality, which suits traditional models such as **diagonal-covariance GMM-HMMs** very well.
- **Why deep models prefer log-Mel**:
    - The DCT destroys the **locality** of the spectrum — adjacent Mel bands are adjacent in log-Mel, and a CNN can exploit this local time-frequency correlation and shift invariance (e.g. a pitch change appears as a shift along the frequency axis), whereas DCT coefficients mix all bands;
    - Truncation discards part of the information (e.g. harmonic structure);
    - Deep networks can learn decorrelation and dimensionality reduction themselves, so the handcrafted DCT is unnecessary.
    - Therefore modern models for ASR, sound event detection and speaker recognition (e.g. Whisper uses 80/128-dim log-Mel) mostly use log-Mel (filterbank, Fbank) features.
- **Delta features (Delta / Delta-Delta)**: describe how features change over time, commonly computed with the regression formula:
$$d_t=\frac{\sum_{n=1}^{N}n\,(c_{t+n}-c_{t-n})}{2\sum_{n=1}^{N}n^2}$$
usually with $N=2$; applying the delta once more to $d_t$ gives the second-order delta. 13-dim MFCC + Δ + ΔΔ = 39 dimensions. For models with temporal modeling capacity (RNN/Transformer), the gain from delta features is usually small.

Q47: **Which handcrafted audio features are commonly used?**

- **Time-domain features**:
    - **Zero-crossing rate (ZCR)**: the number of times the signal crosses zero per unit time; unvoiced sounds/noise have a high ZCR and voiced sounds a low one, so it can be used for voiced/unvoiced discrimination and percussion detection;
    - **Short-time energy / RMS**: reflects loudness, used for silence detection.
- **Spectral shape features**:
    - **Spectral centroid** $\frac{\sum_k f_k|X(k)|}{\sum_k|X(k)|}$: the "center of mass" of the spectrum, related to the "brightness" of timbre;
    - **Spectral rolloff**: the frequency below which the cumulative energy reaches 85% or 95% of the total energy;
    - **Spectral flux**: the sum of squared differences between the (normalized) spectra of adjacent frames, $\sum_k(|X_t(k)|-|X_{t-1}(k)|)^2$, used for onset detection;
    - In addition: spectral flatness (distinguishing noise from tones), spectral bandwidth, spectral contrast, etc.
- **Music features**: **Chroma**: folds energy into 12 semitone pitch classes (C, C#, …, B), ignoring the octave; commonly used for chord recognition and cover-song detection, usually computed from the CQT.
- **Fundamental frequency F0 / pitch**: can be estimated with autocorrelation, the cepstrum method, **YIN / pYIN** (difference function + cumulative mean normalization) or the neural method CREPE; F0 and its statistics (mean, range, jitter), shimmer and HNR are important features for emotion recognition and pathological voice detection.
- Statistics such as mean, standard deviation and percentiles can then be computed over frame-level features to obtain a fixed-length segment-level vector (e.g. the eGeMAPS feature set of openSMILE).

Q48: **What is voice activity detection (VAD)? What implementation methods are there?**

- **VAD** determines whether each frame is speech or non-speech (silence/noise). It is a front-end step for ASR, speaker recognition, speech coding and denoising, reducing computation and preventing models from learning on silent segments.
- **Implementation methods**:
    - **Energy + zero-crossing-rate thresholding**: simple and fast, but performs poorly at low SNR;
    - **Spectral-feature-based**: spectral entropy (speech spectra are highly structured with low entropy), spectral flatness, long-term spectral envelope, etc.;
    - **Statistical models**: model speech and noise separately and perform a likelihood ratio test (e.g. WebRTC VAD uses GMMs);
    - **Neural VAD**: e.g. Silero VAD and the segmentation model in pyannote, which are more robust in noisy environments.
- **Post-processing**: add a **hangover** mechanism and minimum speech/silence duration constraints to smooth frame-level decisions, avoiding cutting off word endings or frequent decision flipping.
- Key trade-off: the balance between missed detections (losing speech) and false alarms (keeping noise); the threshold should be set according to the downstream task.

Q49: **What data augmentation methods are there for audio (and physiological signals)?**

- **SpecAugment** (operates directly on the log-Mel spectrogram): **time warping**, **frequency masking** (randomly zeroing $f$ consecutive Mel bands), **time masking** (randomly zeroing $t$ consecutive frames); low computational cost and significantly improves ASR robustness.
- **Waveform-level augmentation**:
    - **Speed perturbation** (e.g. 0.9/1.0/1.1× speed, changing both duration and pitch);
    - **Time-stretch**: uses a phase vocoder to change duration without changing pitch;
    - **Pitch-shift**: changes pitch without changing duration;
    - **Noise addition**: mixing in background noise at a specified SNR (e.g. the MUSAN noise corpus);
    - **Reverberation**: convolution with room impulse responses (RIR);
    - Gain changes, random cropping, codec/bandwidth simulation, etc.
- **Mixup**: linearly mixes two samples with $\lambda\sim Beta(\alpha,\alpha)$, mixing the labels in the same proportion; commonly used for sound event classification.
- **Physiological signal augmentation**: adding Gaussian noise (jitter), amplitude scaling, time warping, random window cropping/shifting, channel dropout, baseline wander simulation, segment permutation, etc.
- **Note**: augmentation must not change the label semantics — for example, pitch-shift changes speaker characteristics or musical key; time-stretching an ECG changes the heart rate, which changes the label in heart-rate-related tasks. Augmentation should only be applied to the training set.

Q50: **How do self-supervised representations such as wav2vec 2.0 / HuBERT compare with handcrafted features (MFCC/Fbank)?**

- **wav2vec 2.0**:
    - Takes the **raw waveform** as input; a multi-layer 1D-CNN feature encoder produces latent representations at about 20 ms per frame, which are fed into a Transformer context network;
    - During pretraining, part of the latent frames are **masked**, discrete targets are obtained with **product quantization (Gumbel-softmax)**, and the model is trained with a **contrastive loss** (identifying the true quantized representation among distractors) plus a diversity loss.
- **HuBERT**:
    - First generates pseudo-labels by offline **k-means clustering** of features (MFCC in the first iteration, then re-clustering intermediate-layer features of the previous model), and performs **masked prediction** (cross-entropy) on the masked frames; training is more stable than contrastive learning;
    - **WavLM** builds on this by adding denoising and speaker mixing, improving performance on speaker- and noise-related tasks.
- **Comparison with handcrafted features**:
    - Pros: learn hierarchical representations from massive unlabeled speech (shallow layers lean toward acoustic/speaker information, middle layers toward phonetic/semantic information), and far outperform MFCC/Fbank in **low-resource** labeling scenarios; the model can be frozen and a weighted sum of layer representations (the SUPERB approach) fed to a lightweight downstream head, or the whole model can be fine-tuned;
    - Cons: large models (about 95M–1B parameters), slow inference, high memory requirements; usually pretrained on 16 kHz speech, so transferring to non-speech (e.g. music, bioacoustics, heart sounds) or different sampling rates suffers from **domain mismatch**; poor interpretability.
- **Practical advice**: with little data and a speech-related target task, try self-supervised representations first; when resources are limited or real-time operation or interpretability is required, Fbank/MFCC + a small model remains a strong baseline. In the physiological signal domain there are similar ideas as well (e.g. contrastive learning or masked-reconstruction pretraining on large-scale EEG/ECG).
