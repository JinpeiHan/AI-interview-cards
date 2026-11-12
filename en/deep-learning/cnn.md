# CNN

Q1: ***What is the difference between CNN and RNN? When should each be used?***

- CNNs are best suited to situations requiring translation invariance. Translation invariance means that no matter where the target appears in the image, the same features are detected and the same response is output.
- RNNs are neural networks that can remember historical input state information; they remember past input samples and use them to help classify the current input sample. RNNs are suitable when the order of the data matters. Therefore RNNs are often used in speech, video (frames are ordered) and text processing. Generally speaking, problems related to time-series data (data with timestamps) are well suited to RNNs.

Q2: ***How can CNNs be used for time-series forecasting?***

- The ability of **CNNs** to learn and automatically extract features from raw input data can be applied to time-series forecasting problems. A sequence of observations can be treated as a one-dimensional image, which a CNN model can read and extract features from, creating an informative representation of the time series
- Common examples of using CNNs for time-series forecasting include TCN and WaveNet; they are highly noise-robust models and can extract information-rich deep features

Q3: ***In CNNs, what are the pros and cons of Max Pooling vs. Average Pooling?***

- Errors in the convolutional-layer parameters cause a shift in the estimated mean, which max pooling can reduce; the limited neighborhood size increases the variance of the estimate, which average pooling can reduce. In other words, average pooling preserves the background better, while max pooling is better at extracting texture. For tasks such as digit recognition, max pooling is generally used.

Reference: [https://www.zhihu.com/question/34898241](https://www.zhihu.com/question/34898241)

Q4: ***Compare CNNs and multilayer perceptrons (MLP)***

- An MLP consists of fully connected layers, with each neuron connected to **all** nodes in the previous layer, leading to parameter redundancy; in contrast, thanks to weight sharing, a CNN has fewer parameters, which makes the network easier to train and makes it easier to design deep networks;
- An MLP only accepts vector inputs, **losing the spatial information between pixels; a CNN accepts matrix and vector inputs and can exploit the spatial relationships between pixels**
- An MLP is a special case of a CNN: **when the CNN kernel size equals the input size, its computation is equivalent to an MLP**

Reference: [https://medium.com/data-science-bootcamp/multilayer-perceptron-mlp-vs-convolutional-neural-network-in-deep-learning-c890f487a8f1](https://medium.com/data-science-bootcamp/multilayer-perceptron-mlp-vs-convolutional-neural-network-in-deep-learning-c890f487a8f1)

*Q5*: *****What is the role of the *fully connected layer* in a *CNN*?**  

- The fully connected layer acts as the "classifier" in the whole convolutional neural network. If operations such as convolutional layers, pooling layers and activation functions map the raw data into a hidden feature space (the process of feature extraction + selection), the fully connected layer maps the learned feature representation to the label space of the samples. In other words, it integrates the features together (highly refining the features), making it convenient to pass them to the final classifier or regressor.



*Q6*: *****Explain the significance of the ReLU activation function in convolutional neural networks*****

- A ReLU operation is applied after each convolution operation. ReLU is a non-linear activation function. This operation is applied to each pixel and replaces all negative pixel values in the feature map with zero. The convolution operation itself is linear; to model image features with non-linear variation, a non-linear function such as ReLU is needed to introduce non-linearity into the model.

Q7: ***Given the input image size, filter size, stride and padding size, what is the size of the feature map?***

- If the input image size is nxn, the filter size is fxf, p is the padding size and s is the stride, then the dimension of the feature map is: **floor[ ((n-f+2p)/s)+1] x floor[ ((n-f+2p)/s)+1]**

Q8: ***Explain Valid Padding and Same Padding***

- valid padding: **the convolution is computed only when the filter lies entirely inside the image**
- same padding: **the convolution starts when the center (K) of the filter coincides with the corner of the image; the output feature map after convolution keeps the same size**

Q9: ****What are the different types of pooling? Describe their characteristics.****

- **max pooling: returns the maximum value of the region covered by the kernel and suppresses noise**
- **average pooling: computes the average value of the feature map region covered by the kernel**
- sum pooling: computes the sum of all elements in the kernel window
- The most widely used pooling technique is **max pooling**, because it captures the most important features.

Q12: **Explain the role and function of the flatten layer in a CNN**

- After a series of convolution and pooling operations on the image's feature representation, we flatten the output of the final pooling layer into one long continuous linear array or vector. The process of converting all the resulting two-dimensional arrays into a vector is called Flattening.
The flatten output is fed as input to a fully connected neural network with a varying number of hidden layers, to learn the non-linear complexities present in the feature representation.

Q13: ****List the hyperparameters of a pooling layer****

- The hyperparameters of a pooling layer include filter size, stride and max/average pooling; if the input dimension of the pooling layer is $n_{h}*n_{w}*n_{c}$, the output dimension will be $((n_{h}-f)/s+1)*((n_{w}-f)/s+1)*n_{c}$

Q14: ****Explain the significance of "parameter sharing" and "sparse connectivity" in CNNs****

- Parameter sharing: in convolution, parameters are shared while convolving over the input. The intuition behind this is that a feature detector useful for one part of the image is probably also useful for another part of the image. Therefore, by using a single convolution kernel, we convolve over the whole input, so the parameters are shared.
- Sparse connectivity: for each layer, each output value depends only on a small number of inputs rather than on all inputs.

Q15: **Can a CNN be used to perform dimensionality reduction? If so, which sub-layer in a CNN performs the dimensionality reduction?**

- CNNs can be used for dimensionality reduction
- The pooling layer: the main goal of the pooling layer is to reduce the spatial dimensions of the CNN. To reduce the spatial dimensions, it performs a down-sampling operation, creating a pooled feature map by sliding a kernel matrix over the input matrix.

Q16: ***What are the roles of 3×3 and 1×1 convolutions, and how do they differ?***

- **3×3 convolution**: aggregates information from a $3\times3$ spatial neighborhood, performing spatial feature extraction and channel fusion at the same time; it is the most commonly used kernel in modern CNNs. VGG pointed out that two stacked 3×3 convolutions have the same receptive field as one 5×5, and three equal one 7×7, but with fewer parameters ($2\times9C^2=18C^2$ vs $25C^2$) and more non-linearity
- **1×1 convolution**: does not look at the spatial neighborhood; it only linearly combines the channels at each pixel position (equivalent to a per-pixel fully connected layer). Its roles include:
    - Increasing/reducing dimensionality: changes the number of channels, e.g. in Inception and the ResNet bottleneck a 1×1 convolution first reduces the channels before the 3×3 convolution, greatly reducing computation
    - Cross-channel information fusion: integrates features from different channels
    - Adding non-linearity: a 1×1 convolution followed by an activation function deepens the network without changing the receptive field (Network in Network)
    - Few parameters: the parameter count is $C_{in}\times C_{out}$, $1/9$ of a 3×3 convolution; it serves as the pointwise convolution in depthwise separable convolutions
- Differences: 3×3 has a spatial receptive field and can extract local spatial features (edges, textures); 1×1 has a receptive field of 1 and only transforms channels, extracting no spatial features; a 3×3 convolution has $9C_{in}C_{out}$ parameters, a 1×1 has $C_{in}C_{out}$
- See also: [Deep Learning Fundamentals](dl-fundamentals.md) Q15; [Computer Vision](cv.md) Q34

Q17: ***How do you calculate the number of parameters of a Group Convolution, and how does it differ from a standard convolution?***

- Standard convolution: each kernel convolves over **all** $C_{in}$ input channels; the parameter count is $k\times k\times C_{in}\times C_{out}$ (excluding bias)
- Group convolution: the input channels are split into $g$ groups of $C_{in}/g$ channels each, and the output channels are likewise split into $g$ groups of $C_{out}/g$ each; each group is convolved independently and the results are concatenated along the channel dimension. The parameter count is $g\times k\times k\times\frac{C_{in}}{g}\times\frac{C_{out}}{g}=k\cdot k\cdot\frac{C_{in}}{g}\cdot C_{out}$, i.e. $1/g$ of a standard convolution, and the computation (FLOPs) also drops to $1/g$
- Example: with $k=3,C_{in}=C_{out}=256,g=8$, a standard convolution has $3\times3\times256\times256\approx590\text{K}$ parameters, while the group convolution has about $74\text{K}$
- Special cases: $g=1$ is a standard convolution; $g=C_{in}=C_{out}$ is a **Depthwise Convolution**, which combined with a 1×1 pointwise convolution forms a depthwise separable convolution (MobileNet)
- Differences, pros and cons: group convolution has fewer parameters and less computation; it was first used in AlexNet for multi-GPU parallelism, and ResNeXt uses it to increase width (cardinality) at the same computational cost to improve accuracy; the drawback is that there is no information exchange between groups, which limits feature expressiveness, and ShuffleNet uses **channel shuffle** to connect information across groups; in addition, group convolution has lower memory-access efficiency on real hardware, so the reduction in FLOPs does not necessarily yield a proportional speed-up
