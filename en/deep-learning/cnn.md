# CNN

Q1: ***What is the difference between CNN and RNN? When should each be used?***

- **CNNs are good at data with local spatial structure (such as images) and exploit translation invariance: the same feature is detected wherever the object appears. RNNs carry a memory state and suit sequence data where order matters.**
- Typical CNN use cases: image classification, detection, segmentation.
- Typical RNN use cases: speech, text, video frames, time series of all kinds.

Q2: ***How can CNNs be used for time-series forecasting?***

- **Treat a time series as a one-dimensional "image" and use 1D convolutions along the time axis to automatically extract local patterns, producing a representation of the sequence that is then used for prediction.**
- Typical models: TCN and WaveNet (causal convolution + dilated convolution, so the receptive field grows quickly with depth).
- Advantages: parallelizable, fairly robust to noise, and able to extract information-rich deep features.

Q3: ***In CNNs, what are the pros and cons of Max Pooling vs. Average Pooling?***

- **Max pooling keeps the strongest response and is better at extracting textures and salient features; average pooling keeps overall information and preserves the background better.**
- From an error perspective: max pooling reduces the mean shift caused by convolution-parameter errors; average pooling reduces the estimation variance caused by the limited neighborhood size.
- Recognition tasks (e.g. digit recognition) generally use max pooling.

Reference: [https://www.zhihu.com/question/34898241](https://www.zhihu.com/question/34898241)

Q4: ***Compare CNNs and multilayer perceptrons (MLP)***

- **A CNN uses local connectivity and weight sharing, so it has far fewer parameters than a fully connected MLP, and it preserves the spatial relationships between pixels, which makes it better suited to images.**
- In an MLP every neuron connects to all nodes of the previous layer, so parameters are redundant and the network is hard to make deep.
- An MLP needs the input flattened into a vector, which loses spatial information.
- An MLP can be seen as a special case of a CNN: when the kernel size equals the input size, the computation is equivalent to a fully connected layer.

Reference: [https://medium.com/data-science-bootcamp/multilayer-perceptron-mlp-vs-convolutional-neural-network-in-deep-learning-c890f487a8f1](https://medium.com/data-science-bootcamp/multilayer-perceptron-mlp-vs-convolutional-neural-network-in-deep-learning-c890f487a8f1)

*Q5*: *****What is the role of the *fully connected layer* in a *CNN*?**  

- **The fully connected layer acts as the "classifier": convolution and pooling map the data into a feature space, and the fully connected layer integrates those features and maps them to the label space for the final classification or regression.**

*Q6*: *****Explain the significance of the ReLU activation function in convolutional neural networks*****

- **Convolution itself is linear, and stacking linear layers is still linear; ReLU introduces non-linearity so the network can fit complex image features.**
- It sets negative values to zero pixel by pixel and is usually applied after every convolution.
- It is cheap to compute and has gradient 1 in the positive region, which eases vanishing gradients.

Q7: ***Given the input image size, filter size, stride and padding size, what is the size of the feature map?***

- **Output side length = (input size − kernel size + 2 × padding) / stride + 1, rounded down.**
- Key formula: $\lfloor (n-f+2p)/s \rfloor + 1$ (n is the input size, f the kernel size, p the padding, s the stride)

Q8: ***Explain Valid Padding and Same Padding***

- **Valid padding adds no border: the kernel is applied only where it lies fully inside the image, so the output shrinks. Same padding pads the edges with zeros so the output has the same size as the input (when stride = 1).**

Q9: ****What are the different types of pooling? Describe their characteristics.****

- **The common types are max, average and sum pooling; max pooling is the most widely used because it keeps the most salient features.**
- Max pooling: takes the maximum in the window and suppresses noise.
- Average pooling: takes the mean of the window and is smoother.
- Sum pooling: takes the sum of the elements in the window.

Q12: **Explain the role and function of the flatten layer in a CNN**

- **Flatten turns the multi-dimensional feature maps from the last convolution/pooling layer into a single vector, which is fed to the fully connected layers that learn non-linear relationships in the features for classification.**

Q13: ****List the hyperparameters of a pooling layer****

- **The hyperparameters of a pooling layer are the window size f, the stride s and the pooling type (max/average); it has no learnable parameters.**
- Key formula: for an input of $n_h\times n_w\times n_c$, the output is $((n_h-f)/s+1)\times((n_w-f)/s+1)\times n_c$

Q14: ****Explain the significance of "parameter sharing" and "sparse connectivity" in CNNs****

- **Both greatly reduce the number of parameters and exploit the locality of images: parameter sharing means the same kernel is slid and reused over the whole image; sparse connectivity means each output depends only on a small region of the input.**
- Intuition for parameter sharing: a feature detector that is useful in one part of the image (e.g. an edge detector) is useful elsewhere too, and it gives translation equivariance.
- Intuition for sparse connectivity: image features are mostly local, so no output needs to see the entire input.

Q15: **Can a CNN be used to perform dimensionality reduction? If so, which sub-layer in a CNN performs the dimensionality reduction?**

- **Yes. Spatial reduction is mainly done by the pooling layer, which downsamples with a sliding window to shrink the height and width of the feature map.**
- In addition, convolutions with stride greater than 1 also reduce spatial size, and 1×1 convolutions can reduce the number of channels.

Q16: ***What are the roles of 3×3 and 1×1 convolutions, and how do they differ?***

- **A 3×3 convolution looks at a 3×3 neighborhood and extracts local spatial features (edges, textures); a 1×1 convolution ignores the neighborhood and only linearly combines channels at each pixel, so it changes the number of channels and fuses channel information.**
- 3×3: the most common kernel in modern CNNs. VGG pointed out that two stacked 3×3 convolutions have the receptive field of one 5×5 and three equal one 7×7, but with fewer parameters and more non-linearity.
- 1×1: used to increase/reduce channels (Inception and the ResNet bottleneck reduce channels before the 3×3, saving a lot of computation); followed by an activation it adds non-linearity without changing the receptive field (Network in Network); it is also the pointwise convolution in depthwise separable convolutions.
- Parameter count: 3×3 has $9C_{in}C_{out}$, 1×1 has $C_{in}C_{out}$, just 1/9 of the former.
- See also: [Deep Learning Fundamentals](dl-fundamentals.md) Q15; [Computer Vision](cv.md) Q34

Q17: ***How do you calculate the number of parameters of a Group Convolution, and how does it differ from a standard convolution?***

- **Group convolution splits the input and output channels into g groups, convolves each group independently and concatenates the results, so both parameters and computation drop to 1/g of a standard convolution.**
- Example: with k=3, 256 input and output channels and g=8, a standard convolution has about 590K parameters, the group convolution about 74K.
- Special cases: g=1 is a standard convolution; g equal to the number of channels is a depthwise convolution, which together with a 1×1 convolution forms MobileNet's depthwise separable convolution.
- Uses: AlexNet first used it for multi-GPU parallelism; ResNeXt uses it to increase width at the same computational cost and improve accuracy.
- Drawbacks: no information exchange between groups (ShuffleNet fixes this with channel shuffle); memory-access efficiency on hardware is low, so fewer FLOPs do not necessarily give a proportional speed-up.
- Key formula: $\text{params} = k\times k\times\frac{C_{in}}{g}\times C_{out}$
