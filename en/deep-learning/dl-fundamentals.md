# Deep Learning Fundamental

# Beginner

Q1: ***What is the essential difference between deep learning and machine learning?***

- Machine learning relies more on human factors: people need to manually select and combine effective features; deep learning automates most of the feature-extraction steps, removing human intervention
- Machine learning depends less on the amount of data, while **deep learning** needs large amounts of data to deliver higher accuracy
- How deep learning extracts features automatically: through multiple layers of nonlinear transformations it performs **representation learning**, and the parameters of each layer are learned end-to-end by backpropagation from the loss of the final task; shallow layers learn low-level features such as edges and textures, deeper layers combine low-level features into high-level semantic features such as parts and objects, and the feature extractor and classifier are optimised jointly, no longer relying on hand-crafted features
- Why big data usually needs more complex models: the larger the dataset, the more patterns and edge cases it contains; a simple model has limited capacity, so it **underfits** (high bias) and its performance saturates quickly; complex models (more parameters, deeper layers) are more expressive and can fit more complex mappings; at the same time, big data itself acts as a regulariser, lowering the risk (variance) of a complex model overfitting, so only big data + big models together achieve the full effect

Q2: ***Why does the performance of deep learning improve as the amount of data increases?***

- As the amount of data fed into the learning algorithm increases, the model takes more edge cases into account, so the algorithm learns to make correct decisions in these edge cases, providing better generalisation

Q3: ***What is ensemble learning, and how is it used in deep learning?***

- Ensemble learning means learning with two or more models and combining the results of the individual models with some rule, in order to improve the *generalisation* ability of the overall model
- Applications of ensemble learning in deep learning include dropout, zoneout and drop-connect

Q4: **In deep learning, how do you choose an activation function**?

- If the output to be predicted is the probability of a binary class, the Sigmoid function can be used as the output layer of the network.
- If the output to be predicted has two classes, the Tanh function can be used as the output layer of the network.
- Because the ReLU function is simple to compute and helps suppress vanishing gradients, it is widely used in the intermediate layers of networks

![Untitled](dl-fundamentals/Untitled.png)

Q5: ***Compared with machine learning, what are the advantages of deep learning?***

- Deep learning does not require the model designer to have a deep understanding of feature engineering; deep learning is end-to-end, which makes it quick to get started with
- Deep learning achieves higher accuracy when the dataset is large and the modelling problem is complex

Q7: ***How do you tell whether a deep learning model suffers from the vanishing gradient problem?***

- The model loss decreases very slowly during training
- After training stops (a certain number of epochs is reached), the model loss is still very large
- The model weights shrink exponentially and become very small during training
- During training, the model weights become zero

Q8: ***How do you tell whether a deep learning model suffers from the exploding gradient problem? How do you confirm it?***

- The following phenomena indicate that your model may have exploding gradients: the model cannot learn the training data well (the loss stays high), training is unstable (the loss changes a lot with every update), or the loss becomes NaN during training;
- When these phenomena occur, you can dig deeper into the model to check for exploding gradients; the following confirm exploding gradients: the model weights become NaN during training; during training, the error gradient values of every node and layer are consistently above `1.0`;

Q9: ***What is the role of an Embedding Layer?***

- The purpose of the embedding layer is to map the model input, a one-hot vector (a high-dimensional, sparse representation), to a low-dimensional dense representation, saving model computation

Q10: ***What is the early stopping strategy?***

- early-stopping: a form of regularisation. A NN updates its parameters with gradient descent, but past a certain stage of training, improving the model's fit to the training data increases the generalisation error. Past that point, improving the model's fit to the training data increases the generalisation error. early-stopping specifies the number of iterations the network may be trained for.

Q11: ***What is incremental learning?***

- Incremental learning is the ability of an algorithm, after generating a classifier from a previously available dataset, to learn from new data
- Incremental learning algorithms continually use input data to extend the knowledge of the existing model, i.e. they use a dynamic data stream to further train the model; the goal is for the learning model to adapt to new data without forgetting its existing knowledge

Reference: [https://link.springer.com/](https://link.springer.com/)

Q12: ***Explain the principle of the perceptron***

- The perceptron is an algorithm for supervised learning of binary classifiers. Its input is the feature vector of an instance and its output is the class of the instance, +1 for the positive class and -1 for the negative class. The perceptron is a discriminative model; its goal is to separate the positive and negative classes of input instances with a separating hyperplane.
- The goal of perceptron learning is to find a separating hyperplane that completely separates positive and negative samples, i.e. to find the hyperplane parameters w, b; the perceptron learns its parameters through supervised learning.

![Untitled](dl-fundamentals/Untitled%201.png)

# Intermediate

Q13: ***What is a computational graph? Which algorithms are computational graph models?***

- Computational graphs are graphs containing equation data. They are a form of directed graph representing mathematical expressions. A very common example is postfix, infix and prefix evaluation. Each node in the graph can contain an operation, a variable or an equation itself. These graphs appear in most computations performed in computers.

![Untitled](dl-fundamentals/Untitled%202.png)

- Perceptrons, deep neural networks and linear regression are all computational graph models

Reference: [https://towardsdatascience.com/evolution-of-graph-computation-and-machine-learning-3211e8682c83](https://towardsdatascience.com/evolution-of-graph-computation-and-machine-learning-3211e8682c83)

Q14: ***What is the difference between linear and nonlinear activation functions?***

- A linear activation function predicts the output as a linear function of the input, while the output of a nonlinear function (such as tanh, sigmoid, relu) is a nonlinear function of the input; nonlinear activation functions allow neural networks to learn more complex behaviour patterns

Q15: ***Explain the meaning of the 1*1 convolution operation.***

- The 1x1 convolution can change the dimension of the data along the channel axis: with input data $（c_{in},H,W）$, after processing by a $(c_{out},1,1)$ kernel the output is $(c_{out},H,W)$
- In the Inception architecture, we use 1x1 convolution filters to reduce the dimension along the filter dimension

Q16: ***When designing a neural network, which is more important, width or depth?***

- For a standard feed-forward neural network, depth is more valuable than width; depth adds more nonlinearity to the network and strengthens its learning ability; but at the same time, deeper models overfit more easily.
- (Related research paper: [http://proceedings.mlr.press/v49/eldan16.pdf](http://proceedings.mlr.press/v49/eldan16.pdf)).

Reference: [https://www.quora.com/What-are-the-tradeoffs-of-increasing-the-depth-of-a-neural-network](https://www.quora.com/What-are-the-tradeoffs-of-increasing-the-depth-of-a-neural-network)

Q17: ***What do the hidden layers of a neural network compute?***

- Hidden layers provide nonlinear transformations; they extract high-dimensional features from the input
- Hidden layers transform the input into features usable by the output, and transform the input into the dimension required by the output

Reference: [https://stats.stackexchange.com/](https://stats.stackexchange.com/)

Q18: ***In a neural network, can nonlinear activation functions be replaced with linear ones? Why?***

- No, nonlinear activation functions are indispensable in neural networks; this nonlinearity is the key difference between a neural network and a simple linear regression model, and it **allows neural networks to learn complex function representations**

Q19: ***In a deep learning model, how do you choose the loss function?***

- The choice of loss function must match the framing of the specific predictive modelling problem, e.g. classification and regression problems use different loss functions; in addition, the configuration of the output layer must also suit the chosen loss function.
- Regression loss functions: mean squared error loss, mean squared logarithmic error loss, mean absolute error loss
- Binary classification loss functions: cross-entropy, Hinge Loss;
- Multi-class classification loss functions: multi-class cross-entropy loss, K-L divergence; K-L divergence measures loss by computing the loss of information; KL divergence loss is more commonly used when using a model that learns to approximate a more complex function rather than simple multi-class classification, for example in autoencoders used to learn dense feature representations;

Reference: [https://machinelearningmastery.com/](https://machinelearningmastery.com/)

Q20: ***Introduce the principle of GAN***

- GAN, the generative adversarial network, is a kind of generative model; model training finds an equilibrium in an adversarial game
- The main structure of a GAN consists of a generator G (Generator) and a discriminator D (Discriminator). The generator randomly samples from the latent space as input, and its output should imitate the real samples in the training set as closely as possible. The input of the discriminator is a real sample or the output of the generator, and its goal is to distinguish the generator's output from real samples as well as possible, while the generator tries to fool the discriminator as much as possible. The two networks compete and keep adjusting their parameters; the ultimate goal is for the discriminator to be unable to judge whether the generator's output is real.

![Untitled](dl-fundamentals/Untitled%203.png)

Q21: ***What is transfer learning?***

- Put simply, transfer learning means using existing knowledge to learn new knowledge; the core is to find the similarity between existing and new knowledge
- Transfer learning is a long-standing concept in machine learning; by learning approach it can be divided into instance-based transfer, feature-based transfer, model-based transfer, and relation-based transfer.
- Instance-based transfer achieves knowledge transfer by weighting and reusing labelled samples from the source domain; feature-based transfer maps the source and target domains into the same space (or one into the other's space) and minimises the distance between them; model-based transfer combines the models and samples of the source and target domains to adjust model parameters; relation-based transfer learns relations between concepts in the source domain and then transfers them by analogy to the target domain.

![Untitled](dl-fundamentals/Untitled%204.png)

Q22: ***When is a convolutional layer equivalent to a fully connected layer?***

- In fact, the fully connected layer is a subset of the convolutional layer; if the kernel size is set equal to the size of the input image and the number of channels equals the number of output neurons of the fully connected layer, the convolutional layer is equivalent to a fully connected layer

Q23: ***What are the differences and connections between GANs and autoencoders?***

- An autoencoder learns an encoder network and a decoder network simultaneously: the input (e.g. an image) is fed to the encoder, which tries to reduce it to a compressed encoding, which is then fed to the decoder. The learning objective of the autoencoder is the reconstruction loss; it is commonly used for dimensionality reduction, anomaly detection, etc.
- A generative adversarial network learns a generator and a discriminator through a game: the generator produces images realistic enough to pass as real, and the discriminator detects whether an image is real; GANs are commonly used for image generation, etc.
- GANs and AEs are both generative models, but they have different learning objectives and different application areas

Reference: [https://stackoverflow.com](https://stackoverflow.com/)

Q24: ***What are the benefits of pre-training for neural networks?***

- Pre-trained models can learn knowledge of the language itself from massive unlabelled data, and are then fine-tuned on a small amount of labelled data, so that downstream tasks can better learn the features of the language itself and task-specific knowledge. Pre-trained models not only make full use of widely available web resources, but also neatly address the complexity of manually labelling data

Q25: ***Are there any problems with using Batch-Norm in deep neural networks?***

- Batchnorm normalises a batch of data and gives faster convergence and higher accuracy; but Batchnorm also has its drawbacks:
    - Batchnorm is memory-intensive: the statistics of all batch processing need to be stored in the layer, consuming a lot of compute
    - There is a discrepancy between training and inference; inference results depend on the batch size
    - It breaks the independence between examples in a mini-batch
    - Model performance is sensitive to the batch size; when the batch is too small, batch-norm performs poorly
    - batchnorm is not suited to sequence problems: **in sequence models we may have sequences of different lengths, and smaller batch sizes corresponding to longer sequences**
- To address these drawbacks, current improvements include layer-norm, instance-norm, etc.

Reference: [https://medium.com/geekculture](https://medium.com/geekculture)

Q26: ***What is multitask learning? When is it appropriate to use it?***

- multitask-learning trains a model to perform multiple tasks simultaneously; in deep learning, MTL refers to training a neural network to perform several tasks by sharing some layers and parameters of the network across tasks. MTL improves generalisation by exploiting information shared across tasks. By sharing some parameters of the network, the model can learn a more efficient and compact data representation
- Multi-task learning helps when the learning tasks are somewhat correlated. Alternatively, when tasks share underlying principles or information, multi-task learning can improve performance

Reference: [https://towardsdatascience.com/](https://towardsdatascience.com/)

Q27: ***What is the difference between AE and VAE?***

- The input of the AE decoder is the output of the encoder, while the input of the VAE decoder is sampled from a Gaussian distribution whose mean and variance are output by the encoder; the VAE has generative ability.
- VAE solves the problem of the non-regularised latent space of AE, which enables it to generate data from randomly sampled vectors from the latent space

![Untitled](dl-fundamentals/Untitled%205.png)

Reference: [https://towardsdatascience.com/difference-between-autoencoder-ae-and-variational-autoencoder-vae-ed7be1c038f2](https://towardsdatascience.com/difference-between-autoencoder-ae-and-variational-autoencoder-vae-ed7be1c038f2)

Q28: ***Name some CNN architectures you know***

- A typical CNN architecture stacks a few convolutional layers, then a pooling layer, then a few more convolutional layers (+ReLU), then another pooling layer, and so on
- Typical CNN architectures include: LetNet, AlexNet, GoogleNet, Inception V2 V3 V4, VGG, ResNet, DenseNet, ResNeXt, Channel Boosted CNN, EfficientNet

Reference: [https://www.kaggle.com/general/255114](https://www.kaggle.com/general/255114)

Q29: ***What is a Boltzmann machine?***

- A Boltzmann machine is a neural network made of symmetrically connected nodes, where nodes make binary decisions and decide by themselves whether to activate. Boltzmann machines can be strung together to form more complex systems, such as deep belief networks;
- Boltzmann machines are often used to solve various computational problems; for example, for search problems, the weights on the connections can be fixed and used to represent the cost function of an optimisation problem

![Untitled](dl-fundamentals/Untitled%206.png)

Reference: [https://deepai.org/](https://deepai.org/)

Q30: ***Which neural network would you choose for a video classification task?***

- A video consists of an ordered sequence of frames. Each frame contains *spatial* information, and the sequence of frames contains *temporal* information. To model both aspects, a hybrid architecture CRNN consisting of convolutions (for spatial processing) and recurrent layers (for temporal processing) can be used for classification

Q31: ***What effect does dropout have on neural network training?***

- By randomly deactivating some network nodes, dropout lets a single model simulate a large number of different network architectures; it provides a computationally very cheap and very effective regularisation method to reduce overfitting and improve the generalisation error of various deep neural networks.

Q32: ***Can an AE model be used for feature generation? If so, how?***

- Yes; an AE model can learn a compressed representation of the raw data, which can be used as features;
- Method: train the AE to reconstruct the input data; after training, discard the decoder and use the model's output at the bottle-neck as the features of the input data; these features retain most of the information in the input data and have a fixed dimension

Q33: ***What are the differences between Batch Normalization, Instance Normalization, Layer Normalization, and Group Normalization?***

- The four normalisation methods are shown in the figure below; the three dimensions of the cube are batch / channel / H*W. When computing mean and standard deviation, BN fixes the channel (within one channel) and averages over HW and batch; LN fixes the batch (within one sample) and averages over HW and channel; IN fixes both channel and batch (within one channel of one sample) and averages over HW; GN fixes the batch and groups the channels (groups channels within one sample), averaging over HW within each group

Reference: "Group Normalization"

![Untitled](dl-fundamentals/Untitled%207.png)

Q34: ***With the same number of parameters, why do deep neural networks perform better than shallow ones?***

- The operation of a hidden layer can be seen as applying a nonlinear activation to a linear combination of the input vector, and each hidden layer can be seen as a modular function. A deep network with many hidden layers is like a stack of many functions, and compared with a shallow network, the same number of parameters can realise more complex functions. In other words, deep networks use their parameters more efficiently
- Taking a concrete task as an example, such as speech recognition, the initial layers learn low-level features such as edges and curves (manner of articulation), and later layers learn high-level features (phonemes); deep networks with many hidden layers learn such high-level features more easily than shallow networks

Q35: ***Compare SVM and deep learning***

- Intuitive difference: deep learning stacks many neural network layers to learn increasingly abstract representations of the data in a hierarchical way; SVM finds a separating boundary among data points (a hyperplane in the linearly separable case) that is as far as possible from the support vectors of either class;
- Deep learning is relatively uninterpretable, while SVM is backed by strong mathematical derivations;
- Sensitivity to weight initialisation: because neural networks use gradient descent, they are sensitive to the initial randomisation of their weight matrices. This is because if the initial randomisation places the network close to a local minimum of the optimised function, the accuracy will never rise above a certain threshold. In contrast, SVMs are more reliable: regardless of their initial configuration they are guaranteed to converge to the global minimum.
- Deep learning is more convenient for multi-class problems; SVM needs multiple SVM classifiers to handle multi-class problems, while deep learning needs only a single neural network.
- SVM performs poorly when the number of features exceeds the number of samples; SVM requires more work on feature engineering;

Q36: ***In what ways is the ReLU activation function better than sigmoid?***

- Principle: Sigmoid is $\sigma(x)=\frac{1}{1+e^{-x}}$, with output range $(0,1)$ and derivative $\sigma'(x)=\sigma(x)(1-\sigma(x))\le 0.25$; ReLU is $f(x)=\max(0,x)$, with derivative 1 for $x>0$ and 0 for $x<0$
- Sigmoid characteristics: the output can be interpreted as a probability, suitable for binary-classification output layers; but it saturates at both ends so gradients vanish easily, its output has non-zero mean (not zero-centred), which makes the gradient updates of the next layer zig-zag, and it involves an exponential, which is slow to compute
- ReLU characteristics: no saturation in the positive region, simple to compute, fast convergence, produces sparse activations; drawbacks are non-zero-mean output, no upper bound, and zero gradient in the negative region, which can cause **Dead ReLU** (neurons that never activate again); variants such as Leaky ReLU, PReLU, ELU, GELU can mitigate this
- Summary of differences: Sigmoid is commonly used in output layers (binary classification, gates such as those in LSTM), ReLU is commonly used in hidden layers
- ReLU requires less computation;
- ReLU solves the vanishing gradient problem that sigmoid tends to cause; as the gradient plots of the two show, during backpropagation the saturation regions of Sigmoid and tanh are very flat and close to 0, which easily causes vanishing gradients and slows convergence. The gradient of ReLU is constant in most cases, which helps with the convergence of deep networks
- Because ReLU does not easily suffer vanishing gradients, networks train faster
- ReLU makes the output of some neurons 0, which creates sparsity in the network and reduces the interdependence of parameters, alleviating overfitting
    
    ![Untitled](dl-fundamentals/Untitled%208.png)
    

![Untitled](dl-fundamentals/Untitled%209.png)

Q37: ***How is the Fourier transform used in deep learning?***

- The Fourier transform can save computation in convolutional layers; the input and filter matrices in a CNN can be converted to the frequency domain for multiplication, and the resulting matrix of the frequency-domain multiplication can be converted back to the time domain. Conversion from the time domain to the frequency domain can be done with the Fourier transform or the fast Fourier transform, and conversion from the frequency domain to the time domain with the inverse Fourier transform or the inverse fast Fourier transform.

![Untitled](dl-fundamentals/Untitled%2010.png)

Q38: ***How is ensemble learning used in deep learning?***

- Ensemble learning means training multiple models and combining their predictions to reduce the variance of the final prediction and improve accuracy; ensemble learning can be applied to deep learning in three aspects: data, models, and combination of results
- Varying the training data: k-fold cross-validation (k different models are trained on k different subsets of the training data; these k models are saved and used as members of the ensemble), resampling (resample the training dataset, then train the network on the resampled dataset)
- Models: dropout, drop-connect
- Combination of results: weighted average ensemble (slightly improved by weighting the prediction of each model, with weights optimised on a hold-out validation set), model weight averaging (average the weights of multiple networks in the hope of producing a new single model whose overall performance is better than any of the original models)

Reference: [https://machinelearningmastery.com/](https://machinelearningmastery.com/)

Q39: ***How does deep learning mitigate/avoid the curse of dimensionality?***

- The curse of dimensionality usually arises because the data contains correlated and too many irrelevant (noise) features. Deep learning architectures can extract low-dimensional features of high-dimensional data representations, so DL systems reduce the influence of irrelevant features during learning while increasing the influence of relevant features.

Reference: [https://www.quora.com/](https://www.quora.com/)

Q40: ***How many skip-connection methods do you know?***

- ****ResNet:**** information from the initial layers is passed to deeper layers via matrix addition; this operation has no additional parameters
- ****DenseNets:**** DenseNets concatenate the output feature maps of a layer with the next layer instead of summing them

![Untitled](dl-fundamentals/Untitled%2011.png)

- ****U-Net: long skip connections****

Reference: [https://www.analyticsvidhya.com/](https://www.analyticsvidhya.com/)

Q41: What is a ***Deep Recurrent Q-Network*?**

- DRQN is a network that replaces the first post-convolutional fully connected layer in DQN with an LSTM; although DRQN sees only one frame at each time step, it successfully integrates temporal information and replicates DQN's performance on standard Atari games and on partially observed equivalents with flickering game screens

Reference: [https://arxiv.org/abs/1507.06527](https://arxiv.org/abs/1507.06527)

Q42: *****What are the differences between the *region-based object detection network* (R-CNN), *Fast R-CNN* and *Faster R-CNN*?**

- R-CNN uses the selective search algorithm to extract only 2000 regions from the image as region proposals, avoiding the computational explosion caused by selecting a huge number of regions in the image; but even so, RCNN is not real-time
- Instead of feeding the region proposals to the CNN, Fast R-CNN feeds the input image to the CNN to generate a convolutional feature map. From the convolutional feature map, the proposed regions are identified and warped into squares, and then, using an RoI pooling layer, they are reshaped to a fixed size so they can be fed into the fully connected layers.
- Instead of using the selective search algorithm on the feature map to identify region proposals, Faster R-CNN uses a separate network to predict region proposals. The predicted region proposals are then reshaped with an RoI pooling layer, which is used to classify the image within the proposed region and predict the offset values of the bounding boxes.

Q43: ***For sequence problems, compare HMM models and RNN models***

- HMM is simpler and based on a strong assumption: state transitions depend only on the current state, not on anything in the past; HMM suits small datasets
- RNN performs better on large datasets and requires no extra assumptions;
- HMM is a generative model, RNN is a discriminative model

Reference: [https://stats.stackexchange.com/](https://stats.stackexchange.com/)

Q44: ***Which object detection network architectures do you know?***

- Common object detection architectures include RetinaNet, YOLO, CenterNet, SSD (single shot multibox detector), and region proposal methods (R-CNN, Fast-RCNN, Faster RCNN, Cascade R-CNN); they can be roughly divided into two categories: single-stage and multi-stage object detection
- YOLO is single-stage detection: a single end-to-end trained neural network takes a photo as input and directly predicts the bounding boxes and class labels of each bounding box
- Multi-stage detection splits the object detection task into two stages: extracting RoIs (regions of interest), then classifying and regressing the RoIs. Examples of 2-stage object detection architectures include R-CNN, Fast-RCNN, Faster-RCNN, Mask-RCNN, etc.

Reference: [https://www.v7labs.com/](https://www.v7labs.com/blog/object-detection-guide)

![Untitled](dl-fundamentals/Untitled%2012.png)

Q45: ***What are the difficulties in training GANs?***

- GAN models have the following main problems:
    - Non-convergence: model parameters oscillate, are unstable and never converge
    - Mode collapse: the generator collapses and produces a limited variety of samples
    - Vanishing gradients: the discriminator is so successful that the generator's gradient vanishes and it learns nothing
- Mode collapse is one of the hardest problems to solve in GANs; ideally, a generative model should be able to generate all 10 digits (for the MNIST dataset); if it can only generate a few of them and misses the other modes, that is mode collapse (the generated samples are largely repetitive and similar)
- Cause of mode collapse: current deep neural networks can only approximate continuous mappings, while the transport map from input data to output data is a discontinuous mapping with discontinuity points. In other words, during GAN training, the target mapping is not within the representable function space of the NN, and this contradiction makes convergence difficult; if the support of the target probability measure has multiple connected components while GAN training yields a continuous mapping, the range of the continuous mapping may **concentrate on a single connected component**, which is mode collapse
- Solutions:
    - Empirical methods:
        - **unrolled GAN (when updating its parameters, the generator considers not only the current state but also the discriminator's response over K additional steps, avoiding short-sighted behaviour)**
        - **DRAGAN (prevents the GAN from entering bad local Nash equilibria)**
    - Theoretical method: **according to Brenier's theory, the transport map is discontinuous at singular points; computing the Brenier potential function and identifying the set of singular points can avoid mode collapse**
    
    ![Untitled](dl-fundamentals/Untitled%2013.png)
    

Reference: [https://jonathan-hui.medium.com/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b](https://jonathan-hui.medium.com/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b)

"Mode Collapse and Regularity of Optimal Transportation Maps"

Q46: ***Suppose you want to train a classifier and you have a large amount of unlabelled training data but only a few thousand labelled instances. How would you proceed?***

- Use a semi-supervised approach: train model A with the few thousand labelled examples, label the unlabelled training data with it, select the high-confidence samples by confidence score, and use them as labelled data to fine-tune model A to obtain the final model;

Q47: ***How can genetic algorithms be used to optimise neural networks?***

- Optimising a BP neural network with a genetic algorithm consists of three parts: determining the BP network structure, genetic algorithm optimisation, and BP network prediction. The BP network structure is determined by the number of input and output parameters of the fitted function, which in turn determines the length of a genetic algorithm individual. The genetic algorithm optimisation uses the GA to optimise the weights and thresholds of the BP network: each individual in the population contains all the weights and thresholds of a network, individual fitness values are computed via a fitness function, and the GA finds the individual with the optimal fitness through selection, crossover and mutation. In BP network prediction, the optimal individual obtained by the GA is used to assign the network's initial weights and thresholds, and after training the network predicts the function output.

![Untitled](dl-fundamentals/Untitled%2014.png)

Reference: [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243030](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243030)

Q48: ***Why can neural networks fit arbitrary functions? Using sin and cos as examples, explain the role of nonlinear activation functions. How is the loss computed, and how does it relate to the number of samples and the feature dimension?***

- **Universal Approximation Theorem**: as long as there are enough hidden neurons, a feed-forward network with a single hidden layer and a nonlinear activation function (such as sigmoid or ReLU) can approximate any continuous function on a compact set to arbitrary precision. Intuition: each ReLU neuron provides a "kink", and multiple ReLUs combine into a piecewise linear function; sigmoid neurons can be combined into "steps/bumps", and superimposing many such basis functions can approximate a curve of any shape
- Taking $y=\sin(x)$ as an example: without nonlinear activations, a multi-layer network $W_2(W_1x+b_1)+b_2=W'x+b'$ is still a linear function and can only fit a straight line, no matter how deep, so it cannot fit a sine curve; with ReLU added, the network output is a piecewise linear function, and with enough line segments it can approximate the waveforms of $\sin(x)$ and $\cos(x)$; the more neurons and the finer the segments, the smaller the approximation error
- Note: the universal approximation theorem only guarantees that such a network "exists"; it does not guarantee that gradient descent can find it, nor that it generalises; in practice, deeper networks can often achieve the same approximation with fewer parameters
- Loss computation: for $N$ samples, the empirical risk is $L=\frac{1}{N}\sum_{i=1}^{N}\ell(f(x_i;\theta),y_i)$; regression commonly uses MSE $\ell=(f(x_i)-y_i)^2$, classification commonly uses cross-entropy $\ell=-\sum_{c}y_{i,c}\log p_{i,c}$
- Relation to the number of samples: the loss is averaged over samples, so the number of samples $N$ does not change the magnitude of the loss, but it affects computation ($O(N)$) and the variance of the estimate — the more samples, the closer the empirical risk is to the expected risk and the better the generalisation; in mini-batch training, the average loss within a batch approximates the full-data loss
- Relation to the feature dimension: the feature dimension $d$ determines the number of input-layer parameters (e.g. $d\times h$ in the first layer) and the per-sample computation ($O(Nd)$); the higher the dimension and model capacity, the more easily the model overfits when samples are insufficient (curse of dimensionality), usually requiring more samples or regularisation

Q49: ***What is overfitting? Why can training get stuck in a local optimum?***

- **Overfitting**: the model performs well on the training set (low training loss) but poorly on the validation/test set (large generalisation error); the cause is that the model capacity is too large relative to the amount of data, so it memorises noise and incidental patterns in the training data rather than the true data distribution. A typical symptom is the training loss continuing to fall while the validation loss starts to rise
- **Local optima**: the loss surface of a neural network is highly **non-convex**, with many local minima, saddle points and flat regions; gradient descent uses only local gradient information, so when it reaches a point with zero gradient (a local minimum or saddle point) the updates stall and it cannot "see" better solutions further away
- In fact, in high-dimensional parameter spaces, **saddle points** are more common than local minima, and most local minima have a loss not far from the global optimum; training difficulty comes more from saddle points, flat regions, vanishing/exploding gradients and ill-conditioned curvature
- Mitigations: use momentum or adaptive optimisers such as Adam; the mini-batch noise of SGD helps escape sharp minima and saddle points; proper weight initialisation (Xavier/He), learning-rate warm-up and decay (warm-up, cosine, periodic restarts); BN and residual connections make the loss surface smoother

Q50: ***How do you reduce overfitting and improve the generalisation ability of a network?***

- **Data level**: add training data; data augmentation (flipping, cropping, rotation, colour jitter, Mixup, CutMix; for text, back-translation, synonym replacement, etc.); clean noisy labels
- **Model level**: reduce model complexity (fewer layers/neurons); use parameter sharing and appropriate structural inductive biases (e.g. CNN); use pre-trained models + fine-tuning (transfer learning)
- **Regularisation**: L2 regularisation (weight decay) $L=L_0+\frac{\lambda}{2}\lVert w\rVert^2$, L1 regularisation $L=L_0+\lambda\lVert w\rVert_1$ (produces sparse weights); Dropout; Label Smoothing
- **Training strategy**: early stopping; Batch Normalization (has some regularising effect); appropriate learning rate and batch size (small-batch noise tends towards flat minima, which generalise better)
- **Ensemble methods**: ensembling multiple models (bagging, model averaging, snapshot ensemble) reduces variance
- **Evaluation**: use cross-validation and an independent validation set to monitor generalisation performance, and select hyperparameters on the validation set

Q51: ***What does Batch Normalization do? How is backpropagation performed? How does it behave differently in training and inference?***

- **Forward computation**: for a mini-batch $\{x_1,...,x_m\}$, on each channel compute $\mu_B=\frac{1}{m}\sum_i x_i$, $\sigma_B^2=\frac{1}{m}\sum_i(x_i-\mu_B)^2$, normalise $\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$, then scale and shift $y_i=\gamma\hat{x}_i+\beta$, where $\gamma,\beta$ are learnable parameters used to restore the network's expressive power
- **Role**: stabilises the input distribution of each layer (mitigating Internal Covariate Shift) and makes the loss surface smoother; allows larger learning rates and speeds up convergence; mitigates vanishing/exploding gradients and reduces sensitivity to weight initialisation; the noise from batch statistics has some regularising effect
- **Backpropagation**: the gradients of $\gamma,\beta$ are $\frac{\partial L}{\partial\gamma}=\sum_i\frac{\partial L}{\partial y_i}\hat{x}_i$, $\frac{\partial L}{\partial\beta}=\sum_i\frac{\partial L}{\partial y_i}$; since $\mu_B,\sigma_B^2$ both depend on all samples in the batch, the gradient with respect to the input must be chained through the mean and variance, which after simplification gives $\frac{\partial L}{\partial x_i}=\frac{\gamma}{m\sqrt{\sigma_B^2+\epsilon}}\left(m\frac{\partial L}{\partial y_i}-\sum_j\frac{\partial L}{\partial y_j}-\hat{x}_i\sum_j\frac{\partial L}{\partial y_j}\hat{x}_j\right)$
- **Training (offline / training phase)**: normalise using the mean and variance of the current mini-batch, while updating the global statistics with a moving average: $\mu_{run}\leftarrow(1-\alpha)\mu_{run}+\alpha\mu_B$, $\sigma^2_{run}\leftarrow(1-\alpha)\sigma^2_{run}+\alpha\sigma^2_B$ ($\alpha$ is the momentum, 0.1 by default in PyTorch)
- **Inference (online / inference phase)**: batch statistics are no longer computed (they cannot be computed for single-sample inference anyway); instead the fixed $\mu_{run},\sigma^2_{run}$ obtained during training are used, so BN reduces to a linear transformation that can be fused with the preceding convolution/fully connected layer to speed up inference; hence in PyTorch you need to call `model.eval()` to switch modes
- For related questions see Q25 (drawbacks of BN) and Q33 (differences between the various normalisations)

Q52: ***What is a mini-batch? Why use it?***

- Mini-batch training: at each iteration, take a small batch of samples from the training set (e.g. 32, 64, 256), compute the gradient of the average loss over this batch and update the parameters: $\theta\leftarrow\theta-\eta\frac{1}{B}\sum_{i\in batch}\nabla_\theta\ell_i(\theta)$; one pass over all the data is called an epoch
- It lies between full gradient descent (Batch GD, using all data each time) and stochastic gradient descent (SGD, using one sample each time)
- Reasons to use it:
    - **Computational efficiency**: the full dataset cannot fit in GPU memory at once; mini-batches exploit the GPU's parallel matrix computation and are much faster than per-sample updates
    - **Update frequency**: parameters can be updated many times within one epoch, so convergence is faster than Batch GD
    - **Gradient stability**: compared with single-sample SGD, averaging within the batch reduces gradient variance, making training more stable
    - **Regularisation/generalisation**: a moderate amount of noise remains in the gradient, which helps escape saddle points and sharp minima and tends to converge to flat minima that generalise better
- Choosing the batch size: too small makes training unstable and hardware utilisation low; too large may hurt generalisation and requires a correspondingly larger learning rate (linear scaling rule + warm-up); for BN, a batch that is too small makes the statistics inaccurate

Q53: ***What are the basic principles of CNN, RNN and LSTM? Which tasks is each suited to?***

- **CNN (convolutional neural network)**: learnable kernels slide over the input computing local weighted sums, with **local connectivity** and **weight sharing**; combined with activation functions and pooling, it extracts spatial features layer by layer from low to high level, and finally a fully connected layer/global pooling produces the output; it has translation invariance and few parameters. Suited to data with grid structure/local correlation such as image classification, object detection and semantic segmentation, and also usable for speech, text and time series (1D convolution)
- **RNN (recurrent neural network)**: processes a sequence time step by time step; the hidden state at each step is determined jointly by the current input and the previous hidden state: $h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b)$, output $y_t=W_{hy}h_t$; parameters are shared across time steps, and training uses BPTT (backpropagation through time). Suited to modelling short sequences, such as simple text classification and sequence labelling; drawbacks are that over long sequences repeated multiplication of gradients causes vanishing/exploding gradients, making long-term dependencies hard to learn, and it cannot be parallelised
- **LSTM (long short-term memory network)**: builds on the RNN by introducing a cell state $c_t$ and three gates: the forget gate $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$, the input gate $i_t=\sigma(W_i[h_{t-1},x_t]+b_i)$, and the output gate $o_t=\sigma(W_o[h_{t-1},x_t]+b_o)$; candidate state $\tilde{c}_t=\tanh(W_c[h_{t-1},x_t]+b_c)$, update $c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t$, $h_t=o_t\odot\tanh(c_t)$. The additive update of the cell state lets gradients propagate relatively stably over long distances, mitigating vanishing gradients and capturing long-term dependencies. Suited to long-sequence tasks such as machine translation, speech recognition, text generation and time-series forecasting
- Summary: CNNs excel at local spatial features, RNN/LSTM at temporal dependencies; GRU is a simplified version of LSTM (two gates); today Transformers have gradually replaced RNN/LSTM on many sequence tasks
