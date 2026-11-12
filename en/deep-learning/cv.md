# CV interview Q&A

Q1: ***In CV, what is the difference between semantic segmentation and instance segmentation?***

- **Semantic segmentation only assigns a class to every pixel and does not distinguish objects of the same class; instance segmentation additionally separates the different individuals within the same class.**
- In the figure below: in semantic segmentation all chairs have the same color; in instance segmentation each chair is a separate instance.

![https://i.stack.imgur.com/MEB9F.png](https://i.stack.imgur.com/MEB9F.png)

- The second image is the semantic segmentation result, the third is the instance segmentation result.

Q2: ***How do you evaluate an object detection model?***

- **The basic metric is IoU: the intersection area of the predicted box and the ground-truth box divided by their union, which measures how well the object is localized. On top of that, predictions are labeled TP/FP by an IoU threshold, AP is computed per class and averaged into mAP.**
- Any model that outputs bounding boxes (R-CNN, Faster R-CNN, YOLO, etc.) can be evaluated this way.
- In the figure below the red box is the prediction and the green box is the ground truth; their IoU is the localization quality.

![Untitled](cv/Untitled.png)

Q3: ***Describe the main processing steps of a standard computer vision system***

- **The standard pipeline is: acquire the image → preprocess → extract features → run a machine learning model for classification or another task.**
- Input image: a single image or a sequence of video frames from an imaging device such as a camera.
- Preprocessing: resizing, denoising, rotation, color-space conversion, etc., to standardize images so they can be compared and analyzed uniformly.
- Feature extraction: extract information such as shape and color that describes the object, producing a feature vector.
- Model: feed the features into an ML model such as a classifier to get the result.

![Untitled](cv/Untitled%201.png)

Q4: ***In object detection, how do you judge whether a feature is a good feature?***

- **A good feature lets you reliably recognize the same object under all kinds of conditions; check three properties: repeatable, distinctive, localizable.**
- Repeatable: roughly the same features are detected in two images of the same scene, robust to scale, lighting, viewpoint and noise.
- Distinctive: the neighborhood around the feature is discriminative enough to be matched reliably against other features.
- Localizable: the feature has a unique, well-defined position that does not change with viewing conditions.

Q5: **Why CNNs perform well on image processing:**

- **Because the CNN structure matches the nature of images: images have local correlation, which CNNs capture with local convolutions; and through layer-by-layer composition CNNs go from low-level edges and curves to high-level semantics while keeping spatial invariance.**
- Local correlation: each pixel is closely related to its neighbors; shuffle the pixels and the image becomes unrecognizable.
- Hierarchical features: shallow layers extract low-level features such as edges, deeper layers combine them into high-level features.

Reference: [https://www.zhihu.com/question/41233373/answer/91113816](https://www.zhihu.com/question/41233373/answer/91113816)

Q6: **The computation and purpose of the BatchNorm layer**

- **BN standardizes each channel with the mean and variance of the current batch, then restores expressiveness with a learnable scale γ and shift β; its purpose is to stabilize the input distribution of each layer so training is faster and more stable.**
- It prevents parameter updates and increasing depth from shifting the data distribution drastically.
- γ and β let the network keep what it has learned and its non-linear expressiveness, and can even restore the original distribution if needed.
- Key formula: $y=\gamma\cdot \frac{x-\mu }{\sqrt{\sigma ^2+\varepsilon }}+\beta$

Q7: **How to effectively handle the common problem of few foreground and many background samples in object detection**

- **The core idea is to make the model pay less attention to the many easy background samples and more to hard samples.**
- Use Focal Loss or OHEM (online hard example mining) to increase the loss weight of hard samples.
- During training, sample only prior boxes around the GT, ignore most of the background and keep only hard background regions.

Q8: **The difference between ROIPool and ROIAlign, and a simple implementation of ROIAlign**

- **ROIPool rounds twice (rounding the box coordinates and rounding the bin boundaries), which misaligns the features; ROIAlign does no rounding and samples at floating-point coordinates with bilinear interpolation, preserving exact spatial positions.**
- Simple implementation: divide the ROI evenly into k×k bins, take a few sample points per bin, compute each sample by bilinear interpolation from the 4 surrounding feature points, then pool.
- On large objects the two differ little; with many small objects ROIAlign is clearly more accurate.

Reference: [https://github.com/donnyyou/cv-interview](https://github.com/donnyyou/cv-interview)

Q9: ***What is the difference between Batch Norm during training and inference?***

- **During training it uses the mean and variance of the current batch and updates a global moving average; during inference it no longer depends on the batch and uses the fixed mean and variance accumulated during training.**
- At inference the mean is the average of the batch means, and the variance is the unbiased estimate from the batch variances.
- This makes the output for a single sample deterministic and independent of batch composition.

Q10: ***Why has the ReLU activation function gradually replaced the sigmoid function? What are its advantages?***

- **ReLU has a constant gradient of 1 in the positive region, which solves the vanishing gradients caused by sigmoid saturation; it is also extremely simple and fast to compute.**
- Its form is max(0, x), a non-linear function made of two linear pieces; combinations of it can approximate any function.
- One-sided suppression makes some neurons output 0, giving sparsity, reducing interdependence between parameters and easing overfitting; it is also closer to biological neurons.

Q11: ***ReLU is not differentiable at 0; how is this handled in practice?***

- **A subgradient is assigned by convention, usually defining the derivative of ReLU at 0 as 0 (this is how frameworks such as TensorFlow implement it); besides, the probability of an input being exactly 0 is tiny, so the effect is negligible.**

Reference: [https://blog.csdn.net/ningyanggege/article/details/82493023](https://blog.csdn.net/ningyanggege/article/details/82493023)

Q12: ***What is the purpose of adding pooling layers in image processing, and how does backpropagation work through a pooling layer?***

- **Pooling reduces dimensionality and computation, keeps the main features and gives invariance to small translations/deformations, which helps prevent overfitting; in backpropagation it has no parameters and simply routes the gradient back according to the forward rule.**
- Average pooling backward: the gradient of an output pixel is split evenly into n parts and given to the n pixels of its window.
- Max pooling backward: the gradient goes only to the pixel that was the maximum in the forward pass; the other pixels in the window get 0.

Q13: ***List several common deep learning optimizers and their pros and cons***

- **Common optimizers improve SGD along two lines: Momentum uses past gradients to speed up and help escape local minima, Adagrad/RMSProp give each parameter an adaptive learning rate, and Adam combines the two.**
- SGD: uses only the current first-order gradient; simple but slow and can get stuck in local optima.
- Momentum: adds the previous update direction to the gradient; converges faster and escapes local minima more easily.
- Adagrad: scales the learning rate by the accumulated squared gradients, so parameters with large gradients get their learning rate reduced faster; the drawback is that the accumulator only grows, so later updates become too slow.
- RMSProp: adds decay to Adagrad's accumulator, fixing the slow-down; Adam = Momentum + RMSProp and is the most common default.

Reference: [https://medium.com/towards-data-science/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c](https://medium.com/towards-data-science/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c)

Q14: ***FLOPs and number of parameters of a convolutional layer***

- **FLOPs means the number of floating point operations and measures model complexity. The parameter count of a conv layer is the weights per kernel (k²·C_in) plus a bias, times the number of kernels C_out; FLOPs are the multiply-adds per output point times the total number of output points.**
- The parameter count is (k²·C_in + 1)·C_out, where +1 is the bias.
- Key formula (FLOPs, counting multiply and add separately): $2C_{in}k^2\cdot C_{out}HW$, where H and W are the height and width of the output feature map.

Q15: ***What are the advantages of using the F1-Score to evaluate model performance?***

- **F1 is the harmonic mean of precision and recall, so it accounts for both, and it leans toward the smaller of the two: F1 is high only when both P and R are high.**
- The harmonic mean is used instead of the arithmetic mean so that one very high metric cannot hide a very low one.
- Under class imbalance it reflects real model performance better than accuracy.
- Key formula: $F1 = \frac{2PR}{P+R}$

Q16: ***What is the difference between ROI Pooling and ROI Align?***

- **Both turn candidate boxes of varying sizes into fixed-size features (e.g. 7×7) for the detection head; the main difference is that ROI Pooling quantizes (rounds) twice, while ROI Align does no quantization and uses bilinear interpolation instead.**
- ROI Pooling steps: divide the ROI coordinates by the stride to map them onto the feature map and round (first quantization) → split evenly into k×k bins and round the bin boundaries (second quantization) → max pooling within each bin.
- Why ROI Pooling matters: all ROIs share the convolutional features of the whole image, so there is no need to run the convolution once per box as in R-CNN, which is much faster.
- ROI Align: keeps floating-point coordinates, takes several (usually 4) sample points per bin, computes their values by bilinear interpolation and then pools; in the backward pass gradients flow back according to the interpolation weights.
- Misalignment from quantization: a 1-pixel offset on the feature map is about 16 pixels in the original image; this matters little for classification but a lot for mask prediction and small objects, which is why Mask R-CNN switched to ROI Align.

Q17: ***In feature fusion, what is the difference between concat and add?***

- **add sums values at corresponding positions and keeps the number of channels, so each feature carries more information; concat stacks along the channel dimension and increases the number of channels, so there are more features but each one carries the same information.**
- add is cheaper and requires both inputs to have exactly the same shape (e.g. ResNet).
- concat keeps each branch's original information and lets later layers decide how to fuse it (e.g. DenseNet, U-Net).

Q18: ***Explain depthwise separable convolution***

- **Depthwise separable convolution splits a standard convolution into two steps: first a spatial convolution on each channel separately (depthwise), then a 1×1 convolution to fuse channels (pointwise), greatly reducing parameters and computation.**
- Depthwise convolution: each input channel gets its own k×k kernel; the output has as many channels as the input, and the per-channel results are not summed.
- Pointwise convolution: a 1×1 convolution then combines channels and changes the number of channels.
- It is the core building block of lightweight networks such as MobileNet.

Q19: ***Compute the FLOPs of a depthwise separable convolution***

- **The total FLOPs are the sum of the depthwise part and the 1×1 pointwise part, roughly 1/C_out + 1/k² of a standard convolution.**
- Depthwise part: one k×k convolution per channel, about 2k²·HW·C_in.
- Pointwise part: one C_in → C_out linear transform per position, about 2·C_in·HW·C_out.
- Key formula (total): $2HWC_{in}(k^2 + C_{out})$

Q20: ***Briefly describe the principle of the LBP algorithm***

- **LBP is a local texture descriptor: in a 3×3 window, each of the 8 surrounding pixels is compared with the center pixel, marked 1 if it is larger and 0 otherwise; the resulting 8-bit binary number (256 possibilities) is the LBP value of the center pixel.**
- Advantages: rotation invariance (in the improved variants), robustness to lighting changes, low feature dimension and fast computation.

Q21: ***Briefly describe the principle of the HOG algorithm***

- **HOG describes object shape by computing histograms of gradient orientations over local image regions; it is commonly used for pedestrian detection.**
- Divide the image into small cells, compute the gradient magnitude and orientation of each pixel, and build an orientation histogram per cell.
- Group cells into blocks and apply contrast normalization within each block for better invariance to lighting and shadows.
- Concatenating the descriptors of all blocks gives the HOG feature of the whole image.

Q22: ***What edge detection operators are there, and how do they compare?***

- **Edge detection operators fall into three main groups: first-derivative operators that look for gradient extrema (Roberts, Prewitt, Sobel), second-derivative operators that look for zero crossings (Laplacian), and the combined Canny operator.**
- First-derivative operators: simple and fast, but edges are thick and fairly sensitive to noise.
- Second-derivative operators: accurate localization but even more sensitive to noise.
- Canny: Gaussian smoothing + gradient + non-maximum suppression + double threshold; thin, continuous edges and the best results, but more computation.

Q23: ***Common image interpolation methods***

- **The three common ones are nearest neighbor, bilinear and bicubic interpolation, in increasing order of quality and computational cost.**
- Nearest neighbor: takes the value of the nearest pixel; fastest but produces jagged edges.
- Bilinear: linear interpolation in each of the two directions using the 4 surrounding pixels; smooth results.
- Bicubic: weights the 16 surrounding pixels with a cubic polynomial; preserves detail best but is the slowest.

Q24: ***The principle of the FAST algorithm***

- **FAST is a fast corner detector: look at the 16 pixels on a circle around candidate point p; if n contiguous pixels are all clearly brighter or all clearly darker than p, p is a corner.**

![Untitled](cv/Untitled%202.png)

- The advantage is high speed; the drawbacks are strong sensitivity to noise and no scale or rotation invariance.

Q25: **Briefly describe the principle of the BRIEF algorithm**

- **BRIEF is a binary feature descriptor: randomly choose n point pairs in the s×s neighborhood of a keypoint, compare the intensities of each pair to get 0/1, and concatenate the bits into an n-bit binary string as the descriptor.**
- Apply Gaussian filtering first, because comparing random point pairs is very sensitive to noise.
- n is usually 128, 256 or 512; matching uses the Hamming distance and is very fast.
- The drawback is the lack of rotation invariance.

Q26: ***Briefly describe the principle of the ORB algorithm***

- **ORB = FAST for keypoint detection + BRIEF for description, and it fixes BRIEF's lack of rotation invariance.**
- The method is the intensity centroid: compute the intensity centroid of the keypoint's neighborhood and use the vector from the keypoint to the centroid as its main orientation.
- The patch is rotated to this orientation before computing BRIEF, so the descriptor stays the same when the image rotates, giving rotation invariance.

![Untitled](cv/Untitled%203.png)

Q27: **Briefly describe SIFT features; what is the difference between SIFT and SURF features?**

- **SIFT is a local feature that finds keypoints and their orientations across scale space; it is invariant to rotation, scale and brightness changes and fairly stable under viewpoint changes, affine transforms and noise. SURF is an accelerated SIFT that approximates it with box filters and integral images, making it much faster with comparable robustness.**
- SIFT's four steps: scale-space extrema detection, keypoint localization, orientation assignment, keypoint description.
- Main differences:

|  | SIFT | SURF |
| --- | --- | --- |
| Scale space | DoG convolved with images at different scales | Box filters at different scales convolved with the original image |
| Keypoint detection | Non-maximum suppression + remove low-contrast points + Hessian to remove edge points | Hessian matrix to find candidates first, then non-maximum suppression |
| Orientation | Gradient histogram over a square region, take the peak direction | Haar wavelet responses in sectors of a circular region, take the strongest direction |
| Descriptor | 4×4 subregions × 8 orientations = 128 dims | 4×4 subregions × 4 Haar statistics = 64 dims |

Reference: [Differences between Sift and Surf (eternity1118_'s blog, CSDN)](https://blog.csdn.net/eternity1118_/article/details/51152162)

Q28: **What losses do you know for image semantic segmentation tasks?**

- **The most common is per-pixel cross-entropy; to handle class imbalance and hard samples there are weighted cross-entropy and Focal Loss; to directly optimize overlap there are Dice Loss and IoU Loss.**
- Cross-entropy: compute the cross-entropy between each pixel's predicted distribution and its one-hot label, then average over the image.
- Weighted cross-entropy: give classes with few pixels larger weights to ease class imbalance.
- Focal Loss: automatically down-weights easy samples and focuses training on hard ones.
- Dice / IoU Loss: directly target the overlap between prediction and label, friendlier to small objects and imbalance.
- Key formula (per-pixel cross-entropy): $-\sum_{c} y_c\log \hat{y}_c$

Reference: [Image Segmentation: Architectures, Losses, Datasets, and Frameworks - neptune.ai](https://neptune.ai/blog/image-segmentation)

Q29: **In edge detection, compare the Sobel operator and the Canny operator**

- **Sobel is simple and fast but gives thick edges; Canny adds non-maximum suppression and double thresholding, giving thinner, smoother and more continuous edges at the cost of more complexity and lower speed.**

Reference: [Microsoft Word - p107.doc (utm.my)](https://engineering.utm.my/comp/wp-content/uploads/sites/2/2013/04/Comparison-of-Canny-and-Sobel-Edge-Detection-in-MRI-Images.pdf)

Q30: **Why normalize images before feeding them into the network?**

- **Normalization brings inputs to a common scale and removes differences in units, so gradient descent converges faster and more stably, and sometimes accuracy improves too.**
- Removes units: makes different features comparable.
- Faster convergence: the loss surface is more "round", so gradient descent oscillates less.
- Better accuracy: for distance-based methods such as KNN, it prevents features with a large range from dominating the distance.

Q31: ***What problems might you encounter when using YOLO for object detection?***

- **Each grid cell in early YOLO can predict only a few objects, so it does poorly on groups of small objects (such as a line of ants); it is also fairly sensitive to lighting and other environmental changes, and its computational cost makes edge deployment difficult.**
- Small and dense objects: objects are easily missed when several fall into one grid cell.
- Environmental changes: performance is unstable in real scenes with strongly varying lighting.
- Deployment: it consumes considerable compute, so edge devices need pruning, quantization or a lightweight version.

Reference: [YOLO Algorithm for Object Detection Explained [+Examples] (v7labs.com)](https://www.v7labs.com/blog/yolo-object-detection)

## Advantages of deep learning in the image domain

Q32: ***Why does deep learning perform so well in image processing and computer vision?***

- **Because deep networks learn hierarchical features directly from pixels end to end, far better than hand-crafted SIFT or HOG; on top of that, CNNs match the structure of images, and large datasets and GPU compute make training very deep networks possible.**
- Hierarchical features: shallow layers learn edges and textures, middle layers learn parts, deep layers learn object semantics.
- Inductive bias: local connectivity and weight sharing match the local correlation of images, with few parameters, easy training and some translation invariance (see Q5).
- End-to-end optimization: feature extraction and the downstream task are trained jointly, avoiding error accumulation across stages.
- Data, compute and tricks: ImageNet/COCO, data augmentation, GPUs, plus ReLU, BN and residual connections; pretrained models also transfer well to downstream tasks.

## CNN basic building blocks

Q33: ***What are the roles of the convolutional layer, the pooling layer and the fully connected layer in a CNN?***

- **Convolutional layers extract local features, pooling layers downsample to reduce dimensionality and provide some invariance, and fully connected layers map features to classes for classification.**
- Convolutional layer: kernels slide over the input computing local weighted sums; local connectivity + weight sharing keep parameters few; stacking layers enlarges the receptive field and builds high-level semantics from low-level features.
- Pooling layer (see Q12): max/average downsampling that reduces computation, enlarges the receptive field and gives invariance to small translations/deformations.
- Fully connected layer: integrates global information and acts as the "classifier"; drawbacks are many parameters (most of VGG's parameters are here) and a fixed input size, so modern networks often replace it with global average pooling (GAP).

Q34: ***Why can a 1×1 convolution reduce or change the number of channels? How does it introduce nonlinearity?***

- **A 1×1 convolution applies a linear transform to the channel vector at each pixel (a per-pixel fully connected layer); the number of output channels equals the number of kernels, so fewer than the input means reduction and more means expansion, with the spatial size unchanged. It is linear itself; the non-linearity comes from the BN + ReLU that follow it.**
- Less computation: the Inception/ResNet bottleneck first reduces 256 channels to 64, applies the 3×3, then expands back to 256, greatly cutting computation and parameters.
- More non-linearity: deepens the network without changing the receptive field (the Network in Network idea).
- Cross-channel fusion: linearly combines features from different channels, enabling information exchange between channels.
- Key formula: $y_{i,j} = W x_{i,j} + b,\ W \in \mathbb{R}^{C_{out}\times C_{in}}$
- See also: [Deep Learning Fundamentals](dl-fundamentals.md) Q15; [CNN](cnn.md) Q16

## Image classification models (VGG, ResNet, Inception)

Q35: ***Briefly describe the classic image classification networks VGG, ResNet and its variants, and Inception V2/V3***

- **VGG makes the network deep by stacking small 3×3 convolutions; ResNet uses residual connections to solve the degradation problem of deep networks and can train hundreds of layers; Inception extracts features with parallel multi-scale convolution branches, and V2/V3 mainly add BN and convolution factorization.**
- VGG: all 3×3 convolutions + 2×2 pooling, a regular structure; two 3×3 equal one 5×5 with fewer parameters; drawback: many parameters (about 138M, mostly in the fully connected layers).
- ResNet: learns residuals; ResNet50 and above use a bottleneck of 1×1 reduce → 3×3 → 1×1 expand. Variants: Pre-activation ResNet (BN and ReLU moved before the conv), ResNeXt (group convolution), Wide ResNet (wider), DenseNet (concat with all previous layers), SE-ResNet (channel attention).
- Inception V1 (GoogLeNet): parallel 1×1, 3×3, 5×5 convolutions and pooling concatenated along channels, with 1×1 reduction to save computation.
- Inception V2: introduces BN for faster convergence and replaces a 5×5 with two 3×3.
- Inception V3: asymmetric factorization (e.g. 7×7 → 1×7 + 7×1), an improved downsampling module that avoids representational bottlenecks, Label Smoothing, input size 299×299.

Q36: ***What is the main innovation of ResNet? What are the structure and principle of the residual block? Why do residual connections improve the training of deep networks, alleviate vanishing gradients, and mitigate performance degradation as depth increases?***

- **ResNet's core innovation is residual learning with identity shortcuts: each block only learns "the change on top of its input", and when there is nothing to learn it falls back to the identity, so adding depth at least does not make things worse and networks with hundreds of layers become trainable.**
- Structure: the residual branch is two 3×3 convolutions (basic block) or 1×1-3×3-1×1 (bottleneck), each conv followed by BN, with ReLU after the addition; when dimensions differ the shortcut uses a strided 1×1 convolution as a projection.
- Easing vanishing gradients: in backpropagation the gradient can flow straight back to shallow layers through the shortcut without decay, instead of being a product of many terms smaller than 1.
- Easing degradation: degradation means training error rises as the network gets deeper; it is an optimization difficulty, not overfitting. Plain deep networks struggle to learn the identity, while the residual structure makes the identity the default solution (zero residual), so added layers only need to learn small perturbations.
- Another view: a residual network acts like an ensemble of many sub-networks of different depths; information and gradients have short paths and the loss surface is smoother.
- Key formula: $y = F(x) + x$

## Object detection (Faster R-CNN, Mask R-CNN)

Q37: ***How do Faster R-CNN and Mask R-CNN work, and what are the differences between them?***

- **Faster R-CNN is a two-stage detector: the backbone extracts features, the RPN generates candidate boxes, ROI Pooling turns each box into a fixed-size feature, and the detection head does classification and box regression. Mask R-CNN adds a parallel mask branch for instance segmentation and replaces ROI Pooling with the more precise ROI Align.**
- Faster R-CNN: the RPN generates anchor-based proposals filtered by NMS; the head performs K+1-class classification and class-specific box regression, followed by another NMS.
- Mask R-CNN: usually ResNet + FPN for multi-scale features; the mask branch is a small FCN that outputs one 28×28 binary mask per class for each ROI.
- Main differences:

|  | Faster R-CNN | Mask R-CNN |
| --- | --- | --- |
| Task | Object detection (class + box) | Detection + instance segmentation (plus pixel-level mask) |
| ROI feature extraction | ROI Pooling (two quantizations) | ROI Align (bilinear interpolation, no quantization) |
| Head branches | Classification + regression | Classification + regression + mask (parallel) |
| Loss | $L_{cls}+L_{box}$ | $L_{cls}+L_{box}+L_{mask}$ |

Q38: ***What is the RPN? Describe its network structure in detail***

- **The RPN is a small fully convolutional network introduced by Faster R-CNN that quickly generates candidate boxes from anchors on the shared feature map, replacing the slow Selective Search and allowing end-to-end training.**
- Anchors: k reference boxes are preset at every feature-map position; the original paper uses 3 scales × 3 aspect ratios, so k=9.
- Structure: the feature map goes through a 3×3 convolution (512 channels + ReLU), followed by two parallel 1×1 convolutions: the classification branch outputs 2k channels (foreground/background scores per anchor) and the regression branch outputs 4k channels (4 offsets per anchor).
- Proposal generation: apply the offsets to the anchors → clip to the image and remove tiny boxes → take the top-N by score → NMS (IoU 0.7) → about 2000 proposals in training and 300 at test time.
- Sample assignment: IoU > 0.7 with a GT is positive, < 0.3 is negative, the rest are ignored; 256 anchors are sampled per image with a positive:negative ratio close to 1:1. In FPN the RPN head is shared across levels and each level uses a single scale.
- Key formula (offset parameterization): $t_x=\frac{x-x_a}{w_a},\ t_y=\frac{y-y_a}{h_a},\ t_w=\log\frac{w}{w_a},\ t_h=\log\frac{h}{h_a}$

Q39: ***What does the RPN loss consist of, and how is it computed? Is the classification loss binary or multi-class?***

- **The RPN loss consists of a classification loss and a box regression loss; the classification is binary (foreground vs background) and only decides whether an anchor contains an object; the actual multi-class classification is done by the second-stage detection head.**
- Classification loss: binary cross-entropy, computed on both sampled positive and negative anchors.
- Regression loss: smooth L1, computed only on positives (negatives have no matching GT). Smooth L1 behaves like a squared loss with smooth gradients for small errors and is linear for large errors, so it is insensitive to outliers.
- Normalization and weighting: the classification term is divided by the mini-batch size (256), the regression term by the number of anchor locations (about 2400), and λ=10 makes the two terms comparable; in practice normalizing by the number of positives with λ=1 is also common.
- Key formula: $L = \frac{1}{N_{cls}}\sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}}\sum_i p_i^* L_{reg}(t_i, t_i^*)$

Q40: ***What does the Faster R-CNN loss consist of, and what does each part mean?***

- **The total Faster R-CNN loss is the sum of the losses of the two stages, RPN and detection head, four terms in all: the RPN's foreground/background classification and box regression, and the head's K+1-class classification and box regression.**
- RPN classification: binary cross-entropy, learning whether an anchor contains an object.
- RPN regression: smooth L1 on the offsets from positive anchors to their GT, making proposals closer to real objects (see Q39).
- Head classification: K+1-class (including background) softmax cross-entropy for each ROI; head regression: smooth L1 only for foreground ROIs and only on the offsets of their GT class, further refining the proposals.
- Training details: 128 or 512 ROIs are sampled per image, IoU ≥ 0.5 is positive and [0.1, 0.5) negative, with about 25% positives; the original paper used four-step alternating training, while nowadays the four terms are usually summed and trained end to end.
- Key formula: $L = L_{cls}^{rpn} + L_{reg}^{rpn} + L_{cls}^{det} + L_{reg}^{det}$

Q41: ***What components does Mask R-CNN add compared with Faster R-CNN, and why? What is the basic structure of Mask R-CNN?***

- **Mask R-CNN's structure is: ResNet + FPN backbone → RPN for proposals → ROI Align for features → three parallel heads for classification, box regression and mask. Compared with Faster R-CNN it mainly adds the mask branch, replaces ROI Pooling with ROI Align, and is commonly paired with FPN, all to achieve precise pixel-level instance segmentation.**
- Mask branch: a small fully convolutional network that outputs one 28×28 mask per class for each ROI; convolutions rather than fully connected layers are used to preserve spatial structure.
- ROI Align: removes the misalignment caused by quantization, significantly improving mask accuracy, especially for small objects.
- FPN: fuses high-level semantics with low-level detail, improving multi-scale and small-object results (not unique to Mask R-CNN, but its common configuration).
- Decoupling classification and mask: each class predicts its own binary mask independently (per-pixel sigmoid + binary cross-entropy, computed only on the GT-class channel), so classes do not compete and the class is decided by the classification branch; this works much better than per-pixel softmax.
- At inference, the detection boxes are obtained first, then the mask branch runs on those boxes; the mask of the predicted class is resized to the box and binarized at 0.5.
