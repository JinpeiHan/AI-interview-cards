# GAN

Q1: ***What is a Generative Adversarial Network (GAN)? What is its basic structure?***

- A GAN consists of two networks: a **generator G** and a **discriminator D**.
    - Generator G: takes random noise $z\sim p_z(z)$ (e.g., a standard Gaussian) as input and outputs a fake sample $G(z)$. Its goal is to make the distribution of generated samples $p_g$ approach the real data distribution $p_{data}$.
    - Discriminator D: takes a sample as input and outputs the probability $D(x)\in[0,1]$ that it comes from the real data. Its goal is to distinguish real samples from generated ones.
- The two play a **zero-sum game**: G tries to fool D while D tries to catch G. They are trained alternately and, ideally, reach a **Nash equilibrium** where $p_g=p_{data}$ and $D(x)\equiv\frac{1}{2}$.
- A GAN is an **implicit generative model**: it can only sample and cannot directly give the likelihood $p(x)$ of a sample.

Q2: ***Write down the GAN objective (the minimax objective).***

$$\min_G\max_D V(D,G)=\mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

- For D, this is a **binary cross-entropy**: real samples have label 1, generated samples have label 0, and D maximizes $V$.
- For G, G only appears in the second term; G minimizes $\log(1-D(G(z)))$, i.e., it pushes $D(G(z))$ toward 1.
- Training usually **alternates**: fix G and update D for several steps (k steps in the original paper), then fix D and update G for one step.

Q3: ***For a fixed G, what is the optimal discriminator $D^*$?***

- Write $V$ as an integral over $x$: $V=\int_x \big[p_{data}(x)\log D(x)+p_g(x)\log(1-D(x))\big]dx$.
- Maximize $a\log y+b\log(1-y)$ pointwise for each $x$; the maximum is attained at $y=\frac{a}{a+b}$, so:

$$D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$$

- When $p_g=p_{data}$, $D^*(x)=\frac{1}{2}$ and the discriminator cannot tell real from fake.

Q4: ***What is the relationship between GANs and the JS divergence?***

- Substituting $D^*$ into the objective gives:

$$C(G)=\max_D V(D,G)=-\log 4+2\cdot JS(p_{data}\,\|\,p_g)$$

- where $JS(P\|Q)=\frac{1}{2}KL(P\|M)+\frac{1}{2}KL(Q\|M)$ and $M=\frac{P+Q}{2}$.
- Therefore, with an optimal discriminator, **optimizing the generator is equivalent to minimizing the JS divergence between the real and generated distributions**. The global optimum is $p_g=p_{data}$, where $C(G)=-\log 4$.

Q5: ***What is the non-saturating loss? Why is it used?***

- The original generator loss $\min_G \mathbb{E}_z[\log(1-D(G(z)))]$ has a problem early in training: when G is weak, D easily identifies fake samples, $D(G(z))\approx 0$, and $\log(1-D(G(z)))$ **saturates**, so the gradient is tiny and G cannot learn.
- The **non-saturating loss** instead uses:

$$\max_G \mathbb{E}_z[\log D(G(z))]\quad\Leftrightarrow\quad \min_G -\mathbb{E}_z[\log D(G(z))]$$

- Both have the same fixed point, but when $D(G(z))\approx0$ the non-saturating loss has large gradients, **providing stronger gradients early in training**.
- The cost: under the optimal D, this loss is equivalent to optimizing $KL(p_g\|p_{data})-2JS(p_{data}\|p_g)$. This objective is unbalanced and **penalizes producing fake-looking samples more than it penalizes missing diversity**, which is one of the causes of mode collapse.

Q6: ***Why is GAN training unstable?***

- **Non-convex–non-concave minimax game**: alternating gradient descent is not guaranteed to converge to a Nash equilibrium and may oscillate or cycle.
- **Imbalance between G and D**: if D is too strong, G's gradients vanish; if D is too weak, it cannot give G useful guidance.
- **Non-overlapping supports**: in high-dimensional spaces, real and generated data usually lie on low-dimensional manifolds that barely overlap. The JS divergence is then constant at $\log 2$ and its gradient is meaningless.
- **Hyperparameter sensitivity**: learning rate, optimizer, architecture, normalization, etc. all significantly affect results.
- There is no reliable loss metric that indicates training progress; the G and D loss curves often do not reflect generation quality.

Q7: ***What is mode collapse?***

- Mode collapse means the generator produces samples from only **a few modes**, losing the diversity of the data distribution. For example, on MNIST it generates only a few digits, or different $z$ map to nearly identical outputs.
- Causes:
    - G only needs to find samples that currently fool D to lower its loss; nothing explicitly requires it to cover all modes.
    - In alternating optimization, G chases D's weaknesses, D corrects them, then G moves to another mode, forming a cycle.
    - The reverse KL term $KL(p_g\|p_{data})$ in the non-saturating loss is **mode-seeking** and places little penalty on missing modes.

Q8: ***What methods can mitigate mode collapse?***

- **Better losses/distances**: use Wasserstein-based methods such as WGAN and WGAN-GP, which give smoother gradients.
- **Minibatch discrimination / minibatch standard deviation**: let the discriminator see statistics across samples in a batch, so overly similar samples are easy to detect (ProGAN uses a minibatch standard deviation layer).
- **Unrolled GAN**: when updating G, take into account several future updates of D, so G does not over-chase the current D.
- **Regularization**: spectral normalization, gradient penalties, etc., to make D smoother.
- **Conditioning**: use class labels or other conditions to force G to generate different content under different conditions.
- **Multiple generators/ensembles**, **experience replay** (training D on historical generated samples), etc.
- Add **diversity/mode-coverage** regularizers, e.g., encouraging different $z$ to produce different outputs (such as mode-seeking regularization).

Q9: ***How does the vanishing gradient problem arise in GANs?***

- When the discriminator is trained **close to optimal** and the supports of $p_{data}$ and $p_g$ do not overlap (or overlap on a measure-zero set), $JS(p_{data}\|p_g)=\log 2$ is constant and the gradient for G is zero.
- From the original loss: when D confidently rejects fake samples, $D(G(z))\to0$, $\log(1-D(G(z)))$ saturates, and the gradient goes to 0.
- This creates a dilemma: if D is not trained well enough, G's gradients are inaccurate; if D is trained too well, G's gradients vanish.
- Remedies: non-saturating loss, WGAN (replacing the JS divergence with the Wasserstein distance), adding noise to inputs (instance noise), gradient penalties, etc.

Q10: ***What is the core idea of WGAN? What are the advantages of the Earth-Mover distance?***

- WGAN replaces the JS divergence with the **Wasserstein-1 distance (Earth-Mover distance)**:

$$W(p_{data},p_g)=\inf_{\gamma\in\Pi(p_{data},p_g)}\mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$$

- Intuitively, it is the minimum cost of "moving" one distribution into the other.
- **Advantage**: even when the two supports do not overlap, the Wasserstein distance is still **continuous and differentiable almost everywhere**, and it decreases smoothly as the distributions get closer, providing meaningful gradients.
- Using the **Kantorovich-Rubinstein duality**:

$$W(p_{data},p_g)=\sup_{\|f\|_L\le1}\mathbb{E}_{x\sim p_{data}}[f(x)]-\mathbb{E}_{x\sim p_g}[f(x)]$$

- The discriminator becomes a **critic** $f$ (it no longer outputs a probability; the final sigmoid is removed), and $f$ must satisfy a **1-Lipschitz** constraint.
- The critic loss correlates well with sample quality and can be used as an indicator of training progress.

Q11: ***How does WGAN enforce the Lipschitz constraint? What are the problems with weight clipping?***

- The original WGAN uses **weight clipping**: after each update, all critic parameters are clipped to $[-c,c]$, which guarantees the function is K-Lipschitz.
- Problems with weight clipping:
    - If $c$ is too large: slow convergence and the Lipschitz constraint is ineffective; if $c$ is too small: gradients easily vanish.
    - Most weights end up concentrated at $\pm c$, and the critic tends to learn **overly simple functions**, limiting its capacity.
    - It works poorly with BatchNorm and momentum-based optimizers (the original paper recommends RMSProp).

Q12: ***What improvement does WGAN-GP make?***

- WGAN-GP replaces weight clipping with a **gradient penalty**:

$$L=\mathbb{E}_{\tilde x\sim p_g}[D(\tilde x)]-\mathbb{E}_{x\sim p_{data}}[D(x)]+\lambda\,\mathbb{E}_{\hat x\sim p_{\hat x}}\big[(\|\nabla_{\hat x}D(\hat x)\|_2-1)^2\big]$$

- $\hat x=\epsilon x+(1-\epsilon)\tilde x$ with $\epsilon\sim U[0,1]$, i.e., points sampled randomly along lines between real and generated samples.
- Rationale: the optimal critic has gradient norm 1 along these lines, so deviations of the gradient norm from 1 are penalized.
- Note: **BatchNorm cannot be used in the critic** (it makes samples depend on each other and breaks the per-sample gradient penalty); LayerNorm can be used instead. Adam can be used.
- Effect: more stable training and more robustness to architecture choices.

Q13: ***What is Spectral Normalization?***

- Divide each weight matrix $W$ of the discriminator by its **spectral norm** (largest singular value) $\sigma(W)$: $\bar W=\frac{W}{\sigma(W)}$.
- The Lipschitz constant of a linear layer equals its spectral norm, and activations such as ReLU have Lipschitz constant ≤1, so the Lipschitz constant of the whole network is bounded by 1.
- $\sigma(W)$ is estimated with **power iteration**, using only one step per training iteration, so the overhead is small.
- Advantages: simple to implement, cheap, no extra tuning, and significantly stabilizes training; widely used in SNGAN, SAGAN, BigGAN, etc.

Q14: ***What are the DCGAN design guidelines?***

- Replace pooling in the discriminator with **strided convolutions**, and use **transposed (fractional-strided) convolutions** for upsampling in the generator.
- Use **BatchNorm** in both the generator and the discriminator (except the generator output layer and the discriminator input layer).
- Remove **fully connected hidden layers** in deeper architectures.
- Use **ReLU** in all generator layers except the output layer, which uses **Tanh**.
- Use **LeakyReLU** in all discriminator layers.
- For optimization, use Adam with a small learning rate (0.0002 in the paper) and $\beta_1$ lowered to 0.5.
- Significance: DCGAN was the first to stably combine GANs with deep convolutional networks, and it demonstrated **vector arithmetic** properties in the latent space.

Q15: ***What is a conditional GAN (cGAN)?***

- **Conditioning information** $y$ (e.g., a class label, text, or image) is fed into both the generator and the discriminator:

$$\min_G\max_D\ \mathbb{E}_{x}[\log D(x|y)]+\mathbb{E}_{z}[\log(1-D(G(z|y)|y))]$$

- The generator learns the conditional distribution $p(x|y)$, so the **generated content can be controlled**.
- Ways to inject the condition: simple concatenation, conditional BatchNorm, **projection** in the discriminator (e.g., the Projection Discriminator), an auxiliary class-prediction head in the discriminator as in AC-GAN, etc.
- Applications: class-conditional image generation, text-to-image generation, image-to-image translation, etc.

Q16: ***What is the difference between Pix2Pix and CycleGAN? What is the cycle-consistency loss?***

- **Pix2Pix**:
    - For image-to-image translation with **paired data**, e.g., semantic map → photo, edge map → object.
    - It is a conditional GAN: the generator is a **U-Net** and the discriminator is a **PatchGAN** (it classifies local image patches as real or fake).
    - Loss = cGAN loss + **L1 reconstruction loss**; L1 ensures correct low-frequency structure, while the GAN loss handles high-frequency detail.
- **CycleGAN**:
    - For **unpaired data**, e.g., horse ↔ zebra, photo ↔ painting.
    - Two generators $G:X\to Y$ and $F:Y\to X$, and two discriminators $D_X,D_Y$.
    - Introduces the **cycle-consistency loss**:

    $$L_{cyc}=\mathbb{E}_x[\|F(G(x))-x\|_1]+\mathbb{E}_y[\|G(F(y))-y\|_1]$$

    - Cycle consistency requires that a translated image can be translated back, preventing the generator from mapping all inputs to the same target-domain image and losing the input content.
- Summary: with paired data, Pix2Pix is better and more precise; with only unpaired data from two domains, use CycleGAN.

Q17: ***What is progressive growing in ProGAN?***

- Training starts at **low resolution** (e.g., 4×4) and gradually **adds higher-resolution layers to both G and D**, until the target resolution is reached.
- New layers are added with a **fade-in**: the new layer's output is linearly blended with the upsampled output of the old layers using a coefficient $\alpha$ going from 0 to 1, avoiding a sudden shock to the already-trained low-resolution layers.
- Benefits: coarse structure is learned first, then details, so **training is more stable and faster**, enabling high-resolution generation.
- Other tricks: minibatch standard deviation to improve diversity, equalized learning rate, and pixelwise normalization in the generator.

Q18: ***What are the core designs of StyleGAN?***

- **Mapping network**: an MLP maps $z\in\mathcal{Z}$ to an intermediate latent space $w\in\mathcal{W}$. $\mathcal{W}$ does not have to follow a fixed prior distribution and is therefore more **disentangled**.
- **Style injection**:
    - StyleGAN v1 uses **AdaIN**: $\text{AdaIN}(x_i,y)=y_{s,i}\frac{x_i-\mu(x_i)}{\sigma(x_i)}+y_{b,i}$, where the scale $y_s$ and bias $y_b$ come from an affine transform of $w$ and control the "style" of the features at each layer.
    - StyleGAN2 switches to **weight modulation/demodulation**, applying the style directly to the convolution weights, which removes the "droplet" artifacts caused by AdaIN.
- **Constant input**: the generator starts from a learned constant tensor rather than taking $z$ directly as input.
- **Noise inputs**: per-pixel noise is added at each layer to control **stochastic details** such as hair and pores.
- **Style mixing**: different layers use different $w$; coarse-resolution layers control pose and face shape, fine-resolution layers control color and texture.
- StyleGAN v1 keeps ProGAN's progressive growing; StyleGAN2 removes progressive growing and uses skip/residual architectures instead.
- **Truncation trick**: pull $w$ toward the mean $\bar w$, trading diversity for quality.

Q19: ***How is GAN generation quality evaluated? What are IS and FID?***

- **Inception Score (IS)**:

    $$IS=\exp\big(\mathbb{E}_x\,KL(p(y|x)\,\|\,p(y))\big)$$

    - A pretrained Inception network classifies the generated images. A sharper $p(y|x)$ means images are **clear and recognizable**; a more uniform marginal $p(y)$ means **high diversity**. Higher IS is better.
    - Drawbacks: it does not compare against real data; it depends on ImageNet classes and does not transfer to other domains; it cannot detect intra-class mode collapse.
- **Fréchet Inception Distance (FID)**:

    $$FID=\|\mu_r-\mu_g\|^2+\text{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big)$$

    - Fit Gaussians to the Inception features of real and generated images and compute the Fréchet distance between them; **lower is better**.
    - It reflects both quality and diversity and is more commonly used than IS; drawbacks are sensitivity to sample size (it is biased) and the Gaussian assumption on features.
- **Precision / Recall**:
    - Precision measures how many generated samples fall within the real data manifold (**quality/fidelity**).
    - Recall measures how much of the real data is covered by the generated distribution (**diversity/coverage**).
    - They split the single FID number into quality and diversity dimensions, which helps diagnose problems like mode collapse.
- Other options include KID and human evaluation.

Q20: ***What are the pros and cons of GANs, VAEs, and diffusion models?***

- **GAN**:
    - Pros: **sharp, crisp** samples; sampling needs only **one forward pass**, so it is fast.
    - Cons: unstable training, mode collapse, limited diversity; no explicit likelihood; harder to scale to complex multimodal conditional generation.
- **VAE**:
    - Pros: stable training; an explicit encoder and a structured latent space; a computable lower bound on the likelihood (ELBO).
    - Cons: samples are often **blurry**, and generation quality is usually below GANs and diffusion models.
- **Diffusion models**:
    - Pros: stable training (a simple MSE regression objective), **good mode coverage and high diversity**, high generation quality, and easy conditioning and scaling.
    - Cons: **sampling requires many iterative steps and is slow** (it can be accelerated with DDIM, distillation, etc.).
- This trade-off is often called the **generative learning trilemma**: high quality, high diversity, and fast sampling are hard to achieve simultaneously — GANs sacrifice diversity, VAEs sacrifice quality, and diffusion models sacrifice speed.
