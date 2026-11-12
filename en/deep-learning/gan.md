# GAN

Q1: ***What is a Generative Adversarial Network (GAN)? What is its basic structure?***

- **A GAN consists of two networks, a generator and a discriminator: the generator turns random noise into fake samples, the discriminator judges whether samples are real or fake, and the two are trained adversarially until the generated distribution approaches the real data distribution.**
- Generator G: takes noise $z$ (e.g. a standard Gaussian) and outputs $G(z)$; its goal is to fool the discriminator.
- Discriminator D: outputs the probability that a sample comes from real data; its goal is to tell real from fake.
- Ideally a Nash equilibrium is reached: $p_g=p_{data}$ and the discriminator outputs 1/2 everywhere.
- A GAN is an implicit generative model: it can only sample and cannot directly give a sample's likelihood $p(x)$.

Q2: ***Write down the GAN objective (the minimax objective).***

- **The GAN objective is a minimax game: the discriminator maximizes the log-likelihood of telling real from fake, and the generator minimizes it, making the discriminator take fake samples for real ones.**
- For D it is just binary cross-entropy: real samples are labeled 1 and generated samples 0.
- G appears only in the second term; it wants $D(G(z))$ as close to 1 as possible.
- Training alternates: fix G and update D for a few steps (k steps in the original paper), then fix D and update G for one step.
- Key formula: $\min_G\max_D\ \mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z}[\log(1-D(G(z)))]$

Q3: ***For a fixed G, what is the optimal discriminator $D^*$?***

- **The optimal discriminator outputs, at each point, "the share of the real data density in the sum of the two densities"; when the generated distribution equals the real one, it equals 1/2 everywhere and cannot tell real from fake.**
- Derivation idea: write the objective as an integral over $x$ and maximize it pointwise to get this ratio.
- Key formula: $D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$

Q4: ***What is the relationship between GANs and the JS divergence?***

- **Plugging the optimal discriminator back into the objective turns the generator's objective into minimizing the JS divergence between the real and generated distributions (plus a constant); the global optimum is $p_g=p_{data}$.**
- The JS divergence is the average of each distribution's KL to their mixture; it is symmetric and bounded.
- At the global optimum the objective value is $-\log 4$.
- Key formula: $C(G)=-\log 4+2\cdot JS(p_{data}\,\|\,p_g)$

Q5: ***What is the non-saturating loss? Why is it used?***

- **The original generator loss saturates early in training: when the discriminator easily spots fakes, the gradient is nearly zero and the generator cannot learn; the non-saturating loss instead maximizes $\log D(G(z))$, which has the same fixed point but gives much larger gradients early on.**
- The original loss $\log(1-D(G(z)))$ is very flat when $D(G(z))\approx0$.
- Cost: under the optimal discriminator it is equivalent to an unbalanced objective (reverse KL minus JS) that penalizes "producing fake samples" more than "missing diversity", one cause of mode collapse.
- Key formula: $\min_G -\mathbb{E}_z[\log D(G(z))]$

Q6: ***Why is GAN training unstable?***

- **Because a GAN solves a non-convex non-concave minimax game where alternating gradient descent is not guaranteed to converge; combined with the difficulty of balancing the two networks and the near-zero overlap of real and generated distributions in high dimensions, training easily oscillates or suffers vanishing gradients.**
- G-D imbalance: if D is too strong, G's gradients vanish; if D is too weak, it gives no useful guidance.
- When the supports do not overlap, the JS divergence is constant at $\log 2$ and its gradient is meaningless.
- Very sensitive to hyperparameters such as learning rate, optimizer, architecture and normalization.
- No reliable loss metric tracks progress; the loss curves often do not reflect generation quality.

Q7: ***What is mode collapse?***

- **Mode collapse means the generator produces only a few kinds of samples and loses the diversity of the data distribution, e.g. generating only a few digits on MNIST, or different $z$ mapping to almost the same output.**
- Cause 1: G lowers its loss just by finding samples that currently fool D; nothing requires covering all modes.
- Cause 2: in alternating optimization G chases D's weaknesses, D corrects, and G moves to another mode, forming a cycle.
- Cause 3: the reverse KL in the non-saturating loss is mode-seeking and barely penalizes missing modes.

Q8: ***What methods can mitigate mode collapse?***

- **Mainly from three angles: use a better distance (e.g. WGAN), let the discriminator see the diversity within a batch, and regularize the discriminator to make training smoother.**
- Minibatch discrimination / minibatch standard deviation layer (ProGAN): overly similar samples become easy for the discriminator to spot.
- Unrolled GAN: when updating G, account for several future updates of D, so G does not over-chase the current D.
- Regularization (spectral normalization, gradient penalty) and conditioning (forcing different content under different conditions).
- Others: multiple generators/ensembles, experience replay (training D on past samples), diversity regularizers such as mode seeking.

Q9: ***How does the vanishing gradient problem arise in GANs?***

- **When the discriminator is near optimal and the real and generated distributions barely overlap, the JS divergence is constant and the generator gets no gradient; at the same time, a very confident discriminator saturates the original loss.**
- This creates a dilemma: if D is not trained well enough, G's gradients are inaccurate; if D is trained too well, G's gradients vanish.
- Remedies: the non-saturating loss, WGAN (Wasserstein distance instead of JS), adding noise to inputs (instance noise), gradient penalties.

Q10: ***What is the core idea of WGAN? What are the advantages of the Earth-Mover distance?***

- **WGAN replaces the JS divergence with the Wasserstein distance (Earth-Mover distance, the minimum cost of "moving" one distribution into another); even when the two distributions do not overlap, it decreases smoothly as they get closer, continuously giving the generator meaningful gradients.**
- Via Kantorovich-Rubinstein duality, the discriminator becomes a critic: it does not output a probability (the sigmoid is removed) but must be 1-Lipschitz.
- The critic's loss correlates well with sample quality and can serve as a training progress indicator.
- Key formula: $W(p_{data},p_g)=\sup_{\|f\|_L\le1}\mathbb{E}_{p_{data}}[f(x)]-\mathbb{E}_{p_g}[f(x)]$

Q11: ***How does WGAN enforce the Lipschitz constraint? What are the problems with weight clipping?***

- **The original WGAN clips the critic's parameters to $[-c,c]$ after every update to ensure Lipschitzness; but this is hard to tune: too large a $c$ makes the constraint ineffective, too small causes vanishing gradients, and it makes the critic learn overly simple functions.**
- After clipping most weights pile up at $\pm c$, limiting expressiveness.
- It works poorly with BatchNorm and momentum-based optimizers (the original paper recommends RMSProp).

Q12: ***What improvement does WGAN-GP make?***

- **WGAN-GP replaces weight clipping with a gradient penalty: it samples random interpolations on lines between real and generated samples and penalizes the critic's gradient norm at those points for deviating from 1, making training more stable and more robust to architecture.**
- Rationale: the optimal critic has gradient norm 1 along these lines.
- BatchNorm cannot be used in the critic (it makes samples interdependent and breaks the penalty); LayerNorm can be used instead, and Adam works.
- Key formula: $\lambda\,\mathbb{E}_{\hat x}\big[(\|\nabla_{\hat x}D(\hat x)\|_2-1)^2\big]$

Q13: ***What is Spectral Normalization?***

- **Spectral normalization divides each discriminator layer's weights by their largest singular value, so each layer's Lipschitz constant is at most 1 and the whole discriminator is constrained to be 1-Lipschitz; it is simple, cheap and significantly stabilizes training.**
- Activations like ReLU have Lipschitz constant ≤ 1, so the whole network stays bounded.
- The largest singular value is estimated by power iteration with just one step per training iteration, so the overhead is tiny and no extra tuning is needed.
- Widely used in SNGAN, SAGAN, BigGAN and others.
- Key formula: $\bar W=W/\sigma(W)$

Q14: ***What are the DCGAN design guidelines?***

- **DCGAN was the first work to stably combine GANs with deep convolutional networks; the core is: strided and transposed convolutions instead of pooling, BatchNorm on both sides, no fully connected layers, and suitable activation functions.**
- The discriminator downsamples with strided convolutions; the generator upsamples with transposed convolutions.
- BatchNorm in both G and D (except G's output layer and D's input layer).
- The generator uses ReLU with Tanh at the output; the discriminator uses LeakyReLU throughout.
- Adam with learning rate 0.0002 and $\beta_1=0.5$; it also showed vector arithmetic in the latent space.

Q15: ***What is a conditional GAN (cGAN)?***

- **A cGAN adds a condition $y$ (e.g. a class, text or image) to both the generator and the discriminator, so the generator learns the conditional distribution $p(x|y)$ and the generated content can be controlled.**
- Ways to inject the condition: simple concatenation, conditional BatchNorm, projection in the discriminator, or in AC-GAN the discriminator additionally predicts the class.
- Applications: class-conditional generation, text-to-image, image-to-image translation.

Q16: ***What is the difference between Pix2Pix and CycleGAN? What is the cycle-consistency loss?***

- **Pix2Pix does image translation on paired data and is a conditional GAN with an L1 reconstruction loss; CycleGAN works on unpaired data, using generators in both directions and a cycle-consistency loss that requires "translate there and back" to recover the original image.**
- Pix2Pix: U-Net generator + PatchGAN discriminator; L1 gets the low-frequency structure right and the GAN loss handles high-frequency detail. Example: semantic map → photo.
- CycleGAN: generators $G:X\to Y$ and $F:Y\to X$ plus two discriminators. Example: horse ↔ zebra.
- Cycle consistency prevents the generator from mapping every input to the same target-domain image and losing the content.
- With paired data Pix2Pix is more accurate; with only unpaired data use CycleGAN.
- Key formula: $L_{cyc}=\mathbb{E}_x\|F(G(x))-x\|_1+\mathbb{E}_y\|G(F(y))-y\|_1$

Q17: ***What is progressive growing in ProGAN?***

- **ProGAN starts training at a low resolution like 4×4 and gradually adds higher-resolution layers to both the generator and the discriminator, learning coarse structure before details; training is more stable and faster, and it can generate high-resolution images.**
- New layers are faded in: the new layer's output is blended with the upsampled output of the old layer by a coefficient $\alpha$ going from 0 to 1, avoiding a shock to already-trained layers.
- Other tricks: a minibatch standard deviation layer for diversity, equalized learning rate, pixelwise normalization in the generator.

Q18: ***What are the core designs of StyleGAN?***

- **StyleGAN first uses a mapping network to turn noise $z$ into a more disentangled intermediate latent $w$, then uses $w$ to modulate the "style" of features at every generator layer, and adds per-pixel noise to control stochastic details, achieving hierarchical, controllable generation.**
- Style injection: v1 uses AdaIN (scale and bias from an affine transform of $w$); StyleGAN2 switches to weight modulation/demodulation, removing the "droplet" artifacts.
- The generator starts from a learnable constant tensor instead of taking $z$ directly; noise inputs control details like hair and pores.
- Style mixing: coarse layers control pose and face shape, fine layers control color and texture; the truncation trick pulls $w$ toward its mean, trading diversity for quality.
- v1 kept ProGAN's progressive growing; StyleGAN2 dropped it in favor of skip/residual architectures.

Q19: ***How is GAN generation quality evaluated? What are IS and FID?***

- **IS uses an Inception classifier to check whether generated images are clearly recognizable and diverse in class, higher is better; FID compares the distributions of generated and real images in Inception feature space, reflecting both quality and diversity, lower is better, and is more widely used than IS.**
- Drawbacks of IS: no comparison with real data; depends on ImageNet classes; cannot detect intra-class mode collapse.
- FID fits a Gaussian to each set of features and computes the Fréchet distance; drawbacks are sensitivity to sample size and the Gaussian assumption.
- Precision/Recall separate quality (whether generated samples lie on the real manifold) from coverage (whether real data is covered), helping diagnose mode collapse.
- Also KID and human evaluation.
- Key formula: $FID=\|\mu_r-\mu_g\|^2+\text{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big)$

Q20: ***What are the pros and cons of GANs, VAEs, and diffusion models?***

- **GANs give sharp samples and fast one-step generation but train unstably with poor diversity; VAEs train stably with a structured latent space but give blurry samples; diffusion models give high quality, good diversity and easy control, but sample slowly.**
- GANs have no explicit likelihood and are hard to extend to complex multimodal conditional generation.
- VAEs have an encoder and can compute a likelihood lower bound (ELBO).
- Diffusion models train with a simple MSE regression objective; sampling can be accelerated by DDIM, distillation and more.
- This is called the generative modeling trilemma: high quality, high diversity and fast sampling are hard to get together — GANs sacrifice diversity, VAEs sacrifice quality, diffusion models sacrifice speed.

| Model | Quality | Diversity | Sampling speed |
|---|---|---|---|
| GAN | High | Low | Fast |
| VAE | Lower (blurry) | High | Fast |
| Diffusion | High | High | Slow |
