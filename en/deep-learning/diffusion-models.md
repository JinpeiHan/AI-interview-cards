# Diffusion Models

Q1: ***What is a diffusion model? What is the core idea?***

- **A diffusion model first adds noise to real data step by step until it becomes pure Gaussian noise, then trains a network to reverse this process, denoising step by step from noise to generate data.**
- The forward noising process is fixed and has no learnable parameters; the reverse denoising process is learned by a neural network.
- Core idea: break the hard problem of "generating data directly from noise" into many simple subproblems of "removing just a little noise".
- Representative work: DDPM, score-based SDEs, DDIM, Latent Diffusion (Stable Diffusion), DiT.

Q2: ***Write down the definition of the forward process and the closed form of $q(x_t|x_0)$.***

- **Each step slightly scales down the previous data and adds a bit of Gaussian noise; because Gaussians compose, $x_t$ at any time can be computed from $x_0$ in a single step.**
- The amount of noise is controlled by a preset noise schedule $\beta_t$; write $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s\le t}\alpha_s$.
- Significance: training does not need to simulate noising step by step; sample a random $t$ and construct $x_t$ directly, which makes training very efficient.
- The scaling factor $\sqrt{1-\beta_t}$ keeps the variance from blowing up (variance preserving); when $T$ is large enough, $x_T$ is approximately a standard Gaussian.
- Key formula: $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ \epsilon\sim\mathcal{N}(0,I)$

Q3: ***How is the reverse process modeled?***

- **The true reverse distribution $q(x_{t-1}|x_t)$ cannot be computed, but when each step's noise is small it is approximately Gaussian, so a network outputs the mean (and variance) of a Gaussian to approximate it.**
- Conditioned on $x_0$, the reverse posterior $q(x_{t-1}|x_t,x_0)$ is an analytically computable Gaussian; training makes the network fit it.
- DDPM fixes the variance (to $\beta_t$ or $\tilde\beta_t$) and only learns the mean; Improved DDPM also lets the model learn the variance.
- Key formula: $p_\theta(x_{t-1}|x_t)=\mathcal{N}\big(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\big)$

Q4: ***What is the DDPM training objective? Why predict the noise?***

- **The theoretical objective is maximizing the variational lower bound (ELBO) on the likelihood, which simplifies to a very simple regression task: given a noised image, the network predicts the noise that was added, trained with MSE.**
- Training loop: sample $x_0$, a timestep $t$ and noise $\epsilon$ → construct $x_t$ → the network predicts $\epsilon$ → compute MSE.
- Benefit of predicting noise: the target is always a standard Gaussian, so the scale is uniform and optimization is stable.
- The simplified loss drops the ELBO weighting coefficients, effectively down-weighting the easy low-noise steps, and gave better sample quality in experiments.
- The network is usually a U-Net (or Transformer) with a timestep embedding.
- Key formula: $L_{simple}=\mathbb{E}\big[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\big]$

Q5: ***How does DDPM sample? What are its drawbacks?***

- **Start from pure Gaussian noise $x_T$; at each step use the network's predicted noise to compute the denoised mean, then add a little random noise, and repeat for $T$ steps to get a sample.**
- No noise is added at the last step; the whole process resembles Langevin dynamics: take a step along the learned direction, then add a small perturbation.
- Drawback: every step needs one network forward pass, and the original DDPM uses $T=1000$ steps, so sampling is much slower than a GAN's single-step generation.

Q6: ***What noise schedules exist? Why is the cosine schedule better than the linear one?***

- **The common ones are the linear and cosine schedules; the cosine schedule destroys information more evenly instead of turning into pure noise too early like the linear one, so both sample quality and likelihood improve.**
- Linear schedule: $\beta_t$ increases linearly from $10^{-4}$ to $0.02$; on low-resolution images it destroys information too fast, so many late steps are nearly pure noise and contribute little to training.
- Cosine schedule (Improved DDPM): defines $\bar\alpha_t$ directly by a cosine curve, decreasing roughly linearly in the middle and flat at both ends.
- More generally a schedule can be described by its signal-to-noise ratio (SNR); high-resolution images have more pixel redundancy and need the schedule shifted toward higher noise.
- Note that a non-zero terminal SNR causes a mismatch between training and inference (which starts from pure noise); the schedule can be rescaled to make it zero (often paired with v-prediction).
- Key formula: $\text{SNR}(t)=\frac{\bar\alpha_t}{1-\bar\alpha_t}$

Q7: ***What is the relationship between diffusion models and score matching?***

- **Predicting noise and estimating the score (the gradient of the log density) are essentially the same thing, differing only by a scale factor; the DDPM loss is denoising score matching across multiple noise levels.**
- The score $\nabla_x\log p(x)$ points toward increasing data density; with it, one can sample using Langevin dynamics.
- For noised data, the conditional score is exactly $-\epsilon/\sqrt{1-\bar\alpha_t}$, so learning to predict noise means learning the score.
- NCSN (Noise Conditional Score Network) and DDPM are two formulations of the same idea.
- Key formula: $s_\theta(x_t,t)\approx-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}$

Q8: ***How can diffusion models be understood from the SDE perspective?***

- **As the number of steps goes to infinity, the noising process becomes a stochastic differential equation (SDE); it has a reverse SDE that depends only on the score function and is used for generation, plus a deterministic probability flow ODE with the same marginals.**
- DDPM corresponds to the VP-SDE (variance preserving) and NCSN to the VE-SDE (variance exploding); the SDE framework unifies them.
- Probability flow ODE sampling is deterministic and can be accelerated with high-order ODE solvers (the theoretical basis of DPM-Solver and others).
- The ODE gives an invertible mapping between noise and data (encoding/inversion) and allows exact likelihood computation.
- Key formula (forward SDE): $dx=f(x,t)\,dt+g(t)\,dw$

Q9: ***What does DDIM improve over DDPM?***

- **DDIM constructs a non-Markovian process with the same marginals as DDPM, so it can directly use a trained DDPM model, but it can skip steps when sampling and can be fully deterministic.**
- When sampling, it first estimates $\hat x_0$ from the predicted noise, then jumps directly to an earlier timestep; quality drops only a little with just tens of steps.
- With noise term $\sigma_t=0$ it is fully deterministic (a discretization of the probability flow ODE); with a particular $\sigma_t$ it reduces to DDPM.
- Determinism means the same initial noise maps to the same output, enabling semantic interpolation and DDIM inversion, commonly used for image editing.

Q10: ***What is classifier guidance?***

- **Train an extra classifier that can recognize classes on noised images; during sampling, add its gradient for the target class to the score, pushing samples toward "looking more like this class".**
- The principle is Bayes' rule: conditional score = unconditional score + gradient of the classifier's log probability.
- The larger the guidance strength $w$, the more class-consistent and higher-fidelity the samples, but the lower the diversity.
- Drawbacks: an extra classifier must be trained across all noise levels; its gradients can have an adversarial "fooling" effect; hard to extend to complex conditions such as text.
- Key formula: $\nabla\log p(x_t|y)=\nabla\log p(x_t)+\nabla\log p(y|x_t)$

Q11: ***What is classifier-free guidance (CFG)? How does it differ from classifier guidance?***

- **A single model learns both conditional and unconditional predictions (by randomly dropping the condition during training); at sampling time the difference between them is amplified, strengthening the condition without any extra classifier.**
- During training the condition is replaced by a null condition with about 10% probability (condition dropout); the difference between the two acts like the gradient of an "implicit classifier".
- Compared with classifier guidance: no extra classifier, simple to implement, works with arbitrary conditions such as text, usually better results; it is the standard in text-to-image.
- The cost is two forward passes per step (which can be merged into one batch); too large a scale causes oversaturation, color distortion and reduced diversity. A negative prompt can replace the null condition.
- Key formula: $\tilde\epsilon=\epsilon_\theta(x_t,\varnothing)+w\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)$

Q12: ***What is the architecture of latent diffusion (Latent Diffusion / Stable Diffusion)?***

- **A VAE first compresses images into a low-dimensional latent space; a U-Net performs diffusion denoising in that latent space with text injected via cross-attention; finally the VAE decoder maps back to an image.**
- VAE: about 8x spatial downsampling, responsible for perceptual compression and reconstruction.
- U-Net denoiser: with timestep embeddings, residual and attention blocks, iteratively denoising in latent space.
- Text encoder (e.g. CLIP): encodes the prompt into token embeddings; in cross-attention, image features are the Query and text embeddings are the Key/Value.
- Advantage: diffusing in latent rather than pixel space greatly reduces compute and memory, so high-resolution text-to-image runs on consumer GPUs.

Q13: ***What is DiT (Diffusion Transformer)? What are its advantages over U-Net?***

- **DiT replaces the U-Net with a Transformer as the denoising backbone: latents are cut into patches used as tokens, and conditions are injected via adaptive LayerNorm; its biggest advantage is that quality improves steadily with scale.**
- Timestep and class conditions are injected via adaLN; the best variant is adaLN-Zero (modulation parameters zero-initialized so each residual block starts as the identity).
- Good scalability: more compute (GFLOPs) gives lower FID, with a clear scaling trend.
- A uniform, simple architecture that reuses Transformer engineering, and handles multimodal conditions and variable length/resolution inputs easily.
- Later image and video models such as SD3 (MMDiT) and Sora use Transformer backbones.

Q14: ***How does ControlNet add extra conditional control to a diffusion model?***

- **Freeze the original U-Net, copy its encoder as a trainable branch that takes spatial conditions such as edge maps, depth maps or poses, and add its outputs back into the original network through zero-initialized convolutions.**
- Zero convolutions (1×1 convolutions with weights and biases initialized to 0) guarantee no effect on the original model at the start of training.
- So the pretrained generation ability is not destroyed, and training is stable even on small datasets.
- Similar lightweight control methods include T2I-Adapter and IP-Adapter (image prompts).

Q15: ***How is LoRA used to fine-tune diffusion models? How does it differ from DreamBooth and Textual Inversion?***

- **LoRA freezes the original weights and learns only a low-rank update on the attention layers; DreamBooth fine-tunes the whole model to bind a subject; Textual Inversion learns only a new word embedding. Their parameter count and expressiveness are, respectively: medium, large, small.**
- LoRA: few trainable parameters, small files, fast training, and multiple LoRAs can be combined with weights; commonly used to learn styles, characters or concepts.
- DreamBooth: binds a subject from a few images to a rare identifier, with a prior-preservation loss against overfitting and language drift; effective but expensive, often combined with LoRA.
- Textual Inversion: the model is fully frozen; fewest parameters but limited expressiveness.
- Key formula: $W'=W+BA,\ r\ll\min(d,k)$

Q16: ***What methods accelerate diffusion sampling?***

- **There are two main families: without retraining, use better ODE solvers to reduce the number of steps; or use distillation to train a model that generates in a few steps or even one.**
- Better solvers: DDIM step skipping; DPM-Solver/DPM-Solver++ exploit the semi-linear structure of the probability flow ODE and need only about 10-20 steps; also PNDM, UniPC, Heun/EDM.
- Distillation: progressive distillation (the student mimics two teacher steps in one, halving the steps each round); adversarial distillation (e.g. ADD in SDXL-Turbo); distribution matching distillation (DMD).
- Consistency models: directly learn a one-step mapping to data (see the next question).
- Others: latent diffusion lowers per-step cost, caching intermediate features, straighter trajectories (rectified flow).

Q17: ***What are Consistency Models?***

- **A consistency model learns a function that maps any point on a probability flow ODE trajectory directly to the trajectory's origin (clean data), so it can generate in one step.**
- Self-consistency constraint: points on the same trajectory give the same output; the boundary condition (output equals input at the smallest noise level) is guaranteed by a skip-connection parameterization.
- Training: consistency distillation (CD, using a pretrained diffusion model's ODE solver to get adjacent point pairs) or consistency training (CT, without a pretrained model).
- Multi-step sampling is also possible: generate, re-noise, and map again, trading compute for quality.
- Latent Consistency Models (LCM) apply this in the Stable Diffusion latent space for few-step generation.

Q18: ***What are Flow Matching and Rectified Flow?***

- **Flow matching directly learns a velocity field and uses an ODE to "flow" noise into data; Rectified Flow uses straight-line paths between noise and data, and the straighter the trajectory, the fewer steps are needed to generate.**
- The most common choice is the linear interpolation path, and the training target is for the network to predict the velocity of this line (noise minus data); the time direction convention may be reversed in different papers.
- Reflow: generate noise-data pairs with the current model and retrain, straightening trajectories further, even enabling one-step generation.
- The probability flow ODE of diffusion models can be seen as a special flow matching path; flow matching is simpler, needs no SDE theory, and allows more flexible path design.
- Stable Diffusion 3, FLUX and others use the rectified flow objective.
- Key formula: $x_t=(1-t)x_0+t\epsilon,\quad L=\mathbb{E}\big[\|v_\theta(x_t,t)-(\epsilon-x_0)\|^2\big]$

Q19: ***What is v-prediction? Why use it?***

- **v-prediction has the network predict a weighted combination of noise and clean data (the "velocity" $v$); it is numerically stable at both the high-noise and low-noise ends, avoiding the problems that predicting noise or predicting $x_0$ each have at one end.**
- Problem with predicting $\epsilon$: at high noise $x_t$ is almost all noise, so recovering $x_0$ from the predicted noise amplifies errors; with zero terminal SNR it cannot give a meaningful $x_0$ at all.
- Problem with predicting $x_0$: at low noise the model easily learns an identity mapping.
- The three parameterizations are interconvertible; v-prediction suits progressive distillation and zero-terminal-SNR schedules, and some SD 2.x models use it.
- Key formula: $v=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0$

Q20: ***How are diffusion models (especially text-to-image models) evaluated?***

- **Most commonly FID measures image quality and diversity, CLIP Score measures text-image alignment, complemented by human preference scores and human evaluation.**
- FID: the distance between generated and real distributions in Inception feature space; lower is better.
- CLIP Score: CLIP embedding similarity between image and prompt; higher is better. Raising CFG usually raises CLIP Score but worsens FID, so FID-CLIP curves are drawn for comparison.
- Others: IS, Precision/Recall, KID, FD with DINOv2 features; compositional benchmarks (GenEval, T2I-CompBench); preference models (ImageReward, PickScore, HPS).
- Human evaluation still matters, since automatic metrics are insensitive to aesthetics and details like hands and text; likelihood metrics (bits-per-dim) correlate poorly with visual quality.

Q21: ***What are the pros and cons of diffusion models versus GANs?***

- **Diffusion models train stably, have good diversity and are easy to scale and control, but sample slowly; GANs generate quickly in one step but train unstably, are prone to mode collapse and are hard to scale to open-domain text-to-image.**
- The diffusion objective is a simple MSE regression with no adversarial game; likelihood-based training rarely mode-collapses.
- Diffusion models easily incorporate conditions like text and support editing, inpainting and guidance, but lack a GAN's compact, semantic single-step latent space (DDIM inversion partly compensates).
- GANs reach very high quality in narrow domains like faces, and StyleGAN has strong latent-space editing.
- Trend: the two are merging, e.g. adversarial losses for diffusion distillation to achieve high-quality few-step generation.

Q22: ***What is the general approach of video diffusion models? What are the main challenges?***

- **The approach extends image diffusion to spatio-temporal data: a video VAE compresses clips into spatio-temporal latents, then a network with temporal attention or a spatio-temporal Transformer jointly denoises the whole clip; the biggest challenges are temporal consistency and compute cost.**
- Backbone: add temporal attention/convolution to an image model (factorized spatial + temporal), or cut the video into spatio-temporal patches for full spatio-temporal attention (DiT-style).
- Usually initialized from a pretrained image diffusion model to reuse image priors; conditions can be text, a first frame, motion or camera control.
- Challenges: keeping object identity and motion coherent and avoiding flicker; token count grows fast with duration and resolution; long videos need chunked or autoregressive extension.
- High-quality captioned video data is hard to obtain, and physical laws and causal plausibility remain hard to model.
