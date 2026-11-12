# Diffusion Models

Q1: ***What is a diffusion model? What is the core idea?***

- A diffusion model has two processes:
    - **Forward process (noising/diffusion process)**: Gaussian noise is gradually added to real data; after $T$ steps the data becomes approximately pure Gaussian noise. This process is **fixed** and has no learnable parameters.
    - **Reverse process (denoising process)**: a neural network is trained to start from pure noise and denoise step by step, finally producing samples that follow the data distribution.
- Core idea: break the hard problem of "generating data directly from noise" into **many simple small denoising steps**, where each step only needs to learn to remove a small amount of noise.
- Representative work: DDPM, score-based SDEs, DDIM, Latent Diffusion (Stable Diffusion), DiT, etc.

Q2: ***Write down the definition of the forward process and the closed form of $q(x_t|x_0)$.***

- A single noising step is defined as:

$$q(x_t|x_{t-1})=\mathcal{N}\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big)$$

- where $\beta_t\in(0,1)$ is a predefined **noise schedule**. Let $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$.
- Because Gaussians are closed under addition, $x_t$ at **any timestep can be obtained from $x_0$ in one step**:

$$q(x_t|x_0)=\mathcal{N}\big(x_t;\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)$$

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad\epsilon\sim\mathcal{N}(0,I)$$

- Significance: training does not need to simulate noising step by step; we can **randomly sample a $t$ and construct $x_t$ directly**, which makes training very efficient.
- When $T$ is large enough and $\bar\alpha_T\to0$, $x_T$ approximately follows $\mathcal{N}(0,I)$.
- The coefficient $\sqrt{1-\beta_t}$ keeps the variance from blowing up (**variance preserving**): if $x_0$ has unit variance, $x_t$ always has unit variance.

Q3: ***How is the reverse process modeled?***

- The true reverse conditional $q(x_{t-1}|x_t)$ is intractable (it depends on the whole data distribution), but when $\beta_t$ is small enough it is approximately Gaussian, so it is parameterized by a network:

$$p_\theta(x_{t-1}|x_t)=\mathcal{N}\big(x_{t-1};\mu_\theta(x_t,t),\ \Sigma_\theta(x_t,t)\big)$$

- **Conditioned on $x_0$**, the reverse posterior is an analytically tractable Gaussian:

$$q(x_{t-1}|x_t,x_0)=\mathcal{N}\big(x_{t-1};\tilde\mu_t(x_t,x_0),\ \tilde\beta_t I\big)$$

$$\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\Big),\quad \tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

- The training objective is to make $p_\theta(x_{t-1}|x_t)$ fit $q(x_{t-1}|x_t,x_0)$. In DDPM the variance $\Sigma_\theta$ is fixed to $\beta_t I$ or $\tilde\beta_t I$ and only the mean is learned; Improved DDPM also learns the variance.

Q4: ***What is the DDPM training objective? Why predict the noise?***

- The full objective maximizes the **variational lower bound (VLB/ELBO)** on the log-likelihood. It decomposes into a sum of KL terms, each a KL between two Gaussians, which is equivalent to a weighted MSE between means.
- The mean is parameterized via a noise prediction:

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Big)$$

- Dropping the per-term weighting coefficients gives the **simplified loss**:

$$L_{simple}=\mathbb{E}_{t,x_0,\epsilon}\Big[\big\|\epsilon-\epsilon_\theta\big(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\ t\big)\big\|^2\Big]$$

- Training loop: sample $x_0$ → sample $t$ uniformly → sample $\epsilon$ → construct $x_t$ → the network predicts $\epsilon$ → compute the MSE.
- Benefits of predicting noise: the target is always a standard Gaussian with a **uniform scale**, so optimization is stable; the simplified loss effectively down-weights small $t$ (little noise, easy task), which gave better sample quality in the DDPM experiments.
- The network $\epsilon_\theta$ is usually a **U-Net** (or Transformer) with timestep embeddings.

Q5: ***How does DDPM sample? What are its drawbacks?***

- Start from $x_T\sim\mathcal{N}(0,I)$ and iterate for $t=T,\dots,1$:

$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Big)+\sigma_t z,\quad z\sim\mathcal{N}(0,I)$$

- No noise is added at the last step. This process resembles **Langevin dynamics**: take a step along the learned gradient direction, then add a little random noise.
- Drawback: each step requires one network forward pass, and the original DDPM uses $T=1000$ steps, so **sampling is very slow**, far slower than single-step GAN generation.

Q6: ***What noise schedules exist? Why is the cosine schedule better than the linear one?***

- **Linear schedule**: $\beta_t$ increases linearly from a very small value to a larger one ($10^{-4}$ to $0.02$ in the DDPM paper).
- **Cosine schedule** (proposed in Improved DDPM) defines $\bar\alpha_t$ directly:

$$\bar\alpha_t=\frac{f(t)}{f(0)},\quad f(t)=\cos^2\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)$$

- Problem with the linear schedule: on low-resolution images, information is destroyed **too quickly**; the last stretch of timesteps is almost pure noise and contributes little to training.
- The cosine schedule makes $\bar\alpha_t$ decrease roughly linearly in the middle and change slowly at both ends, so **information is destroyed more evenly**, improving sample quality and likelihood.
- More generally, a schedule can be described by the **signal-to-noise ratio (SNR)** $\text{SNR}(t)=\frac{\bar\alpha_t}{1-\bar\alpha_t}$. High-resolution images need the schedule shifted toward **higher noise**, because neighboring pixels are highly redundant and information is destroyed more slowly at the same noise level.
- Also beware of **non-zero terminal SNR**: if $\bar\alpha_T$ is not exactly 0, there is a mismatch between training and inference (which starts from pure noise), causing issues such as generated images being biased toward medium brightness. This can be fixed by rescaling the schedule to zero terminal SNR (often combined with v-prediction).

Q7: ***What is the relationship between diffusion models and score matching?***

- The **score** is defined as the gradient of the log-density with respect to the data: $\nabla_x\log p(x)$; it points in the direction of increasing data density.
- Given the score function, one can sample with **Langevin dynamics**: $x\leftarrow x+\frac{\eta}{2}\nabla_x\log p(x)+\sqrt{\eta}\,z$.
- **Denoising score matching**: for noisy data $x_t\sim q(x_t|x_0)$, the conditional score is

$$\nabla_{x_t}\log q(x_t|x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

- Hence **predicting noise and estimating the score are essentially equivalent**:

$$s_\theta(x_t,t)\approx\nabla_{x_t}\log q_t(x_t)=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}$$

- DDPM's noise-prediction loss is a weighted denoising score matching over multiple noise levels; NCSN (Noise Conditional Score Network) and DDPM are two formulations of the same idea.

Q8: ***How can diffusion models be understood from the SDE perspective?***

- As the number of steps $T\to\infty$, the discrete noising process becomes a **stochastic differential equation (SDE)**:

$$dx=f(x,t)\,dt+g(t)\,dw$$

- DDPM corresponds to the **VP-SDE** (variance preserving), and NCSN corresponds to the **VE-SDE** (variance exploding).
- Every forward SDE has a corresponding **reverse-time SDE** that depends only on the score function:

$$dx=\big[f(x,t)-g(t)^2\nabla_x\log p_t(x)\big]dt+g(t)\,d\bar w$$

- There is also a **probability flow ODE** with exactly the same marginal distributions:

$$dx=\Big[f(x,t)-\frac{1}{2}g(t)^2\nabla_x\log p_t(x)\Big]dt$$

- Significance:
    - Unifies DDPM and score-based models.
    - ODE sampling is **deterministic**, so higher-order ODE solvers can be used to accelerate sampling (the theoretical basis of DPM-Solver and similar methods).
    - The ODE gives an **invertible mapping** between noise and data (encoding/inversion) and allows exact likelihood computation.

Q9: ***What does DDIM improve over DDPM?***

- DDIM (Denoising Diffusion Implicit Models) constructs a family of **non-Markovian** forward processes that share **the same marginals** $q(x_t|x_0)$ as DDPM, so **a model trained with DDPM can be used directly without retraining**.
- Sampling: first predict $\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$, then

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t,t)+\sigma_t z$$

- When $\sigma_t=0$, sampling is **fully deterministic** (equivalent to a discretization of the probability flow ODE); with a specific choice of $\sigma_t$ it reduces to DDPM.
- Advantages:
    - Sampling can be done on a **subsequence of timesteps** (e.g., only a few dozen steps), with a small quality drop even when the number of steps is greatly reduced.
    - Deterministic sampling means the same initial noise always gives the same output, so the latent space supports **semantic interpolation**, and **DDIM inversion** is possible for image editing.

Q10: ***What is classifier guidance?***

- The goal is to generate conditional samples $p(x|y)$. By Bayes' rule:

$$\nabla_{x_t}\log p(x_t|y)=\nabla_{x_t}\log p(x_t)+\nabla_{x_t}\log p(y|x_t)$$

- An extra classifier $p_\phi(y|x_t)$ is trained **on noisy images**, and its gradient is used to correct the score during sampling:

$$\tilde\epsilon=\epsilon_\theta(x_t,t)-w\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(y|x_t)$$

- The larger the guidance scale $w$, the more samples match the class and **the higher the quality (fidelity), but the lower the diversity**.
- Drawbacks: requires training an extra classifier that handles all noise levels; classifier gradients can have an adversarial "fooling" effect; hard to extend to complex conditions such as text.

Q11: ***What is classifier-free guidance (CFG)? How does it differ from classifier guidance?***

- **A single model** learns both conditional and unconditional predictions: during training, the condition $c$ is replaced with a null condition $\varnothing$ with some probability (e.g., around 10%) (**condition dropout**).
- At sampling time, the two predictions are combined by extrapolation:

$$\tilde\epsilon_\theta(x_t,c)=\epsilon_\theta(x_t,\varnothing)+w\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)$$

- $w=1$ is the plain conditional model, and $w>1$ strengthens the effect of the condition. The principle is that it implicitly uses $\nabla\log p(c|x_t)\propto\nabla\log p(x_t|c)-\nabla\log p(x_t)$ as an "implicit classifier".
- Differences from classifier guidance:
    - **No extra classifier needed**; simple to implement and works with arbitrary conditions such as text.
    - Each step needs **two forward passes** (conditional + unconditional, which can be merged into one batch), doubling inference cost.
    - Usually performs better and is the standard practice in current text-to-image models.
- Too large a guidance scale causes **oversaturation, color distortion, and reduced diversity**; a negative prompt can be used in place of the null condition.

Q12: ***What is the architecture of latent diffusion (Latent Diffusion / Stable Diffusion)?***

- Three main components:
    - **VAE (autoencoder)**: the encoder compresses images into a low-dimensional latent space (e.g., 8× spatial downsampling), and the decoder maps latents back to images.
    - **U-Net denoiser**: performs diffusion and denoising in the latent space, with timestep embeddings and residual/attention blocks.
    - **Text encoder**: e.g., the CLIP text encoder, which encodes the prompt into a sequence of token embeddings.
- **Text conditioning**: **cross-attention** layers are added to the U-Net, with image features as Query and text token embeddings as Key and Value:

$$\text{Attention}(Q,K,V)=\text{softmax}\Big(\frac{QK^\top}{\sqrt{d}}\Big)V,\quad Q=W_Q\varphi(z_t),\ K=W_K\tau(c),\ V=W_V\tau(c)$$

- Inference: random latent noise → U-Net iteratively denoises under text conditioning + CFG → VAE decodes to an image.
- Advantage: diffusion runs in latent space rather than pixel space, **greatly reducing compute and memory**, which makes high-resolution text-to-image generation feasible on consumer GPUs.

Q13: ***What is DiT (Diffusion Transformer)? What are its advantages over U-Net?***

- DiT replaces the U-Net with a **Transformer** as the denoising backbone of a diffusion model:
    - The latent is split into patches and linearly embedded into a token sequence (similar to ViT).
    - Conditions such as the timestep and class are injected via **adaLN (adaptive LayerNorm)**; the best-performing variant in DiT is **adaLN-Zero** (modulation parameters are zero-initialized so that each residual block starts as the identity).
    - At the output, tokens are rearranged back into a spatial layout to predict the noise (and variance).
- Advantages:
    - Good **scalability**: the more model compute (GFLOPs), the lower the FID, showing a clear scaling trend.
    - A unified, simple architecture that makes it easy to reuse engineering optimizations and training experience from the Transformer world.
    - Easy to handle multimodal conditions and variable-length/variable-resolution inputs.
- Many later image and video generation models (e.g., the MMDiT used in SD3, Sora, etc.) adopt Transformer backbones.

Q14: ***How does ControlNet add extra conditional control to a diffusion model?***

- Goal: add **spatial conditions** such as edge maps, depth maps, human poses, or segmentation maps without destroying the capabilities of the pretrained model.
- Method:
    - **Freeze** the parameters of the original U-Net.
    - **Copy** the encoder part of the U-Net as a trainable copy that takes the condition image as input.
    - The copy's outputs are added to the corresponding skip connections and middle block of the original U-Net through **zero convolutions** — 1×1 convolutions whose weights and biases are initialized to 0.
- Zero convolutions ensure that at the start of training ControlNet has **no effect at all** on the original model, so training does not damage the pretrained generative ability and remains stable even on small datasets.
- Similar lightweight control methods include T2I-Adapter and IP-Adapter (image prompts).

Q15: ***How is LoRA used to fine-tune diffusion models? How does it differ from DreamBooth and Textual Inversion?***

- **LoRA (Low-Rank Adaptation)**: freeze the original weights $W$ and learn only a low-rank update:

    $$W'=W+\Delta W=W+BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k)$$

    - In diffusion models it is usually applied to the **attention layers** of the U-Net/DiT (especially the projection matrices of cross-attention).
    - Few trainable parameters, small files, fast training; multiple LoRAs can be **combined** with weights to learn specific styles, characters, or concepts.
- **DreamBooth**: fine-tunes the **whole model** on a few images of a subject, binding the subject to a rare identifier, and uses a **prior-preservation loss** to prevent overfitting and language drift; effective but expensive (often combined with LoRA to reduce cost).
- **Textual Inversion**: the model is fully frozen and only a new **word embedding vector** is learned for the new concept; fewest parameters but limited expressiveness.

Q16: ***What methods accelerate diffusion sampling?***

- **Better solvers (no retraining)**:
    - DDIM: deterministic sampling with step skipping.
    - **DPM-Solver / DPM-Solver++**: exploit the **semi-linear structure** of the probability flow ODE, solving the linear part exactly and applying a high-order approximation only to the network-output part; high-quality samples in roughly ten to twenty steps.
    - Others such as PNDM, UniPC, and Heun/EDM samplers.
- **Distillation (requires training)**:
    - **Progressive distillation**: the student matches two teacher steps with one step, repeatedly, halving the number of steps each round.
    - **Adversarial distillation** (e.g., ADD used in SDXL-Turbo): adds an adversarial loss to enable one- to few-step generation.
    - Distribution Matching Distillation (DMD), etc.
- **Consistency models**: see the next question.
- **Others**: latent diffusion to reduce per-step cost, caching intermediate features, straighter trajectories (rectified flow), etc.

Q17: ***What are Consistency Models?***

- Core idea: learn a function $f_\theta(x_t,t)$ that maps **any point on the same trajectory** of the probability flow ODE directly to the trajectory's origin $x_0$.
- **Self-consistency** constraint: points on the same trajectory produce the same output, i.e., $f_\theta(x_t,t)=f_\theta(x_{t'},t')$; boundary condition $f_\theta(x_\epsilon,\epsilon)=x_\epsilon$ (guaranteed by a skip-connection parameterization).
- Training:
    - **Consistency distillation (CD)**: use an ODE solver with a pretrained diffusion model to obtain adjacent point pairs and enforce consistent outputs.
    - **Consistency training (CT)**: trained independently without a pretrained model.
- Sampling can be done in **a single step**, or in multiple steps: after generating, re-noise and map again, trading compute for quality.
- **Latent Consistency Models (LCM)** apply this to the latent space of Stable Diffusion, enabling generation in a few steps.

Q18: ***What are Flow Matching and Rectified Flow?***

- **Flow matching**: directly learn a **velocity field** $v_\theta(x_t,t)$ defining an ODE $\frac{dx_t}{dt}=v_\theta(x_t,t)$ that transports a simple distribution (noise) to the data distribution.
- The training objective is built from conditional paths. The most common is the **linear interpolation path** (rectified flow / conditional optimal transport path):

$$x_t=(1-t)\,x_0+t\,\epsilon,\quad \frac{dx_t}{dt}=\epsilon-x_0$$

$$L=\mathbb{E}_{t,x_0,\epsilon}\big[\|v_\theta(x_t,t)-(\epsilon-x_0)\|^2\big]$$

- Here $t=0$ is data and $t=1$ is noise (some papers use the opposite convention).
- **Rectified Flow**: uses straight paths and can further "straighten" trajectories via **reflow** (generating noise–data pairs with the current model and retraining). The straighter the trajectory, the smaller the integration error with fewer steps, possibly even enabling one-step generation.
- Relation to diffusion models: diffusion models (the probability flow ODE) can be viewed as a particular choice of flow-matching path; the flow-matching framework is simpler, needs no SDE theory, and allows more flexible path design.
- Models such as Stable Diffusion 3 and FLUX use the rectified flow objective.

Q19: ***What is v-prediction? Why use it?***

- Besides predicting the noise $\epsilon$ or directly predicting $x_0$, the model can predict the **velocity** $v$:

$$v=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0$$

- The three parameterizations are interconvertible, e.g., $\hat x_0=\sqrt{\bar\alpha_t}\,x_t-\sqrt{1-\bar\alpha_t}\,v$.
- Why use it:
    - **Problem of $\epsilon$-prediction at the high-noise end**: when $\bar\alpha_t\to0$, $x_t$ is almost all noise and predicting $\epsilon$ is easy, but recovering $\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}}$ amplifies errors; with zero terminal SNR, $\epsilon$-prediction cannot yield a meaningful $x_0$ at all.
    - **Problem of $x_0$-prediction at the low-noise end**: when noise is tiny the model easily learns an identity mapping, giving a weak learning signal.
    - v-prediction behaves well at both ends, is **more numerically stable**, and suits **progressive distillation** (where it was introduced) and zero-terminal-SNR schedules; some Stable Diffusion 2.x models use v-prediction.

Q20: ***How are diffusion models (especially text-to-image models) evaluated?***

- **FID**: measures the distance between the generated and real image distributions in Inception feature space, reflecting both **quality and diversity**; lower is better.
- **CLIP Score**: uses CLIP to compute the cosine similarity between embeddings of the generated image and the text prompt, measuring **text–image alignment**; higher is better.
- The two usually trade off: increasing the CFG scale generally raises CLIP Score but may worsen FID, so **FID–CLIP Score curves** are often plotted to compare models.
- Other metrics: IS, Precision/Recall (measuring quality and coverage separately), KID; FD with stronger features (e.g., DINOv2 features); benchmarks for compositionality/text understanding (e.g., GenEval, T2I-CompBench); human preference model scores (e.g., ImageReward, PickScore, HPS).
- **Human evaluation** is still important, since automatic metrics are not sensitive enough to aesthetics and detail correctness (hands, text).
- Likelihood metrics (e.g., NLL/bits-per-dim) can compare density estimation ability but correlate poorly with visual quality.

Q21: ***What are the pros and cons of diffusion models versus GANs?***

- **Diffusion model advantages**:
    - **Stable training**: the objective is simple regression (MSE) with no adversarial game.
    - **Good mode coverage and high diversity**: likelihood-based training is not prone to mode collapse.
    - **Good scalability and controllability**: easy to scale to large models and datasets, easy to add conditions such as text, and supports flexible operations like editing, inpainting, and guidance.
- **Diffusion model disadvantages**:
    - **Slow sampling**: requires many iterative steps, each with a network forward pass.
    - Lacks the naturally compact, semantic single-step latent space of GANs (DDIM inversion partly compensates).
- **GAN advantages**: one forward pass per sample, so **fast**; very high quality in narrow domains (e.g., faces); strong latent-space editing as in StyleGAN.
- **GAN disadvantages**: unstable training, mode collapse, hard to scale to open-domain text-to-image generation.
- Trend: the two are converging, e.g., using adversarial losses for diffusion distillation (adversarial distillation) to achieve few-step high-quality generation.

Q22: ***What is the general approach of video diffusion models? What are the main challenges?***

- Basic idea: extend image diffusion to **spatiotemporal data**, jointly denoising the latents of a whole video clip (frames × height × width × channels).
- Common practices:
    - Use a **video VAE** that compresses both spatially and temporally to obtain spatiotemporal latents.
    - The backbone adds **temporal attention/convolution** on top of an image model (e.g., factorized spatial attention + temporal attention), or uses a Transformer (DiT-style) that splits the video into **spacetime patches** with full spatiotemporal attention.
    - Often initialized from a pretrained image diffusion model and then trained on video data to reuse image priors.
    - Conditions can be text, a first-frame image (image-to-video), motion/camera control signals, etc.
- Main challenges:
    - **Temporal consistency**: keeping object identity, appearance, and motion coherent across frames and avoiding flicker.
    - **Compute cost**: the number of tokens grows rapidly with duration and resolution, making attention expensive.
    - **Long video generation**: requires strategies such as chunked generation, autoregressive extension, or hierarchical generation.
    - **Data**: high-quality captioned video data is hard to obtain and requires heavy filtering and automatic captioning.
    - Modeling physical laws and causal plausibility remains difficult.
