# JEPA

Q1: ***What is JEPA (Joint-Embedding Predictive Architecture)?***

- JEPA was proposed by Yann LeCun in the position paper "A Path Towards Autonomous Machine Intelligence" (2022) as the core module of his **world model** vision.
- Structure: given two related inputs $x$ (e.g., context, current state) and $y$ (e.g., an occluded part, a future state):
    1. Two **encoders** produce representations $s_x=\text{Enc}_x(x)$ and $s_y=\text{Enc}_y(y)$;
    2. A **predictor** predicts the representation of $y$ from $s_x$ (and an optional latent variable $z$): $\hat{s}_y=\text{Pred}(s_x,z)$;
    3. The training objective makes $\hat{s}_y$ close to $s_y$.
- Core idea: **make predictions in an abstract representation space rather than reconstructing in pixel/input space**, thereby ignoring unpredictable details and modeling only the predictable, semantic parts.

Q2: ***How does JEPA fundamentally differ from generative methods (e.g., MAE) in "what is predicted"?***

- **Generative/reconstruction methods** (MAE, autoregressive pixel generation): predict $y$ in **input space (pixels)**, with a loss such as $\|\hat{y}-y\|^2$.
    - Problem: much of the world's detail is inherently **unpredictable** (e.g., the exact texture of leaves, noise), so the model spends a lot of capacity modeling irrelevant details; when several futures are possible, pixel-level regression produces a blurry average.
- **JEPA**: predicts $s_y$ in **representation space**, with a loss such as $\|\hat{s}_y-s_y\|^2$.
    - The target encoder can learn to **discard unpredictable or irrelevant information**, and the predictor only needs to predict abstract, semantic content;
    - As a result, the learned features tend to be **more semantic**, and linear probing outperforms pixel-reconstruction methods.
- In short: MAE "completes the image", JEPA "completes the understanding of the image".

Q3: ***How does JEPA differ from contrastive learning (joint-embedding methods)?***

- **Joint-Embedding Architectures (JEA)**, such as SimCLR, BYOL and DINO: make the embeddings of different augmented views of the same image **identical** (invariance), usually **without an explicit conditional predictor** (BYOL's predictor is not conditioned on information such as position).
    - They rely on **hand-crafted data augmentations** to define "what should be invariant", which injects strong priors and may discard information useful for downstream tasks (e.g., color, position);
    - Contrastive methods also need **negatives**, and in high-dimensional spaces many negatives are required to be effective.
- **JEPA**:
    - Does not require the two embeddings to be the same; instead, a **predictor** **predicts** $s_y$ from $s_x$ with the help of conditioning information (e.g., the position of the target block, an action), so it can express the transformation between $x$ and $y$;
    - **Needs no negatives** and does not rely on hand-crafted augmentations (I-JEPA uses only masking);
    - The learning signal comes from "predicting missing parts" rather than "being invariant to augmentations".

Q4: ***How can JEPA be understood from the energy-based model (EBM) perspective?***

- An **energy-based model** uses a scalar energy function $F(x,y)$ to measure the **compatibility** of $x$ and $y$: compatible pairs have low energy and incompatible pairs have high energy; inference means finding the $y$ that minimizes the energy.
- JEPA's energy is defined as the prediction error in representation space (minimized over $z$ when a latent variable is present):

$$F(x,y)=\min_z D\big(s_y,\text{Pred}(s_x,z)\big)$$

- The **latent variable $z$** expresses the multiple possibilities in $y$ that cannot be determined from $x$ (e.g., multiple possible futures), allowing the model to handle multimodal uncertainty; its information capacity must be regularized to prevent $z$ from directly "leaking" the answer.
- The key to training an energy-based model is **preventing the energy surface from becoming flat** (i.e., giving low energy to every input — which corresponds to **representation collapse**):
    - **Contrastive methods**: explicitly push up the energy of negatives; they need many negatives and are inefficient in high-dimensional spaces;
    - **Regularized (non-contrastive) methods**: prevent collapse by limiting the "volume" of low-energy regions, e.g., VICReg-style variance/covariance regularization, or architectural means such as EMA and stop-gradient. LeCun advocates preferring regularized methods.

Q5: ***Describe the architecture and training procedure of I-JEPA***

- I-JEPA (Assran et al., 2023, "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture") instantiates JEPA for images and consists of three ViTs:
    - **Context encoder**: processes only the visible patches inside the image's **context block**;
    - **Target encoder**: processes the **full image**, and the representations at the positions of the **target blocks** are taken from its output as prediction targets; its parameters are an **EMA** of the context encoder and receive no gradients;
    - **Predictor**: a narrower ViT that takes the context encoder's output plus mask tokens carrying the target block's **position embeddings**, and outputs predictions of the target block representations.
- Loss: the **L2 distance** between predicted and target representations, averaged over all patches of all target blocks:

$$\mathcal{L}=\frac{1}{M}\sum_{i=1}^{M}\sum_{j\in B_i}\|\hat{s}_{y_j}-s_{y_j}\|_2^2$$

- Key details: masking is applied to the target encoder's **output** rather than its input, so the target representations have the context of the full image and are at a higher semantic level; the whole method **uses no hand-crafted data augmentations**.

Q6: ***What is I-JEPA's multi-block masking strategy? Why does it matter?***

- **Target blocks**: sample **several** (4 by default in the paper) rectangular blocks, each covering a **fairly large fraction** of the image, with aspect ratios randomly chosen within a range;
- **Context block**: sample one **large** block (covering most of the image), then **remove the parts overlapping with the target blocks** to avoid information leakage;
- Design principles:
    - Target blocks must be **large enough** that the prediction target is semantic (e.g., part of an object) rather than local texture that can be interpolated from neighboring pixels;
    - The context must be **sufficiently informative and spatially distributed** to provide enough information for the prediction;
    - The paper's ablations show that, compared with random patch masking or a single target block, multi-block masking is crucial for learning semantic representations.
- The predictor knows "where to predict" through position embeddings, so it can make conditional predictions for different locations.

Q7: ***How does JEPA avoid representation collapse? How does it relate to BYOL and data2vec?***

- Without negatives, the trivial solution is for the encoder to output a constant, making the prediction error 0. I-JEPA adopts an **asymmetric architecture** similar to BYOL:
    - **The target encoder is updated by EMA**: $\bar{\theta}\leftarrow m\bar{\theta}+(1-m)\theta$, with $m$ close to 1;
    - **Stop-gradient**: the loss does not backpropagate into the target encoder;
    - The **predictor** serves as an additional asymmetric component.
- **Relation to BYOL**: both use the asymmetric structure of EMA teacher + predictor + stop-gradient; the difference is that BYOL predicts the global embedding of another **augmented view**, while I-JEPA predicts patch-level embeddings of **other spatial regions** of the same image, conditioned on position.
- **Relation to data2vec**: data2vec also has a student predict, from masked input, the EMA teacher's representations of the full input (averaged over several layers) — essentially the same kind of masked prediction in representation space; I-JEPA emphasizes the multi-block masking strategy, a separate predictor, and a student encoder that processes only the context block.
- Limitation: EMA-based collapse prevention relies mainly on empirical designs exploiting optimization dynamics, with weak theoretical guarantees; LeCun's position paper also discusses explicit regularization schemes such as VICReg.

Q8: ***Why are the features learned by I-JEPA more semantic?***

- **Representation-space targets**: the target encoder's output has already filtered out some pixel details, so the prediction target itself is abstract;
- **Large target blocks**: predicting representations of large regions requires understanding object and scene structure and cannot be done by interpolating local texture;
- **No hand-crafted augmentations**: the model is not forced by augmentations to discard information such as color and position, nor is it dominated by low-level details as in pixel reconstruction;
- **The predictor absorbs some low-level work**: acting as a "buffer" between the encoder and the targets, it lets the context encoder focus on abstract representations (similar to the role of the decoder in MAE).
- The paper visualizes predicted representations by decoding them back to pixels with a separately trained generative model: the predictions correctly capture high-level information such as the position and pose of object parts, without precisely reproducing low-level details.

Q9: ***Why is I-JEPA said to be compute-efficient?***

- **Single view**: unlike contrastive learning, it does not need to generate multiple augmented views per image (e.g., DINO's multi-crop); each image requires only one forward pass of the target encoder;
- **The context encoder processes only part of the patches**: like MAE, this significantly reduces compute;
- **No pixel decoder**: prediction happens in representation space, so there is no need to reconstruct high-dimensional pixels;
- **Fast convergence**: the paper reports that it needs fewer pre-training GPU hours than methods such as MAE and iBOT to reach comparable performance, and scales well to large models such as ViT-H.
- Cost: an extra forward pass of the target encoder (but no backward pass).

Q10: ***How are JEPA models such as I-JEPA evaluated? What characterizes their results?***

- **Linear probing**: freeze the encoder and train a linear classifier (usually on average-pooled last-layer patch representations). I-JEPA's linear probing is clearly better than pixel-reconstruction methods such as MAE, and competitive with methods that use hand-crafted augmentations.
- **Low-shot classification**: e.g., using only 1% of ImageNet labels, I-JEPA performs well.
- **Low-level/dense tasks**: on tasks requiring local spatial information, such as object counting and depth prediction, I-JEPA outperforms methods that rely on augmentation invariance, such as DINO and iBOT, because it is not forced to discard information such as position.
- **V-JEPA** is mainly evaluated with a **frozen backbone + attentive probe**: a cross-attention layer with a learnable query aggregates tokens before classification, which better reflects the quality of patch-level features than a simple linear probe.
- General conclusion: JEPA excels in the **frozen-feature** setting, which is its main advantage over MAE.

Q11: ***How does V-JEPA extend JEPA to video?***

- V-JEPA (Bardes et al., 2024, "Revisiting Feature Prediction for Learning Visual Representations from Video") learns from video purely through a **feature prediction** objective, using no text supervision, negatives, pixel reconstruction or pretrained image encoders.
- Key designs:
    - The video is split into **spatio-temporal patches (tubelets)** as tokens;
    - **Spatio-temporal multi-block masking**: several spatial blocks are sampled and extended along the time dimension into "tubes", masking most of the spatio-temporal region and preventing information from being copied directly from adjacent frames;
    - Based on the visible spatio-temporal tokens and the positions of the mask tokens, the predictor predicts the representations of the masked regions under the EMA target encoder, using an L1 loss.
- The learned representations are effective in frozen evaluation for both **motion understanding** and appearance understanding, showing that representation-space prediction works for video as well and is more efficient than pixel reconstruction.

Q12: ***What progress does V-JEPA 2 make? How is it used for planning and robotics?***

- V-JEPA 2 (Meta, 2025) scales video JEPA pre-training to **larger-scale internet video and larger models**, learning more general visual understanding and prediction abilities.
- **Action-conditioned world model (V-JEPA 2-AC)**:
    - The pre-trained video encoder is frozen, and an **action-conditioned predictor** is trained on a **small amount of robot interaction data** to learn "given the current state representation, what the future representation will be after executing an action";
    - For planning, given the representation of a **goal image**, **model predictive control (MPC)** searches in representation space for the action sequence whose predicted future representation is closest to the goal representation; the first action is executed and then the plan is recomputed;
    - This allows robots to perform simple manipulation tasks such as grasping and placing zero-shot in unseen environments, without task-specific rewards or large numbers of demonstrations.
- The V-JEPA 2 encoder can also be aligned with a language model for tasks such as video question answering.
- Significance: this is a concrete validation of LeCun's world-model vision — predicting the future in **representation space rather than pixel space** and using it for planning.

Q13: ***What role does the predictor play in JEPA? Why not simply align the encoder outputs?***

- **Conditional prediction**: the predictor is conditioned on the target position (I-JEPA), spatio-temporal position (V-JEPA) or action (V-JEPA 2-AC), so the model can answer "what is the representation at this position / under this action", rather than simply requiring all views to have the same representation;
- **Breaking symmetry and preventing collapse**: as in BYOL/SimSiam, the predictor is part of the asymmetric structure and helps avoid trivial solutions;
- **Separating levels of abstraction**: the predictor takes on part of the "reasoning" work, letting the encoder focus on producing general representations; the paper found that predictor depth affects representation quality.
- Removing the predictor and directly aligning the encoder outputs would degrade it into an invariance-based joint-embedding method that cannot express spatial/temporal changes.

Q14: ***How does the latent variable $z$ in JEPA show up in practice?***

- In theory, $z$ represents the uncertainty in $y$ that cannot be inferred from $x$ (e.g., multiple possible contents of an occluded region), and is determined at inference by minimizing energy or by sampling.
- In the implementations of I-JEPA and V-JEPA, the predictor is usually **deterministic**, conditioned only on the target position, with no explicit latent variable; uncertainty is partly handled by "predicting the predictable parts and ignoring unpredictable details" in representation space.
- How to efficiently introduce and regularize latent variables to achieve multimodal predictions (multiple possible futures) remains an open problem.

Q15: ***How does JEPA relate to LeCun's hierarchical world model (H-JEPA) vision?***

- The position paper proposes a **Hierarchical JEPA (H-JEPA)**:
    - Lower-level JEPAs predict more detailed representations over short time scales;
    - Higher-level JEPAs make predictions over **longer time scales** on more abstract representations;
    - Predictions at different levels together support **hierarchical planning**.
- In this vision, the world model forms an autonomous agent architecture together with a perception module, a cost module, an actor, short-term memory and a configurator.
- Current I-JEPA and V-JEPA mainly implement single-level representation learning and prediction; a complete hierarchical world model has not yet been realized.

Q16: ***What are the limitations and open questions of JEPA?***

- **Collapse and training stability**: relies on empirical designs such as EMA and stop-gradient, lacks rigorous theoretical guarantees, and is sensitive to hyperparameters such as the EMA coefficient and masking strategy.
- **Hand-designed masking**: block size, number and aspect ratio are still chosen by hand, which is itself another kind of inductive bias.
- **Evaluation**: frozen-feature evaluation is strong, but on some tasks that need global invariance it may still lag behind methods using strong augmentations; its advantage in fine-tuning is less pronounced than in linear probing.
- **No direct generation**: representation-space prediction cannot output pixels directly, making it hard to interpret and visualize (an extra decoder must be trained).
- **Uncertainty modeling**: a deterministic predictor struggles to express multiple possible futures, and practical schemes for latent variables are immature.
- **Limited planning ability**: world-model-based planning has so far been validated mainly on short horizons and relatively simple tasks; long-horizon, hierarchical planning remains to be explored.
- **Integration with language**: how to effectively combine JEPA-style world models with large language models is also an active research direction.
