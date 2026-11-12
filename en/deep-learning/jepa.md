# JEPA

Q1: ***What is JEPA (Joint-Embedding Predictive Architecture)?***

- **JEPA is the core module of LeCun's world-model proposal: two encoders map related inputs x (e.g. context) and y (e.g. a masked part or the future) into representations, and a predictor predicts y's representation from x's.**
- The predictor can also take a latent variable z to express what in y cannot be determined from x.
- Core idea: predict in an abstract representation space rather than reconstruct in pixel space, so unpredictable details are ignored and only the semantic, predictable parts are modeled.

Q2: ***How does JEPA fundamentally differ from generative methods (e.g., MAE) in "what is predicted"?***

- **Generative methods like MAE predict y in pixel space, while JEPA predicts y's representation in representation space.**
- The problem with pixels: many details (leaf textures, noise) are inherently unpredictable and waste model capacity; with several possible futures, pixel regression produces a blurry average.
- In representation space the target encoder can learn to drop irrelevant information and the predictor only predicts semantic content, so features are more semantic and linear probing is better.
- In one line: MAE "completes the image", JEPA "completes the understanding of the image".

Q3: ***How does JEPA differ from contrastive learning (joint-embedding methods)?***

- **Joint-embedding methods like SimCLR, BYOL and DINO require embeddings of different augmented views of the same image to be equal; JEPA instead uses a predictor, conditioned on position or action, to predict y's representation from x's.**
- Joint-embedding methods rely on hand-crafted augmentations to define "what should be invariant", a strong prior that can discard useful information such as color or position; contrastive methods also need many negatives.
- JEPA needs no negatives and no hand-crafted augmentation (I-JEPA uses only masking), and it can express the transformation between x and y.
- The learning signal comes from "predicting the missing part", not "staying invariant to augmentations".

Q4: ***How can JEPA be understood from the energy-based model (EBM) perspective?***

- **An energy-based model uses an energy function to score whether x and y are compatible (low energy if they are); JEPA's energy is the prediction error in representation space, minimized over z when there is a latent variable.**
- The latent z expresses multiple possibilities for y (e.g. different futures), but its information content must be limited so it does not leak the answer.
- The key to training is preventing the energy surface from going flat, i.e. representation collapse.
- Contrastive methods explicitly push up the energy of negatives, which is inefficient in high dimensions; regularized methods limit the volume of low-energy regions (VICReg-style variance/covariance terms, EMA plus stop-gradient), and LeCun favors the latter.
- Key formula: $F(x,y)=\min_z D\big(s_y,\text{Pred}(s_x,z)\big)$

Q5: ***Describe the architecture and training procedure of I-JEPA***

- **I-JEPA uses three ViTs: a context encoder sees only the patches in the context block; a target encoder sees the full image and the target-block representations are taken from its output as targets; a predictor predicts the target representations from the context representation and the target-block positions, with an L2 loss between them.**
- The target encoder is an EMA of the context encoder and receives no gradients.
- The predictor is a narrower ViT whose inputs include mask tokens carrying the target position embeddings.
- Masking is applied to the target encoder's output, not its input, so target representations carry full-image context and are more semantic.
- The method uses no hand-crafted data augmentation at all.

Q6: ***What is I-JEPA's multi-block masking strategy? Why does it matter?***

- **It samples several (4 by default) fairly large target blocks, then one context block covering most of the image with any overlap with the targets removed; this makes the prediction targets semantic rather than textures that can be interpolated from neighboring pixels.**
- Target blocks must be large enough to force understanding of object parts.
- The context must be rich and spatially spread out enough to support prediction; removing the overlap prevents information leakage.
- Ablations show multi-block masking is crucial for semantic representations compared with random patch masking or a single target block.

Q7: ***How does JEPA avoid representation collapse? How does it relate to BYOL and data2vec?***

- **Without negatives, a constant encoder output gives zero error; like BYOL, I-JEPA prevents collapse with an asymmetric design: an EMA-updated target encoder that receives no gradients (stop-gradient), plus a predictor.**
- Difference from BYOL: BYOL predicts the global embedding of another augmented view, while I-JEPA predicts patch-level embeddings of other regions of the same image, conditioned on position.
- Link to data2vec: both have a student on masked input predict an EMA teacher's representation of the full input; I-JEPA emphasizes multi-block masking, a separate predictor and encoding only the context block.
- Limitation: collapse prevention via EMA is mostly empirical, with weak theoretical guarantees.

Q8: ***Why are the features learned by I-JEPA more semantic?***

- **Because it predicts abstract representations rather than pixels, and the target blocks are large, so prediction requires understanding objects and scene structure rather than interpolating local textures.**
- With no hand-crafted augmentation, it is not forced to discard color or position, and it is not dominated by low-level details either.
- The predictor absorbs some low-level work, letting the encoder focus on abstract representations (like MAE's decoder).
- Visualizing with a separately trained decoder shows predictions capture the position and pose of object parts but not fine detail.

Q9: ***Why is I-JEPA said to be compute-efficient?***

- **Because it needs only one view per image, the context encoder processes only some patches, and there is no pixel decoder, so it needs less GPU time than MAE or iBOT to reach similar results.**
- It does not generate multiple augmented views like DINO's multi-crop.
- Prediction happens in representation space, so no high-dimensional pixels are reconstructed.
- The cost is an extra forward pass of the target encoder (no backward pass).

Q10: ***How are JEPA models such as I-JEPA evaluated? What characterizes their results?***

- **They are mainly evaluated with frozen features, which is exactly JEPA's strength: linear probing is clearly better than MAE and competitive with methods that rely on hand-crafted augmentation.**
- Low-label classification (e.g. 1% of ImageNet labels) is also strong.
- On tasks needing spatial information, like object counting and depth prediction, it beats DINO and iBOT because position information is not thrown away.
- V-JEPA is evaluated with a frozen backbone plus an attentive probe: a cross-attention layer with a learnable query aggregates the tokens before classification, which reflects patch-level feature quality better than a linear probe.

Q11: ***How does V-JEPA extend JEPA to video?***

- **V-JEPA learns from video using only a feature-prediction objective: the video is split into spatio-temporal patches, most of the space-time volume is masked, and the predictor predicts the masked regions' representations under the EMA target encoder.**
- It uses no text supervision, negatives, pixel reconstruction or pretrained image encoder.
- Masks are several spatial blocks extended through time into tubes, preventing copying from neighboring frames; the loss is L1.
- The learned representations work for both motion and appearance understanding under frozen evaluation, and are more efficient than pixel reconstruction.

Q12: ***What progress does V-JEPA 2 make? How is it used for planning and robotics?***

- **V-JEPA 2 scales video JEPA pre-training to more video and larger models, then trains an action-conditioned predictor on top of the frozen encoder with a small amount of robot data, using it as a world model for planning.**
- The action-conditioned predictor learns "what the future representation will be after taking this action in the current state".
- For planning, given a goal image's representation, model predictive control (MPC) searches in representation space for the action sequence whose predicted future is closest to the goal, executes the first step, then replans.
- This lets robots do simple grasp and place tasks zero-shot in new environments without task rewards or many demonstrations.
- Its encoder can also be aligned with a language model for video QA; the significance is validating the world-model idea of "predict the future in representation space and plan with it".

Q13: ***What role does the predictor play in JEPA? Why not simply align the encoder outputs?***

- **The predictor lets the model predict conditioned on position or action, answering "what is the representation at this position / after this action", instead of simply requiring all views to match.**
- It is part of the asymmetric design and, as in BYOL/SimSiam, helps prevent collapse.
- It takes on part of the "reasoning", letting the encoder focus on general representations; predictor depth affects representation quality.
- Removing it and aligning outputs directly reduces JEPA to an invariance-based joint-embedding method that cannot express spatial or temporal change.

Q14: ***How does the latent variable $z$ in JEPA show up in practice?***

- **In theory z captures the uncertainty in y that cannot be inferred from x, but in actual I-JEPA and V-JEPA implementations the predictor is deterministic, conditioned only on target positions, with no explicit latent variable.**
- Uncertainty is handled partly by "predicting only the predictable parts and ignoring unpredictable details".
- How to introduce and regularize latents efficiently to predict multiple possible futures remains open.

Q15: ***How does JEPA relate to LeCun's hierarchical world model (H-JEPA) vision?***

- **H-JEPA stacks JEPAs into levels: lower levels predict more detailed representations over short time scales, higher levels predict more abstract representations over longer time scales, together supporting hierarchical planning.**
- In LeCun's vision the world model works alongside perception, cost, actor, short-term memory and configurator modules to form an autonomous agent.
- Current I-JEPA and V-JEPA implement only single-level representation learning and prediction; a full hierarchical world model does not yet exist.

Q16: ***What are the limitations and open questions of JEPA?***

- **The main issues are that collapse prevention is empirical, masking strategies are hand-designed, pixels cannot be generated directly, and uncertainty modeling and long-horizon planning are still immature.**
- Training stability: relies on EMA and stop-gradient and is sensitive to the EMA coefficient, masking and other hyperparameters.
- Evaluation: frozen features are strong, but it may trail strong-augmentation methods on tasks needing global invariance, and its edge under fine-tuning is smaller than under linear probing.
- Interpretability: representations cannot output pixels directly; visualization needs a separately trained decoder.
- Applications: planning has been shown only on short-horizon, simple tasks, and how to combine it with large language models is an active direction.
