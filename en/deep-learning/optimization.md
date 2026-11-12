# Optimisation

## Easy

Q1: **Explain what Bayesian optimization is.**

- **Bayesian optimization is a black-box optimization method: it fits a surrogate model (usually a Gaussian process) to a few sampled points to estimate the distribution of the objective, then uses an acquisition function to pick the next most promising point, iterating and finally returning the best sampled point.**
- The surrogate model gives a predicted mean and an uncertainty at every point.
- The acquisition function (e.g. EI, UCB) trades off exploiting known good regions against exploring uncertain ones.
- It suits functions that are expensive to evaluate; it is commonly used for AutoML hyperparameter search, and some NAS methods use it too.

Q2: **Explain convex optimization in machine learning.**

- **If the optimization problem is convex, every local optimum is the global optimum, so simple methods such as gradient descent reliably find the best solution.**
- Convex problems are faster, simpler and cheaper to solve, and come with theoretical convergence guarantees.
- Linear regression, logistic regression and SVMs are convex; neural network training is non-convex.

![Untitled](optimization/Untitled.png)

Reference: [https://stats.stackexchange.com/](https://stats.stackexchange.com/)

Q3: Which hyperparameter tuning methods do you know?

- **The common ones are grid search, random search and Bayesian optimization.**
- **Grid search**: build a grid of candidate values for each hyperparameter, try every combination and return the best; simple but too expensive once there are many dimensions.
- **Random search**: sample combinations at random from the search space; for the same budget it is usually more efficient than grid search, because the important hyperparameters get to try more distinct values.
- **Bayesian optimization**: treat "hyperparameters -> validation loss" as an unknown function, build a surrogate model with methods such as SMBO, and choose the next configuration deliberately; it needs the fewest evaluations.

Q4: **Explain the Adam optimizer.**

- **Adam (Adaptive Moment Estimation) tracks both the first moment of the gradient (its mean, i.e. momentum) and the second moment (mean of squares), and uses them to adapt the step size of each parameter.**
- Think of it as Momentum + RMSProp: momentum stabilizes the direction, and the second moment shrinks the step for parameters with large gradients.
- It converges fast and is not very sensitive to the learning rate, so it is the default for many tasks (formula in Q16).

Q5: **What are the limitations of the Adam algorithm?**

- **Adam converges quickly, but the final model often generalizes worse than one trained with SGD + Momentum.**
- Despite its adaptive learning rate, it still needs a good learning-rate schedule to perform best.
- Early in training the weights are random and gradients unreliable, so a large learning rate can diverge; warmup is usually used before raising the learning rate.
- In the original Adam, L2 regularization is coupled with the adaptive step size and works poorly, so AdamW is generally used instead.

Q6: **How does *Batch Size* affect the convergence of gradient descent? Why?**

- **Large batches have low gradient noise and tend to converge to sharper minima close to the starting point; small batches have noisy gradients that push the parameters out of sharp basins, so they tend to reach flatter minima, which usually generalize better.**
- A flat minimum is insensitive to parameter perturbations, so it is more robust on test data.
- Large batches give more accurate steps and better parallelism, but usually need a correspondingly larger learning rate plus warmup.

![Untitled](optimization/Untitled%201.png)

As shown in the figure, both minima have the same loss, but the flat minimum is less sensitive to perturbations in parameter space.

Reference:

Q7: **In which situations would you choose a genetic algorithm as the optimization algorithm?**

- **A genetic algorithm fits when the objective is multimodal (many local optima), non-differentiable or a black box, and when many processors are available to evaluate in parallel.**
- It searches with a population and needs no gradients, so it is less likely to get stuck in a single local optimum.
- Each individual is evaluated independently, so it parallelizes naturally.

Reference: [https://stats.stackexchange.com/questions/249471/when-are-genetic-algorithms-a-good-choice-for-optimization](https://stats.stackexchange.com/questions/249471/when-are-genetic-algorithms-a-good-choice-for-optimization)

Q8: **How does the Adagrad algorithm adjust the learning rate?**

- **Adagrad accumulates the squared historical gradients of each parameter and divides the learning rate by the square root of that sum, so frequently updated, large-gradient parameters get small steps and rarely updated parameters get larger steps.**
- This makes it well suited to sparse data (e.g. sparse features, word embeddings).
- The downside is that the sum only grows, so the learning rate decays monotonically and training can stall later on; RMSProp fixes this with an exponential moving average.
- Key formula: $\theta_t=\theta_{t-1}-\dfrac{\eta}{\epsilon+\sqrt{r_t}}\odot g_t$, where $r_t=r_{t-1}+g_t\odot g_t$

Q9: **Compare Newton's method and gradient descent.**

- **Gradient descent uses only first derivatives and moves along the negative gradient; Newton's method uses second derivatives (the Hessian) to build a local quadratic approximation and jumps straight to its extremum, converging faster but at a much higher cost per step.**
- Convergence speed: near the optimum Newton's method converges quadratically, usually faster than gradient descent.
- Cost: Newton's method must compute and invert the Hessian, which is infeasible with many parameters, so quasi-Newton methods (e.g. L-BFGS) are used in practice.
- Non-convex problems: Newton's method is easily attracted to saddle points, while gradient descent escapes them more easily, so neural networks are generally trained with gradient descent.
- Gradient descent oscillates near the optimum and needs step-size adjustment, which led to adaptive methods such as Adagrad, RMSProp and Adam.

Q10: **How is Bayesian optimization used in hyperparameter optimization?**

- **Treat "hyperparameter configuration -> validation performance" as an expensive black-box function: a surrogate such as a Gaussian process predicts how untried configurations will perform from the ones already tried, an acquisition function picks the most promising next configuration to train and evaluate, the new result updates the model, and the loop repeats until the budget runs out.**
- Compared with grid/random search, it finds good hyperparameters with fewer training runs.
- Common tools include Hyperopt (TPE) and Optuna.

Q11: **Are pruning methods applicable to neural network optimization?**

- **Yes. Pruning scores the importance of neurons or weights and removes the unimportant ones, reducing computation and model size, and it can also lower the risk of overfitting.**
- You can add a sparsity regularizer during training so some weights go to 0, or prune by importance after training and then fine-tune.
- Unstructured pruning (individual weights) compresses more but needs sparse-compute support; structured pruning (whole channels/neurons) is easier to turn into real speedups.

![Untitled](optimization/Untitled%202.png)

Q12: **In a random forest model, how do you optimize the number of trees in the forest?**

- **Evaluate different numbers of trees with cross-validation (or the out-of-bag error) and choose the number at which performance levels off.**
- More trees make results more stable and generally do not cause overfitting, but cost grows linearly.
- In practice 10 to 1000 trees are typically evaluated, picking the point where gains stop being noticeable.

Q13: ****If a CNN is very large and runs inefficiently on a mobile phone, which model compression methods do you know?****

- **There are four main families: low-rank approximation, pruning and sparsity constraints, parameter quantization, and knowledge distillation.**
- **Low-rank approximation**: convolution is essentially matrix computation, so approximate the weight matrix with a few small matrices (e.g. SVD, Toeplitz matrix reconstruction) to cut computation and storage.
- **Pruning and sparsity constraints**: remove unimportant neurons and fine-tune, repeatedly; or add a sparsity regularizer to the training objective so some weights go to 0, finishing pruning in a single training run.
- **Parameter quantization**: represent weights with low bit-widths or a few representative values in a codebook, storing only indices in the original matrix, which greatly reduces storage.
- **Knowledge distillation**: a large trained teacher network guides a small student network, yielding a more compact model.

Q14: **Describe the principle and procedure of model distillation.**

- **Distillation trains a small student model to imitate the output distribution (soft labels) of a large teacher model; soft labels carry similarity information between classes, so they teach more than hard labels alone.**
- Procedure: train the teacher first; apply a softmax with temperature $T$ to both teacher and student logits to get softened probability distributions.
- The student's total loss = KL divergence to the soft labels + cross-entropy to the true hard labels, as a weighted sum.
- A higher temperature gives a smoother distribution and exposes more "dark knowledge" between classes; besides outputs, the student can also imitate intermediate feature maps.
- Key formula: $p_i=\dfrac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$

Reference: [https://blog.csdn.net/weixin_46838716/article/details](https://blog.csdn.net/weixin_46838716/article/details)

Q15: **Why can gradient descent reduce the loss? What is its mathematical principle?**

- **The gradient is the direction in which the function rises fastest at the current point, so a small step along the negative gradient always lowers the loss (as long as the gradient is nonzero and the step is small enough).**
- The principle is the first-order Taylor expansion: locally the function is roughly linear, and stepping along the negative gradient changes the loss by about minus "learning rate x squared gradient norm", which is negative.
- Among all directions with a fixed step length, the negative gradient decreases the loss the most, hence "steepest descent".
- If the learning rate is too large, the local linear approximation breaks down and training oscillates or diverges; if too small, convergence is very slow.
- For convex functions it converges to the global optimum; for non-convex functions such as neural networks it is only guaranteed to reach a stationary point (local minimum or saddle point).
- Key formula: $\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)$

Q16: **What are the principles of the Adam and SGD optimizers, and how do they differ?**

- **SGD updates parameters with the mini-batch gradient times one global learning rate, usually with momentum to smooth the direction; Adam adds, on top of momentum (first moment), a moving average of squared gradients (second moment) to scale each parameter's step individually, with bias correction.**
- Learning rate: SGD shares one learning rate across all parameters; Adam gives small steps to large-gradient parameters and large steps to sparse/small-gradient ones.
- Convergence and tuning: Adam converges fast and is insensitive to the learning rate, suiting NLP/Transformers; SGD converges slowly and is more sensitive to the learning rate and its schedule.
- Generalization: SGD + Momentum tends to reach flatter minima and usually generalizes better on CV tasks (e.g. ResNet on ImageNet) (see Q5).
- Memory and weight decay: Adam stores two extra states $m_t, v_t$ per parameter, about 3x the parameter count in memory; AdamW decouples weight decay from the adaptive step and is the common choice for training large models.
- Key formula (Adam): $\theta_{t+1}=\theta_t-\dfrac{\eta\,\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$, where $\hat m_t,\hat v_t$ are the bias-corrected first and second moments
