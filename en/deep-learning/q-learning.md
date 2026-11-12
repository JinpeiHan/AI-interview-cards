# Q-learning

# Beginner:

Q1: **How do you define the state in reinforcement learning?**

- The state-representation problem in reinforcement learning (RL) is similar to feature representation, feature selection and feature engineering in supervised or unsupervised learning. A common way to model complex problems is discretization, but discretization can bring the curse of dimensionality. For such problems, state representation learning (SRL) is usually used to represent the state as a vector of different features.

Q2: **What do the *Alpha* and *Gamma* parameters represent in Q Learning?**

- Alpha is the learning rate; it should decrease as we keep acquiring a larger and larger knowledge base. The learning rate should be in the range `0-1`. The higher the learning rate, the faster it replaces the new Q-values, so we need to tune it somehow so that our agent learns from previous Q-values. The learning rate is a tool that can be used to determine how much of the previously acquired experience we keep for a state-action pair.
- Gamma is the value of future rewards. It can have a large impact on learning and can be dynamic or static. If it equals `1`, the agent values future rewards as much as current rewards. This means that, in ten actions, if the agent does something good, it is as valuable as doing this action directly. So learning does not work well with high gamma values. Conversely, a gamma of `0` makes the agent value only immediate rewards, which only works with very detailed reward functions.

Q3: **How do you tell when the Q-Learning algorithm has converged?**

- A reinforcement learning algorithm is considered converged when the learning curve becomes flat and no longer increases. However, because the exploration parameter ε is not gradually increased, Q-Learning can converge prematurely before reaching the optimal policy.
- Therefore, two conditions must be satisfied to guarantee convergence in the limit:
    - **The learning rate approaches zero, but not too quickly.** Formally, this requires that the sum of the learning rates diverges, while the sum of their squares converges. An example sequence with these properties is 1/1, 1/2, 1/3, 1/4, ....
    - **Every state-action pair must be visited infinitely often.** This has a precise mathematical definition: every action must have a non-zero probability of being selected by the policy in every state, i.e. π(s, a) > 0 for all (s, a). In practice, using an ε-greedy policy (where ε > 0) ensures this condition is met.

Reference: [https://stats.stackexchange.com/](https://stats.stackexchange.com/)

Q4: **How do you understand the epsilon greedy algorithm?**

- (1) At the start our cart encounters very few states, and if it acts according to the Q-table learned so far, it will very likely make mistakes or go around in circles. At the same time, we want the cart to wander randomly at the beginning so it encounters more states.
- (2) For these reasons, we want the cart at the beginning not to act completely according to the Q learning results, i.e. with some probability epsilon it chooses an action randomly instead of choosing the action by maxQ. Then, as learning continues, we lower this random probability, using a decay function to reduce epsilon.
- (3) This solves the so-called exploration and exploitation problem, finding a trade-off between "exploring" and "exploiting".

Reference: [https://blog.csdn.net/Gin07](https://blog.csdn.net/Gin07)

Q5: **Explain the Q-learning algorithm**

- The Q(s, a) function, i.e. the quality function, represents the discounted future reward when the agent takes action a in state s and then takes optimal actions afterwards
(ignoring for now how future actions are chosen):

$$
Q(s_t,a_t)=max(R_{t+1})
$$

Assuming we have this Q function, we can compute the maximum return of each decision at the current time t; by comparing these returns, we can find which decision at time t has the highest return among them. From the recursive formula of the Q function we get:

$$
Q(s_t,a_t)=r+max_aQ(s_{t+1},a_{t+1})
$$

This is the famous Bellman equation. The Bellman equation says that, for a given state, maximizing the future reward is equivalent to maximizing the sum of the immediate reward and the maximum future reward of the next state. The core idea of Q-learning is: we can iteratively approximate the Q-function via the Bellman equation.

Q6: **Write the Bellman equation based on the state value function and the Bellman equation based on the state-action value function.**

- Bellman equation based on the state value function:

$$
{{V}^{\pi }}(s)={{\mathbb{E}}_{a \sim \pi (a|s)}}{{\mathbb{E}}_{s'\sim p(s'|s,a)}}[r(s,a,s')+\gamma {{V}^{\pi }}(s')]
$$

- Bellman equation based on the state-action value function (Q function):

$$
{{Q}^{\pi }}(s,a)={{\mathbb{E}}_{s'\sim p(s'|s,a)}}\left[ r(s,a,s')+\gamma {{\mathbb{E}}_{a'\sim \pi (a'|s')}}[{{Q}^{\pi }}(s',a')] \right]
$$

Q7: **Talk about SARSA, ideally writing out its Q(s,a) update formula. Also, is it on-policy or off-policy, and why?**

- SARSA can be regarded as an improvement of Q-learning; its update formula is:

$$
Q(s, a) \larr Q(s, a) + \alpha [r(s,a) + \gamma Q(s', a') - Q(s, a)]
$$

It is on-policy: SARSA must execute two actions to obtain (s,a,r,s′,a′) before it can make one update; moreover, a′ is the action executed under the guidance of a specific policy π,
so the estimated Q(s,a) is the Q-value under this policy π. The π used to generate samples and the π being estimated
are the same, hence it is on-policy.

Q8: **What is the difference between value-based and policy-based methods?**

1. Difference in the generated policy: the former is deterministic, the latter stochastic. In value-based methods, the estimates of action values eventually converge (usually to different numbers, which can be converted into probabilities between 0 and 1), so a deterministic policy is usually obtained; policy-based methods do not converge to a fixed value, and they tend to produce the optimal stochastic policy. If the optimal policy is deterministic, then the value function of the optimal action will be much larger than that of suboptimal actions, and the magnitude of the value function represents the magnitude of the probability.
2. Whether the action space is continuous: the former is discrete, the latter continuous. For value-based methods on continuous action spaces, although the action space can be discretized, the discretization interval is hard to choose. Too large an interval prevents the algorithm from reaching the optimal action, so it hovers around it; too small an interval increases the action dimensionality, causing the curse of dimensionality just like a high-dimensional action space and hurting the algorithm's speed. Policy-based methods, by contrast, suit continuous action spaces: in a continuous action space there is no need to compute the probability of each action; instead actions are selected via a normal distribution.
3. Value-based methods, e.g. the Q-learning algorithm, solve for the optimal policy indirectly by solving for the optimal value function; policy-based methods, e.g. REINFORCE, parameterize the policy directly and update the parameters through policy search, policy gradient or evolutionary methods to maximize the return. Value-based methods do not extend easily to continuous action spaces, and they have convergence problems when nonlinear approximation, bootstrapping, etc. are used together. Policy gradient has good convergence properties.
4. In addition, regarding value iteration and policy iteration: policy iteration has two loops — one in policy evaluation, where many iterations are needed to obtain the value function of the current policy; the other is the outer loop of policy evaluation and policy improvement. Value iteration does it in one step, directly estimating the optimal value function, so it has no policy-improvement step.

Reference: *Easy RL*

# Intermediate:

Q9: **Can you briefly describe the similarities and differences between dynamic programming, Monte Carlo and temporal difference?**

- **Similarities**: all three estimate value functions (policy evaluation) and then improve the policy; at their core, all of them move the value estimate toward some "target": $V(s) \leftarrow V(s) + \alpha[\text{target} - V(s)]$.
- **Dynamic programming (DP)**: requires a **known environment model** $p(s',r|s,a)$; it uses the Bellman expectation equation for **full-width backups**, with target $\sum_a \pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V(s')]$; it uses **bootstrapping** and needs no sampling.
- **Monte Carlo (MC)**: **model-free**, samples complete episodes and uses the actual return $G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}$ as the target; it **does not bootstrap**, must wait until the episode ends to update, and only applies to episodic tasks; the estimate is **unbiased but high-variance**.
- **Temporal difference (TD)**: **model-free**, **sampling + bootstrapping**, updates with the one-step TD target $r_{t+1}+\gamma V(s_{t+1})$, with TD error $\delta_t = r_{t+1}+\gamma V(s_{t+1})-V(s_t)$; it can **update online after every step** and works for continuing tasks; the estimate is **biased but low-variance**, and usually converges faster.
- Summary: needs a model or not (DP yes, MC/TD no); bootstraps or not (DP/TD yes, MC no); samples or not (MC/TD yes, DP no). **n-step TD and TD(λ)** sit between MC and one-step TD, trading off bias against variance.

Q10: **Briefly describe DQN.**

- **DQN (Deep Q-Network)** was proposed by DeepMind (NIPS workshop 2013, Nature 2015). It approximates the Q function with a deep neural network $Q(s,a;\theta)$ that takes the state as input (e.g. a stack of 4 Atari frames) and outputs a Q-value for each discrete action, reaching human-level performance on many Atari games.
- The loss is the mean squared TD error:

$$
L(\theta)=\mathbb{E}_{(s,a,r,s')\sim D}\left[\left(r+\gamma \max_{a'}Q(s',a';\theta^-)-Q(s,a;\theta)\right)^2\right]
$$

- Two key tricks: **experience replay**, sampling random mini-batches from a replay buffer $D$; and a **target network**, whose parameters $\theta^-$ are copied from $\theta$ every $C$ steps.
- Actions are chosen with **ε-greedy**, with ε decayed over training; in practice reward clipping, Huber loss, frame skipping, etc. are also commonly used.

Q11: **What exactly do the two tricks in DQN — the target network and experience replay — do?**

- **Target network**: if the same network computes both the prediction $Q(s,a;\theta)$ and the target $r+\gamma\max_{a'}Q(s',a';\theta)$, every update of $\theta$ also moves the target ("chasing a moving target"), so training easily oscillates or even diverges. The target network $\theta^-$ is held fixed for a while (hard copy every $C$ steps, or soft update $\theta^- \leftarrow \tau\theta+(1-\tau)\theta^-$), keeping the regression target stable for a period and approximately turning the RL problem into a sequence of supervised regression problems.
- **Experience replay**: transitions $(s,a,r,s')$ from interaction are stored in a buffer and sampled uniformly at random for training. Effects: (1) **breaks the temporal correlation between samples**, making the data closer to i.i.d., as SGD assumes; (2) **improves sample efficiency**, since each experience can be reused many times; (3) smooths the data distribution, avoiding sharp distribution shifts and forgetting caused by policy changes.
- Follow-up: experience replay requires an **off-policy** algorithm (the buffer contains data from old policies), which Q-learning is; an improved version is **Prioritized Experience Replay (PER)**, which samples by TD-error magnitude and corrects the bias with importance-sampling weights.

Q12: **What are the similarities and differences between DQN (Deep Q-learning) and Q-learning?**

- **Similarities**: both are value-based, model-free, **off-policy** TD methods; both update toward the Bellman optimality target $r+\gamma\max_{a'}Q(s',a')$; both commonly use ε-greedy exploration.
- **Differences**:
    - **Representation**: Q-learning uses a **Q-table** and can only handle small discrete state spaces; DQN uses a **neural network as a function approximator**, handling high-dimensional/continuous states (e.g. images) and generalizing across similar states.
    - **Update**: Q-learning incrementally updates a single table entry $Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]$; DQN performs gradient descent on the TD loss, updating all parameters.
    - **Stability**: function approximation + bootstrapping + off-policy form the "deadly triad" and can easily diverge, so DQN adds **experience replay** and a **target network**; tabular Q-learning has convergence guarantees under the right conditions, while DQN has no theoretical convergence guarantee.
- Both only apply to **discrete action spaces** (they need a max over actions).

Q13: **What variants of DQN are there?**

- **Double DQN**: decouples action selection from evaluation to mitigate Q-value overestimation.
- **Dueling DQN**: decomposes Q into a state value $V(s)$ and an advantage function $A(s,a)$.
- **Prioritized Experience Replay**: samples by priority based on TD error $|\delta|$ and corrects the bias with importance-sampling weights.
- **Multi-step learning (n-step DQN)**: uses the n-step return as the target, speeding up reward propagation and trading off bias against variance.
- **Noisy DQN (NoisyNet)**: adds learnable parameterized noise to the network weights for exploration instead of ε-greedy.
- **Distributional DQN (C51 / QR-DQN / IQN)**: learns the full distribution of returns rather than only its expectation.
- **Rainbow** (2017): combines the six improvements above (Double, Dueling, PER, n-step, Noisy, C51).
- Others: **DRQN** (adds an LSTM for partially observable problems), **Ape-X / R2D2** (large-scale distributed training), etc.

Q14: **Briefly describe the principle of Double DQN (double deep Q-network).**

- **Problem**: the DQN target $r+\gamma\max_{a'}Q(s',a';\theta^-)$ uses the same estimates both to **select** and to **evaluate** the action. Because the estimates are noisy, $\mathbb{E}[\max_a \hat Q] \ge \max_a \mathbb{E}[\hat Q]$, so taking the max systematically **overestimates** Q-values, and the error keeps accumulating through bootstrapping.
- **Approach**: decouple selection and evaluation — use the **online network** $\theta$ to select the action and the **target network** $\theta^-$ to evaluate its value:

$$
y = r + \gamma\, Q\left(s', \arg\max_{a'} Q(s',a';\theta);\ \theta^-\right)
$$

- The idea comes from Double Q-learning (keeping two Q-tables that evaluate each other); Double DQN simply reuses the existing target network, adding almost no computation while significantly reducing overestimation and improving stability and final performance.

Q15: **What advantages does the Dueling DQN (dueling deep Q-network) model have?**

- **Architecture**: after the shared feature layers the network splits into two streams, one outputting the **state value** $V(s)$ and the other the **advantage function** $A(s,a)$, which are then combined into Q-values. To fix the non-identifiability of $V$ and $A$ (a constant can be shifted between them), the mean advantage is subtracted:

$$
Q(s,a)=V(s)+\left(A(s,a)-\frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')\right)
$$

- **Advantages**:
    - In many states the choice of action hardly matters (e.g. in an Atari racing game when there is no car ahead); the dueling architecture can **learn the state value directly** without learning it separately for every action.
    - Every update updates $V(s)$, so this information is effectively shared across all actions, giving **better sample efficiency**, especially when there are many actions with similar values.
    - It only changes the network architecture, so it is **orthogonal to and directly combinable with** Double DQN, PER and other improvements.

# Advanced:

Q16: **What are the advantages of Actor - Critic?**

- **Structure**: the Actor (policy network $\pi_\theta(a|s)$) selects actions, and the Critic (value network $V_w(s)$ or $Q_w(s,a)$) evaluates how good they are and guides the Actor's update:

$$
\nabla_\theta J(\theta)\approx \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s)\, A(s,a)\right],\quad A(s,a)\approx r+\gamma V_w(s')-V_w(s)
$$

- **Compared with pure policy gradient (REINFORCE)**: using the Critic's estimate (TD error / advantage) instead of the Monte Carlo return **greatly reduces variance**; it can **update every step** without waiting for the episode to end, learning faster and more sample-efficiently.
- **Compared with pure value methods (DQN)**: it directly optimizes a parameterized policy, naturally handling **continuous action spaces** and learning **stochastic policies**; the policy changes smoothly with the parameters, giving better convergence.
- Cost: the Critic's estimate is biased, the two networks are coupled, and training is sensitive to hyperparameters. Common algorithms: A2C/A3C, PPO, DDPG, TD3, SAC.

Q17: **Briefly describe the DDPG algorithm.**

- **DDPG (Deep Deterministic Policy Gradient, 2015)** is an Actor-Critic algorithm for **continuous action spaces**, which can be seen as "DQN + deterministic policy gradient (DPG)".
- The **Actor** $\mu(s;\theta^\mu)$ directly outputs a deterministic action; the **Critic** $Q(s,a;\theta^Q)$ evaluates that action.
- **Critic update**: minimize the TD error, with targets computed by the target networks:

$$
y = r + \gamma\, Q'\left(s', \mu'(s';\theta^{\mu'});\ \theta^{Q'}\right),\quad L=\frac{1}{N}\sum_i\left(y_i - Q(s_i,a_i;\theta^Q)\right)^2
$$

- **Actor update**: move in the direction that increases the Q-value (deterministic policy gradient):

$$
\nabla_{\theta^\mu}J \approx \frac{1}{N}\sum_i \nabla_a Q(s,a;\theta^Q)\big|_{s=s_i,a=\mu(s_i)}\ \nabla_{\theta^\mu}\mu(s;\theta^\mu)\big|_{s_i}
$$

- Tricks borrowed from DQN: **experience replay** and **target networks** (with **soft updates** $\theta' \leftarrow \tau\theta+(1-\tau)\theta'$, $\tau$ typically around 0.005).
- **Exploration**: noise is added to the deterministic action, $a=\mu(s)+\mathcal{N}$ (the original paper used Ornstein-Uhlenbeck noise; in practice Gaussian noise works just as well).
- Follow-up: DDPG also suffers from Q-value overestimation and hyperparameter sensitivity; **TD3** improves it with twin Critics taking the minimum, delayed Actor updates and target policy smoothing.

Q18: **Is DDPG on-policy or off-policy, and why?**

- **DDPG is off-policy.**
- The **behavior policy** is the noisy policy $\mu(s)+\mathcal{N}$, used to interact with the environment and collect data; the **target policy** is the deterministic policy $\mu(s)$, the one being learned and evaluated — the two are different.
- In the Critic's target $r+\gamma Q'(s',\mu'(s'))$, the next action is given by the target policy rather than the action actually taken by the behavior policy, so the update does not depend on which policy generated the data (the same reason as the $\max_{a'}$ in Q-learning).
- Since the deterministic policy gradient involves no integral over actions, **no importance sampling is needed**, so it can train directly on data generated by old policies stored in **experience replay** — the hallmark of off-policy learning.
