# Q-learning

# Beginner:

Q1: **How do you define the state in reinforcement learning?**

- **The state describes the current situation of the environment and should contain the information needed to make a decision; defining it is essentially the same problem as feature engineering in supervised learning.**
- Simple problems can discretize the state, but in high dimensions this hits the curse of dimensionality.
- Complex problems usually use state representation learning (SRL), learning the state as a feature vector (e.g. extracted from images by a neural network).

Q2: **What do the *Alpha* and *Gamma* parameters represent in Q Learning?**

- **Alpha is the learning rate, deciding how much new experience overwrites the old Q value; Gamma is the discount factor, deciding how much the agent values future rewards.**
- Alpha is in 0–1: larger means faster adoption of new estimates but less stability, and it is usually decreased as learning goes on.
- Gamma is in 0–1: near 0 the agent only cares about immediate reward and is short-sighted; near 1 distant rewards matter as much as immediate ones, but returns can blow up in non-terminating tasks and learning gets harder.

Q3: **How do you tell when the Q-Learning algorithm has converged?**

- **In practice, check whether the learning curve (returns or Q-value changes) has flattened; in theory, tabular Q-learning is guaranteed to converge to the optimum only under two conditions.**
- The learning rate must go to 0, but not too fast: the sum of learning rates diverges while the sum of their squares converges, e.g. 1, 1/2, 1/3, ….
- Every state-action pair must be visited infinitely often, i.e. the policy gives every action non-zero probability, ensured in practice by ε-greedy with ε > 0.
- With too little exploration, a flat curve may just mean premature convergence to a suboptimal policy.

Reference: [https://stats.stackexchange.com/](https://stats.stackexchange.com/)

Q4: **How do you understand the epsilon greedy algorithm?**

- **ε-greedy picks a random action with probability ε (exploration) and the action with the highest current Q value with probability 1−ε (exploitation), balancing exploration and exploitation.**
- At the start the Q table is nearly empty, so following it blindly leads to mistakes or going in circles; the agent should wander randomly to see more states.
- As learning progresses, a decay function lowers ε so the agent increasingly follows the learned policy.

Reference: [https://blog.csdn.net/Gin07](https://blog.csdn.net/Gin07)

Q5: **Explain the Q-learning algorithm**

- **Q-learning learns an action-value function Q(s,a): the expected discounted return of taking action a in state s and acting optimally afterwards; once it is known, picking the action with the highest Q in each state is the optimal policy.**
- It is based on the Bellman optimality equation: the optimal value now equals the immediate reward plus the discounted optimal value of the next state.
- At every step it uses "immediate reward + discounted max Q of the next state" as the target and moves the current Q value a little toward it.
- It is model-free and off-policy: the target uses a max, independent of the action actually taken.
- Key formula: $Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]$

Q6: **Write the Bellman equation based on the state value function and the Bellman equation based on the state-action value function.**

- **Both equations say: current value = expected immediate reward + discounted next-step value; the difference is that V takes the expectation over the current action under the policy, while Q already fixes the current action and only takes the expectation over the next action.**
- State value function: $V^{\pi}(s)=\mathbb{E}_{a\sim\pi}\mathbb{E}_{s'}\left[r+\gamma V^{\pi}(s')\right]$
- State-action value function: $Q^{\pi}(s,a)=\mathbb{E}_{s'}\left[r+\gamma\,\mathbb{E}_{a'\sim\pi}Q^{\pi}(s',a')\right]$

Q7: **Talk about SARSA, ideally writing out its Q(s,a) update formula. Also, is it on-policy or off-policy, and why?**

- **SARSA is very similar to Q-learning, but its update target uses the Q value of the action a′ that will actually be taken next rather than the max Q value, so it is on-policy.**
- Each update needs a five-tuple (s, a, r, s′, a′), which is where the name comes from.
- a′ is chosen by the current policy (e.g. ε-greedy), so it estimates the Q values of that same policy; the policy generating data and the one being evaluated are the same.
- As a result SARSA is more "conservative" and accounts for the risk introduced by exploration.
- Key formula: $Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma Q(s',a')-Q(s,a)\right]$

Q8: **What is the difference between value-based and policy-based methods?**

- **Value-based methods learn a value function first and derive the policy indirectly (e.g. take the action with the highest Q); policy-based methods parameterize the policy directly and optimize the return with policy gradients or similar.**
- Policy form: value methods usually give a deterministic policy; policy methods can learn stochastic policies.
- Action space: value methods need a max over actions, suiting discrete actions, and discretizing continuous actions makes the granularity hard to choose; policy methods can output continuous actions directly (e.g. a Gaussian).
- Convergence: value methods may diverge with function approximation plus bootstrapping; policy gradients converge more reliably but have high variance.
- Aside: policy iteration has two loops, "policy evaluation + policy improvement", while value iteration iterates the optimal value function directly with no separate improvement step.

Reference: *Easy RL*

# Intermediate:

Q9: **Can you briefly describe the similarities and differences between dynamic programming, Monte Carlo and temporal difference?**

- **All three update a value estimate toward some target; they differ in where the target comes from: DP takes an expectation over all successor states using a known model, MC uses the actual return of a whole episode, and TD uses one-step reward plus the estimated value of the next state.**
- DP: needs a model of the environment, bootstraps, does not sample.
- MC: model-free, no bootstrapping, must wait for the episode to end; unbiased but high variance.
- TD: model-free, samples and bootstraps, can update online every step; biased but low variance, usually converges faster.
- n-step TD and TD(λ) sit between MC and one-step TD to trade off bias and variance.

Q10: **Briefly describe DQN.**

- **DQN approximates the Q function with a deep neural network: the input is the state (e.g. 4 stacked Atari frames), the output is a Q value for each discrete action, and it is trained on the mean squared TD error, reaching human level on many Atari games.**
- Two key tricks: experience replay (sample random mini-batches from a buffer) and a target network (copy the main network's parameters every few steps).
- It explores with ε-greedy, decaying ε; in practice reward clipping, the Huber loss and frame skipping are also used.
- Key formula: $L(\theta)=\mathbb{E}\left[\left(r+\gamma\max_{a'}Q(s',a';\theta^-)-Q(s,a;\theta)\right)^2\right]$

Q11: **What exactly do the two tricks in DQN — the target network and experience replay — do?**

- **The target network keeps the regression target fixed for a while, avoiding the oscillation and divergence of "chasing a moving target"; experience replay breaks temporal correlation between samples and reuses data, making training more stable and sample-efficient.**
- Target network: a hard copy every C steps, or a small soft update each step, turning RL into a series of approximately supervised regression problems.
- Experience replay: random sampling makes data closer to i.i.d., matching SGD's assumptions, and smooths distribution shifts caused by policy changes.
- Experience replay requires an off-policy algorithm (buffer data comes from old policies), which Q-learning is.
- The improved Prioritized Experience Replay (PER) samples by TD-error size and corrects the bias with importance-sampling weights.

Q12: **What are the similarities and differences between DQN (Deep Q-learning) and Q-learning?**

- **Both are value-based, model-free, off-policy TD methods using the same "reward + discounted max Q" target; the difference is that Q-learning stores Q values in a table while DQN approximates the Q function with a neural network.**
- A table handles only small discrete state spaces; a network handles high-dimensional states like images and generalizes across similar states.
- Q-learning updates one table entry at a time; DQN does gradient descent on the TD loss, updating all parameters.
- Function approximation + bootstrapping + off-policy is the "deadly triad" and tends to diverge, so DQN adds experience replay and a target network; the tabular version has convergence guarantees, DQN does not.
- Both need a max over actions, so both apply only to discrete actions.

Q13: **What variants of DQN are there?**

- **The main variants target overestimation, architecture, sampling, exploration and return modeling, and Rainbow combines six of them.**
- Double DQN: decouples action selection from evaluation to reduce overestimation; Dueling DQN: splits Q into state value and advantage.
- Prioritized replay: samples by TD error; n-step DQN: uses n-step returns to propagate rewards faster.
- Noisy DQN: learnable noise in the weights replaces ε-greedy; distributional DQN (C51, QR-DQN, IQN): learns the full return distribution.
- Others: DRQN (adds an LSTM for partial observability), Ape-X / R2D2 (large-scale distributed training).

Q14: **Briefly describe the principle of Double DQN (double deep Q-network).**

- **DQN uses the same noisy estimates both to select and to evaluate the action, so taking the max systematically overestimates Q values; Double DQN decouples them: the online network selects the action and the target network evaluates its value.**
- Overestimation happens because the expected max of noisy estimates exceeds the max of the true values, and it accumulates through bootstrapping.
- The idea comes from Double Q-learning (two Q tables evaluating each other); Double DQN simply reuses the existing target network, adding almost no cost.
- Key formula: $y=r+\gamma\,Q\left(s',\arg\max_{a'}Q(s',a';\theta);\theta^-\right)$

Q15: **What advantages does the Dueling DQN (dueling deep Q-network) model have?**

- **Dueling DQN splits the back end of the network into two streams, one estimating the state value V(s) and one estimating each action's advantage A(s,a), then combines them into Q; this lets it learn efficiently how good a state is even when the action barely matters.**
- For example, in a racing game with no car ahead, every action is about the same, so the network can learn V directly rather than learning each action separately.
- Every update updates V, so all actions share that information, improving sample efficiency, especially with many actions of similar value.
- To make V and A identifiable, the mean advantage is subtracted when combining them.
- It only changes the architecture, so it combines directly with Double DQN, PER and others.

# Advanced:

Q16: **What are the advantages of Actor - Critic?**

- **Actor-Critic uses an Actor (policy network) to choose actions and a Critic (value network) to judge them and guide the Actor's updates, combining the strengths of policy gradients and value methods.**
- Versus REINFORCE: the Critic's advantage/TD-error estimate replaces the Monte Carlo return, greatly reducing variance, and updates can happen every step, so learning is faster.
- Versus DQN: it optimizes a parameterized policy directly, handles continuous actions, learns stochastic policies, and the policy changes smoothly with its parameters.
- The cost is a biased Critic and two coupled networks, making training sensitive to hyperparameters; common algorithms are A2C/A3C, PPO, DDPG, TD3, SAC.
- Key formula: $\nabla_\theta J\approx\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a|s)\,A(s,a)\right]$

Q17: **Briefly describe the DDPG algorithm.**

- **DDPG is an Actor-Critic algorithm for continuous actions, roughly "DQN + deterministic policy gradient": the Actor outputs one deterministic action, the Critic estimates that action's Q value, and the Actor is updated in the direction that increases Q.**
- The Critic minimizes the TD error like DQN, with targets computed by a target Actor and a target Critic.
- It borrows experience replay and target networks from DQN, with soft target updates (τ about 0.005).
- Exploration adds noise to the deterministic action (the paper used OU noise; Gaussian noise also works).
- DDPG also suffers from Q overestimation and hyperparameter sensitivity; TD3 fixes this with twin Critics taking the minimum, delayed Actor updates and target policy smoothing.

Q18: **Is DDPG on-policy or off-policy, and why?**

- **DDPG is off-policy: data is collected with a noisy behavior policy, while the policy being learned and evaluated is the noise-free deterministic target policy, and the two differ.**
- The next action in the Critic's target comes from the target policy, not from the action actually taken in the data, so the update does not depend on who generated the data (the same reason as Q-learning's max).
- The deterministic policy gradient needs no integral over actions and hence no importance sampling, so it can train directly on old-policy data from the replay buffer, which is the hallmark of off-policy learning.
