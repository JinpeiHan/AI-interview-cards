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

Q10: **Briefly describe DQN.**

Q11: **What exactly do the two tricks in DQN — the target network and experience replay — do?**

Q12: **What are the similarities and differences between DQN (Deep Q-learning) and Q-learning?**

Q13: **What variants of DQN are there?**

Q14: **Briefly describe the principle of Double DQN (double deep Q-network).**

Q15: **What advantages does the Dueling DQN (dueling deep Q-network) model have?**

# Advanced:

Q16: **What are the advantages of Actor - Critic?**

Q17: **Briefly describe the DDPG algorithm.**

Q18: **Is DDPG on-policy or off-policy, and why?**
