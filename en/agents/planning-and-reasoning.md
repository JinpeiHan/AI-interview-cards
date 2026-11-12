# Planning and Reasoning

Q1: ***Why do agents need planning?***

- **Complex tasks have multiple interdependent steps; an agent without planning tends to improvise step by step, leading to omissions, repetition or drift from the goal.**
- Planning breaks the goal into executable subtasks, fixes order and dependencies, anticipates needed tools and information, and provides checkpoints.
- Good planning must also adapt to execution feedback.

Q2: ***What are common task decomposition methods?***

- **Common methods are prompted decomposition, Plan-and-Execute, hierarchical decomposition, DAG decomposition and external planners; in practice the most common is a to-do list updated during execution.**
- Prompted: have the model list steps before executing, or Least-to-Most (solve simple subproblems first, then harder ones).
- Hierarchical: high-level goal → subgoals → atomic actions, suited for long tasks.
- DAG: subtasks form a directed acyclic graph, and independent ones run in parallel.
- External planner: translate the problem into a formal language like PDDL for a classical planner, suited for well-constrained domains.

Q3: ***What is chain-of-thought (CoT)? Why does it work?***

- **CoT has the model write intermediate reasoning steps before the answer, which breaks a hard problem into simple steps and spends more compute at inference.**
- Two forms: few-shot CoT (examples show the reasoning) and zero-shot CoT (add "let's think step by step").
- Why it works: intermediate results become context for later reasoning, so each token carries less computation.
- Limits: the reasoning may not be faithful to the real decision; intermediate errors propagate; it wastes tokens on simple tasks.

Q4: ***What is self-consistency?***

- **Sample several reasoning paths for the same question at a higher temperature, then majority-vote on the final answers.**
- Rationale: the correct answer is reachable via many paths, while wrong answers are scattered.
- Pros: simple, no training, clear gains on tasks with definite answers like math and commonsense reasoning.
- Cons: cost grows linearly with the number of samples; open-ended generation can't be voted on directly, so use an LLM to pick the most consistent answer (Universal Self-Consistency).

Q5: ***What is tree-of-thoughts (ToT)? How does it differ from CoT?***

- **CoT is a single linear reasoning path; ToT organizes reasoning as a tree: generate several candidate thoughts per step, have the model evaluate them, then use search to expand promising branches and backtrack when needed.**
- Search can be BFS, DFS or beam search.
- It suits problems needing exploration and backtracking, such as Game of 24, puzzles and writing plans.
- Downsides: many calls and high cost, limited benefit for everyday tasks; extensions include Graph-of-Thoughts and LLM + MCTS.

Q6: ***What are reasoning models? How do they differ from ordinary LLMs?***

- **Reasoning models are specially trained to produce a long internal thinking process before answering (e.g. the o-series, DeepSeek-R1, and the thinking modes of Claude and Gemini), and are clearly stronger at math, coding and scientific reasoning.**
- They are trained mainly with reinforcement learning on verifiable rewards, learning to self-verify, backtrack and try other approaches.
- The cost is higher latency and price, and they may overthink simple tasks.
- Many APIs let you set a thinking budget or reasoning effort to trade quality against cost.

Q7: ***What is test-time compute scaling?***

- **Spending more compute at inference to improve results, instead of relying only on bigger models or more training data.**
- Sequential scaling: longer chains of thought, reflection and revision.
- Parallel scaling: sample many answers and pick one, e.g. Best-of-N, majority vote, verifier or reward-model scoring, tree search.
- On many reasoning tasks the gains rival scaling the model, but they saturate and depend on problem difficulty; an agent's multi-step tool use plus reflection is itself test-time compute.

Q8: ***What are process reward models (PRMs) and outcome reward models (ORMs)?***

- **An ORM scores only the final answer; a PRM scores each reasoning step, catching wrong steps earlier and guiding search (e.g. pruning low-scoring branches).**
- PRMs need step-level labels, which are expensive.
- For verifiable tasks like math and code, rule-based verifiers can serve directly as the reward.

Q9: ***In agents, what does "interleaved thinking" mean?***

- **The model doesn't just think once at the start; it can reason again after every tool result before choosing the next action.**
- Benefit: it can correct the plan from new observations, check whether results match expectations, and decide whether more information is needed.
- It is essentially ReAct's thought-action-observation loop, supported natively by the model.

Q10: ***What are common failure modes in agent planning?***

- **The main ones are unrealistic plans, rigid plans, over-planning, wrong decomposition granularity, missed constraints, goal drift, premature completion, and long-horizon error accumulation.**
- Unrealistic: assumes tools, permissions or information that don't exist; rigid: doesn't replan after failures or environment changes.
- Wrong granularity: too coarse to execute, too fine to coordinate; missed constraints: forgetting budget, format and other requirements.
- Long tasks drift, declare completion without verification, and the more steps, the bigger the effect of early errors.

Q11: ***How can these planning failures be mitigated?***

- **Maintain an explicit plan updated after each step, verify each subtask when done, and replan on failure instead of blindly retrying.**
- Restate the original goal and constraints periodically during long tasks.
- Add human review for critical plans.
- Give a clear definition of done.

Q12: ***Is CoT reasoning faithful? What does that mean for agents?***

- **Not necessarily: the model may be influenced by hints in the prompt without admitting it, or reach an answer first and rationalize afterwards.**
- So CoT can't be treated as a fully reliable explanation or safety monitor.
- But CoT monitoring is still a useful signal for spotting anomalies like reward hacking or rule-breaking intent.
- In practice, combine it with behavioral evaluation of what actions were actually taken.

Q13: ***How should prompts for reasoning models be written? How do they differ from ordinary models?***

- **Reasoning models think on their own, so give clear goals, constraints and success criteria and let the model plan, rather than prescribing reasoning steps.**
- Prompts like "think step by step" are usually unnecessary.
- Few-shot examples help little and can even constrain the model.
- Control cost with the thinking budget, and turn thinking off or down for simple tasks.

Q14: ***What is the "overthinking" problem?***

- **Reasoning models generate lots of thinking tokens even on simple questions, adding needless latency and cost, and sometimes second-guess a correct answer into a wrong one.**
- Mitigations: route by difficulty (non-reasoning mode or a small model for easy tasks), set a thinking budget, use adaptive thinking.

Q15: ***How do you decide whether a task should use a reasoning model or an ordinary model?***

- **Use a reasoning model for multi-step math and logic, complex coding and debugging, long-task planning and reviewing high-stakes decisions; use a fast ordinary model for simple QA, classification, extraction, format conversion and latency-sensitive interaction.**
- Common architecture: the reasoning model plans and makes key decisions, fast models execute many simple subtasks.
- Ultimately decide by evaluating on your own tasks, not by intuition.

Q16: ***What is Reflexion? How is it similar to reinforcement learning?***

- **Reflexion has the agent reflect in natural language on why an attempt failed, store the reflection in memory, and put it into context on the next attempt.**
- Like RL it uses success/failure signals to improve the policy, but it doesn't update weights; the lessons are stored as text ("verbal reinforcement learning").
- It requires a reliable success/failure signal (e.g. test results) and multiple allowed attempts.
