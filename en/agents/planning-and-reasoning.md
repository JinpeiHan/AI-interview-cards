# Planning and Reasoning

Q1: ***Why do agents need planning?***

- Complex tasks often require multiple steps with dependencies between them; an agent without planning tends to **take things one step at a time**, leading to omissions, repetition or drifting from the goal.
- What planning does: breaks the goal into executable subtasks, determines order and dependencies, estimates the tools and information needed, and provides checkpoints for execution.
- Good planning must also support **dynamic adjustment based on execution feedback**.

Q2: ***What are common task decomposition methods?***

- **Prompt-based decomposition**: simply ask the model to "list the steps first, then execute", or use Least-to-Most prompting (solve simple subproblems first, then progressively harder ones);
- **Plan-and-Execute**: generate an explicit plan, then execute step by step and replan;
- **Hierarchical decomposition**: high-level goal → subgoals → atomic actions, suited to long tasks;
- **DAG-based decomposition**: organize subtasks as a directed acyclic graph, so subtasks without dependencies can **run in parallel**;
- **External planners**: translate the problem into a formal language (e.g. PDDL) and solve it with a classical planner, suitable for domains with well-defined constraints.
- In practice, agents often maintain a **todo list** as an explicit plan and update it as they go.

Q3: ***What is chain-of-thought (CoT)? Why does it work?***

- CoT has the model **generate intermediate reasoning steps** before giving the final answer.
- Two forms:
    - **Few-shot CoT**: show worked solutions with reasoning in the examples;
    - **Zero-shot CoT**: add a sentence such as "Let's think step by step".
- Why it works (intuitive explanation):
    - It breaks a complex problem into several simple steps, lowering the "amount of computation" each token must carry;
    - The generated intermediate results become context for later reasoning, which is equivalent to **adding computation at inference time**;
- Limitations: the reasoning may be **unfaithful** to the model's actual decision process; errors in intermediate steps propagate to the final answer; it adds unnecessary tokens for simple tasks.

Q4: ***What is self-consistency?***

- **Sample multiple reasoning paths at a higher temperature** for the same question, then take a **majority vote** over the final answers and choose the most frequent one.
- Rationale: the correct answer can often be reached via many different reasoning paths, while wrong answers are more scattered.
- Pros: simple, no training needed, and noticeably improves tasks with well-defined answers such as math and commonsense reasoning;
- Cons: cost grows linearly with the number of samples $N$; not applicable to open-ended generation (answers cannot be voted on directly) — instead an LLM can be used to pick the most consistent answer (Universal Self-Consistency).

Q5: ***What is tree-of-thoughts (ToT)? How does it differ from CoT?***

- CoT is a **single linear** reasoning path; ToT organizes reasoning into a **tree**:
    - Each node is a partial solution (a "thought");
    - At each step, multiple candidate thoughts are generated;
    - The model **evaluates** the candidates (scoring or voting);
    - A **search algorithm** (BFS/DFS/beam search) expands promising branches and **backtracks** when necessary.
- Suited to problems that require exploration and backtracking (e.g. the Game of 24, puzzles, planning for creative writing).
- Downsides: many calls and high cost; limited benefit for many everyday tasks.
- Related extensions: **Graph-of-Thoughts** (allows thoughts to merge into a graph structure), combining **Monte Carlo Tree Search (MCTS)** with LLMs, etc.

Q6: ***What are reasoning models? How do they differ from ordinary LLMs?***

- Reasoning models are specially trained models that generate a **long internal thinking process** before answering (e.g. the OpenAI o-series, DeepSeek-R1, and the extended thinking modes of Claude, Gemini and others).
- Main differences:
    - Trained with **reinforcement learning** (often on verifiable rewards such as math answers and code tests), the model learns behaviors such as self-verification, backtracking and trying different approaches;
    - They spend more tokens "thinking" at inference time and are **clearly stronger on complex tasks such as math, programming and scientific reasoning**;
    - The cost is higher latency and expense, and they may "overthink" simple tasks.
- Many APIs support setting a **thinking budget/reasoning effort** to trade off quality against cost.

Q7: ***What is test-time compute scaling?***

- It means investing more computation at the **inference stage** to improve performance, rather than relying only on bigger models or more training data.
- Two broad categories:
    - **Sequential scaling**: have the model generate longer chains of thought (long thinking in reasoning models, reflection and revision);
    - **Parallel scaling**: sample multiple answers and then select, e.g. Best-of-N, majority voting, selection with a **verifier/reward model**, tree search.
- Research shows that on many reasoning tasks, increasing test-time compute can yield gains comparable to or greater than scaling up the model, but the gains gradually saturate and depend on problem difficulty.
- For agents, "multi-step tool calling + reflection" is itself a form of test-time compute.

Q8: ***What are process reward models (PRMs) and outcome reward models (ORMs)?***

- **ORM (Outcome Reward Model)**: scores only the final answer;
- **PRM (Process Reward Model)**: scores **each step** of the reasoning, can detect wrong steps earlier, and can guide search (e.g. pruning low-scoring branches in beam search).
- PRMs require step-level annotations and are expensive; for verifiable tasks (math, code), **rule-based verifiers** can be used directly as rewards.

Q9: ***In agents, what does "interleaved thinking" mean?***

- The model does not just think once at the start; it can **reason again after every tool result** and then decide the next action.
- Benefits: it can correct the plan promptly based on new observations, reflect on whether the tool result matches expectations, and decide whether more information is needed.
- This is essentially the same as ReAct's Thought-Action-Observation loop, only natively supported by the model.

Q10: ***What are common failure modes in agent planning?***

- **Unrealistic plans**: assuming tools, permissions or information that do not exist;
- **Rigid plans**: sticking to the original plan after the environment changes or a step fails, without replanning;
- **Over-planning**: generating lengthy plans for simple tasks, wasting cost;
- **Wrong decomposition granularity**: subtasks too coarse to execute, or too fine, raising coordination cost;
- **Missed constraints**: forgetting some of the user's requirements (e.g. budget, format);
- **Goal drift**: gradually deviating from the original goal over a long task;
- **Premature completion**: considering the task done without verification;
- **Long-horizon error accumulation**: the more steps, the greater the impact of early mistakes.

Q11: ***How can these planning failures be mitigated?***

- Have the model maintain an **explicit plan/todo list** and update its status after each step;
- Add **verification steps**: confirm each completed subtask with tests, checkers or an evaluator model;
- Trigger **replanning** on failure rather than retrying blindly;
- Periodically **restate the original goal and constraints** in long tasks (e.g. keep the goal statement in the context);
- Introduce **human review** of critical plans;
- Provide an explicit "definition of done".

Q12: ***Is CoT reasoning faithful? What does that mean for agents?***

- Research has found that the reasoning a model writes **does not necessarily reflect the true basis of its decisions**: the model may be influenced by hints in the prompt without acknowledging them in its reasoning, or may arrive at an answer first and then "rationalize" it.
- Implications:
    - CoT cannot be treated as a fully reliable explanation or safety-monitoring mechanism;
    - But CoT monitoring is still one **useful signal** for detecting abnormal behavior (e.g. reward hacking, intent to violate rules).
- Practice: combine with behavior-level evaluation (what actions were actually executed) rather than relying only on the reasoning text.

Q13: ***How should prompts for reasoning models be written? How do they differ from ordinary models?***

- Reasoning models have built-in thinking, so they usually **do not need** prompts like "think step by step", nor overly detailed prescriptions of reasoning steps;
- It works better to give **clear goals, constraints and success criteria** and let the model plan by itself;
- Few-shot examples sometimes have limited effect and may even constrain the model;
- Cost can be controlled by adjusting the thinking budget; for simple tasks thinking can be turned off or reduced.

Q14: ***What is the "overthinking" problem?***

- Reasoning models **generate many thinking tokens even for simple questions**, causing unnecessary latency and cost, and sometimes changing correct answers into wrong ones through repeated self-doubt.
- Mitigation: **route** by task difficulty (non-reasoning mode or a small model for simple tasks), set thinking budgets, use adaptive thinking (letting the model decide how much to think).

Q15: ***How do you decide whether a task should use a reasoning model or an ordinary model?***

- **Suited to reasoning models**: multi-step math/logic, complex programming and debugging, long tasks requiring planning, analysis of ambiguous problems, review of high-stakes decisions;
- **Suited to ordinary/fast models**: simple QA, classification, extraction, format conversion, latency-sensitive interaction;
- A common architecture: **use a reasoning model for planning and key decisions, and fast models to execute many simple subtasks**.
- Ultimately decide through evaluation on your own tasks rather than intuition.

Q16: ***What is Reflexion? How is it similar to reinforcement learning?***

- Reflexion has the agent, after a failed attempt, **reflect in natural language** on the cause of failure (e.g. "I used the wrong API in step 3"), store the reflection in memory, and add it to the context on the next attempt.
- Similarity: like reinforcement learning, it uses a reward signal to improve the policy, but it **does not update model weights**; instead it stores the "lessons learned" as text in the context/memory (described as "verbal reinforcement learning").
- Prerequisites: a reliable success/failure signal (e.g. test results) and permission to make multiple attempts.
