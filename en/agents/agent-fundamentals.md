# Agent Fundamentals

Q1: ***What is an LLM agent?***

- **An LLM agent is a system that uses an LLM as its "brain" and autonomously decides its next action in a loop: observe, reason, call tools, then keep deciding based on the results until the goal is reached.**
- A common simplified definition: Agent = LLM + tools + loop + memory.
- The difference from an ordinary chatbot: the control flow is decided dynamically by the model, not hard-coded by the developer.

Q2: ***What is the difference between an agent and a workflow/chain?***

- **A workflow orchestrates LLMs and tools along predefined code paths, so the steps are fixed and predictable; an agent lets the LLM decide which tools to call, how many times and in what order.**
- They form a spectrum: single call → prompt chaining → routing → parallelization → orchestrator-workers → fully autonomous agent.
- More autonomy means more flexibility, but also higher cost, latency and unpredictability.

Q3: ***What are the common workflow patterns?***

- **There are five common ones: prompt chaining, routing, parallelization, orchestrator-workers, and evaluator-optimizer.**
- Prompt chaining: fixed sequential steps, each output feeding the next, with programmatic checks in between.
- Routing: classify first, then dispatch to a specialized prompt or model (easy questions to a small model).
- Parallelization: run independent subtasks in parallel (sectioning), or run several times and vote.
- Orchestrator-workers: a central LLM dynamically splits, dispatches and aggregates tasks; evaluator-optimizer: one LLM generates, another critiques, looping to improve.

Q4: ***Describe the basic agent loop***

- **Given the goal, system prompt and tool definitions, the model outputs either a final answer or tool calls; the runtime executes the tools and appends the results to the context, and the model continues until it answers or a stop condition fires.**
- The model itself executes nothing; the external runtime (harness) actually runs the tools.
- Termination conditions (max steps, budget, human intervention) and error handling must be designed explicitly in the runtime.

Q5: ***What is ReAct? What are its pros and cons?***

- **ReAct has the model alternate "Thought → Action → Observation", interleaving reasoning with external interaction so the reasoning can adapt to real feedback.**
- Pros: less hallucination and error propagation than pure CoT; readable trajectories that are easy to debug.
- Cons: one LLM call per step, so high latency and cost; prone to loops or stopping early; long trajectories bloat the context.
- Native function calling and interleaved thinking in modern models are essentially engineered versions of ReAct.

Q6: ***What is the Plan-and-Execute pattern? How does it compare with ReAct?***

- **A planner first produces a complete plan, then an executor carries it out step by step, replanning when needed.**
- Pros: better global view and the plan can be reviewed; execution can use cheaper models or run in parallel, reducing large-model calls.
- Cons: the initial plan may rest on wrong assumptions, and in a fast-changing environment the agent may execute a stale plan.
- In practice they are combined: a high-level plan plus a ReAct loop within each step.

Q7: ***What is reflection? What are typical methods?***

- **Reflection means the agent critiques and corrects its own output or trajectory to improve quality; it works best when there is an external, verifiable signal.**
- Self-Refine: generate → self-feedback → improve, for a few rounds.
- Reflexion: after a failure, summarize the cause in words, store it in memory, and consult it on the next attempt.
- Evaluator-optimizer: an independent evaluator model or external signals such as tests and compilers provide feedback.
- Pure self-reflection without external signals has limited effect and can even turn right answers into wrong ones.

Q8: ***What types of memory does an agent have?***

- **Short-term memory (the current context window) and long-term memory (stored externally, persisted across sessions, retrieved into context when needed).**
- Long-term memory is often split into episodic (past interactions), semantic (facts and user preferences) and procedural (how to do things, e.g. skills and rules).
- The model weights themselves can be seen as parametric memory.

Q9: ***What are the core components of an agent?***

- **Model, tools, instructions, memory and context management, runtime, and guardrails.**
- The model reasons and decides; tools are the interfaces to the outside world; the system prompt defines role, goals and constraints.
- Memory and context management decide what information enters the context.
- The runtime runs the loop, executes tools, and handles errors and budgets; guardrails handle input/output checks, permissions and human approval.

Q10: ***When should you NOT use an agent?***

- **Don't use an agent when the steps are fixed and predictable, or when a single LLM call plus retrieval solves the problem; start with the simplest solution and add autonomy only when truly needed.**
- Scenarios extremely sensitive to latency and cost (agents call the model many times).
- Costly operations that are hard to verify or roll back, where human review cannot be added.
- When good tools and evaluation are lacking, agent behavior is hard to debug.

Q11: ***What kinds of tasks are suitable for agents?***

- **Open-ended tasks whose steps and path can't be hard-coded, and that need multi-turn interaction with an environment and adjustment based on feedback, such as coding, research and data analysis.**
- Results are verifiable (tests pass, answers can be checked), so the agent can self-correct.
- Some latency and cost are acceptable, and there is enough sandboxing and trust for the model to act autonomously.

Q12: ***Why is "context engineering" considered central to agent development?***

- **The model decides each step based only on the current context, so what goes in, in what form, and how much directly sets the agent's ceiling.**
- Context is finite: too much content raises cost, dilutes attention and buries key information (context rot).
- Common techniques: lean system prompts, just-in-time retrieval, truncating or summarizing tool results, compaction, external notes, and subagent isolation.
- Prompt engineering is about writing one good prompt; context engineering is about all tokens entering the model across the whole loop.

Q13: ***How should an agent's "tools" and "action space" be designed?***

- **Tools are the agent's only way to interact with the environment (the agent-computer interface, ACI), so design them as carefully as documentation for a new employee.**
- Few and well-crafted, with non-overlapping functions, so the model isn't confused when choosing.
- Clear names, descriptions and parameters; return high-signal results without irrelevant data.
- Poka-yoke design that makes mistakes hard, e.g. require absolute paths.

Q14: ***How do you control an agent's termination conditions and avoid infinite loops?***

- **Set hard limits in the runtime (max steps, token or cost budget, timeout), and tell the model clearly what counts as "done".**
- Detect repeated actions (same tool with the same arguments) and abort or prompt a change of strategy.
- Provide an explicit "finish/submit answer" tool and human checkpoints at key points.
- Exit gracefully at the limit, returning partial results and progress instead of failing silently.

Q15: ***What are common agent failure modes?***

- **Common ones are hallucinated tool calls, loops, premature termination, goal drift, error accumulation, context overflow, and hijacking by prompt injection.**
- Error accumulation is the most damaging: early small mistakes get amplified, and success rate drops exponentially with task length.
- Context overflow: overly long tool results crowd out key information.
- Prompt injection: executing malicious instructions from web pages or documents.
- Key formula: if each step succeeds with probability $p$, all $n$ steps succeed with probability about $p^n$

Q16: ***How do you choose between a single agent and multiple agents?***

- **Start with a single agent by default: simple architecture, unified context, easy to debug. Use multiple agents only when the task is highly parallelizable, doesn't fit in one context, or needs isolated responsibilities or permissions.**
- The cost of multi-agent: higher token usage, coordination overhead, and new failure modes like information loss and duplicated work.

Q17: ***What role do agent frameworks (e.g. LangGraph, vendors' agent SDKs) play? Are they required?***

- **They are not required. Frameworks provide the loop, tool registration, state, memory, tracing and multi-agent orchestration, which speeds up prototyping.**
- Downsides: abstraction layers hide the underlying prompts and responses, making debugging harder and encouraging over-engineering.
- Recommendation: implement directly with the model API first (the core loop is a few dozen lines), understand the mechanics, then adopt a framework if needed.
