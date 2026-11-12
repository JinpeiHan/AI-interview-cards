# Agent Fundamentals

Q1: ***What is an LLM agent?***

- An LLM agent is a system that uses a large language model as its "brain" and **autonomously decides its next action**: in a loop, the model observes the environment, reasons, calls tools (search, code execution, APIs, etc.), and keeps making decisions based on the tool results until the goal is achieved or a stopping condition is met.
- A common simplified definition: **Agent = LLM + tools + loop + memory**.
- The key difference from an ordinary chatbot: an agent's **control flow is decided dynamically by the model**, rather than hard-coded in advance by the developer.

Q2: ***What is the difference between an agent and a workflow/chain?***

- **Workflow/chain**: LLMs and tools are orchestrated along **predefined code paths**, e.g. "classify → retrieve → generate". The steps are fixed, predictable and easy to test.
- **Agent**: the LLM **decides by itself** which tools to call, how many times, and in what order; the number of steps is not known in advance.
- The two form a spectrum: single call → prompt chaining → routing → parallelization → orchestrator-workers → fully autonomous agent. **The more autonomy, the more flexibility, but also the higher the cost, latency and unpredictability.**

Q3: ***What are the common workflow patterns?***

- **Prompt chaining**: split the task into fixed sequential steps, with each step's output feeding the next; programmatic checks (gates) can be inserted in between.
- **Routing**: classify the input first, then dispatch it to a specialized prompt or model (e.g. easy questions to a small model, hard ones to a large model).
- **Parallelization**: process independent subtasks in parallel (sectioning), or run the same task multiple times and take a consensus (voting).
- **Orchestrator-workers**: a central LLM dynamically breaks down subtasks, dispatches them to worker LLMs, and then aggregates the results.
- **Evaluator-optimizer**: one LLM generates, another LLM evaluates and gives feedback, iterating in a loop to improve.

Q4: ***Describe the basic agent loop***

1. Receive the user's goal and combine it with the system prompt and tool definitions to form the context;
2. The LLM reasons and outputs either a **final answer** or one or more **tool calls**;
3. The runtime (harness) executes the tool calls and appends the **tool results (observations)** to the context;
4. Go back to step 2 until the model gives a final answer, a maximum number of steps/budget is reached, or human intervention is triggered.

- Key point: the model itself does not execute anything — **the external runtime is what actually runs the tools**; the loop's termination conditions, budget and error handling must be explicitly designed in the runtime.

Q5: ***What is ReAct? What are its pros and cons?***

- **ReAct (Reasoning + Acting)**: the model alternately outputs **Thought → Action → Observation**, interleaving the reasoning process with external interaction.
- Pros:
    - Reasoning can adapt dynamically to real observations, reducing the **hallucination and error propagation** of pure CoT;
    - The trajectory is readable, which makes debugging and explanation easier.
- Cons:
    - Every step requires an LLM call, so **latency and token cost are high**;
    - It easily gets stuck in loops (repeatedly calling the same tool) or stops too early;
    - Long trajectories bloat the context.
- Native function calling and "interleaved thinking" in modern models are essentially engineered implementations of the ReAct idea.

Q6: ***What is the Plan-and-Execute pattern? How does it compare with ReAct?***

- A **planner** first generates a complete multi-step plan, then an **executor** carries it out step by step, **replanning** when necessary.
- Compared with ReAct:
    - Pros: better global view, and the plan can be reviewed; execution steps can use cheaper models or even run in parallel, **reducing the number of large-model calls**;
    - Cons: the initial plan may rest on wrong assumptions, and if the environment changes quickly without timely replanning, the agent **executes a stale plan**.
- In practice the two are often combined: a high-level plan plus a ReAct-style loop within each step.

Q7: ***What is reflection? What are typical methods?***

- Reflection means the agent **critiques and corrects its own output or trajectory** to improve quality.
- Typical methods:
    - **Self-Refine**: generate → self-feedback → improve, iterated for several rounds;
    - **Reflexion**: after a failed task, summarize the cause of failure in natural language, store it in memory, and consult it on the next attempt ("verbal reinforcement learning signal");
    - **Evaluator-optimizer**: use an independent evaluator model or external signals (unit tests, compilers, validators) to provide feedback.
- Note: **pure self-reflection without external feedback has limited effect** — the model may fail to spot its own errors, or even change correct answers into wrong ones; reflection works best when there is a verifiable signal (tests, execution results).

Q8: ***What types of memory does an agent have?***

- **Short-term memory (working memory)**: the context window of the current session, including conversation history, tool calls and results. Limited by the context length.
- **Long-term memory**: information persisted across sessions, usually stored externally (vector stores, databases, files) and retrieved back into the context when needed. It is often further divided into:
    - **Episodic memory**: past interactions and experiences;
    - **Semantic memory**: facts and knowledge, such as user preferences;
    - **Procedural memory**: how to do things, such as system prompts, skills and learned rules.
- In addition, the model weights themselves can be viewed as **parametric memory**.

Q9: ***What are the core components of an agent?***

- **Model**: responsible for reasoning and decision-making;
- **Tools**: interfaces for interacting with the outside world (APIs, code execution, browser, file system, etc.);
- **Instructions/system prompt**: define the role, goals, constraints and behavioral norms;
- **Memory and context management**: decide what information enters the context;
- **Runtime (harness/orchestration)**: runs the loop, calls tools, handles errors, limits the budget, and records logs;
- **Guardrails**: input/output checks, permission control, human approval.

Q10: ***When should you NOT use an agent?***

- When task steps are **fixed and predictable**, a workflow or ordinary code is cheaper, more reliable and easier to test;
- Problems that a single LLM call + retrieval + a few examples can solve;
- Scenarios extremely sensitive to **latency and cost** (agents usually call the model many times);
- Operations where errors are costly and **hard to verify or roll back**, and human review cannot be added;
- When good tools and evaluation methods are lacking, agent behavior is hard to debug.
- Principle: **start with the simplest solution, and add autonomy only when truly necessary (open-ended problems with unpredictable steps).**

Q11: ***What kinds of tasks are suitable for agents?***

- **Open-ended problems** where it is hard or impossible to hard-code the number of steps and the path in advance;
- Tasks that require multi-turn interaction with an environment and adjustment based on feedback, such as coding, research, data analysis and complex customer support;
- Tasks whose results **can be verified** (whether tests pass, whether the answer can be checked), which lets the agent self-correct;
- Cases where some latency and cost are acceptable, and there is enough trust and sandboxing to let the model operate autonomously.

Q12: ***Why is "context engineering" considered central to agent development?***

- Each decision the model makes depends only on the current context, so **what information goes in, in what format, and how much** determines the agent's ceiling.
- Context is a finite resource: too much content raises cost, dilutes attention, and buries key information (often called "context rot").
- Common techniques: a lean system prompt, just-in-time retrieval, truncating/summarizing tool results, context compaction, external notes (e.g. a to-do file), and subagents for context isolation.
- Prompt engineering focuses on "how to write a good prompt"; context engineering focuses on "**how to manage all the tokens entering the model across the entire loop**".

Q13: ***How should an agent's "tools" and "action space" be designed?***

- Tools are the agent's only way to interact with the environment and can be seen as an **agent-computer interface (ACI)**; their design matters as much as the prompt itself.
- Principles:
    - Keep tools **few and well-crafted**, with non-overlapping functions, so the model is not confused when choosing;
    - Make names, descriptions and parameters clear — write tool descriptions as if documenting for a new employee;
    - Return **high-signal** results and avoid returning large amounts of irrelevant data;
    - Make mistakes hard to make (poka-yoke), e.g. require absolute paths instead of relative paths.

Q14: ***How do you control an agent's termination conditions and avoid infinite loops?***

- Set a **maximum number of steps/tool calls**, a **token or cost budget**, and a **timeout**;
- Detect repeated actions (the same tool called with the same arguments over and over) and abort or prompt the model to change strategy;
- State the criteria for "done" explicitly in the prompt, and provide an explicit "finish/submit answer" tool;
- Set **human checkpoints** at key points;
- Exit gracefully when limits are hit: return partial results and current progress rather than failing silently.

Q15: ***What are common agent failure modes?***

- **Hallucinated tool calls**: calling tools that do not exist or fabricating arguments;
- **Loops/getting stuck**: repeating the same action;
- **Premature termination**: claiming completion before finishing;
- **Goal drift**: gradually drifting away from the original goal over a long task;
- **Error accumulation**: early errors amplified by later steps (compounding error — if each step succeeds with probability $p$, the probability that all $n$ steps succeed is roughly $p^n$);
- **Context overflow**: overly long tool results crowd out key information;
- **Hijacking by prompt injection**: executing malicious instructions from web pages/documents.

Q16: ***How do you choose between a single agent and multiple agents?***

- **Start with a single agent** by default: simple architecture, unified context, easy to debug;
- Consider multiple agents when the task is **highly parallelizable** (e.g. breadth-first research), does not fit in a single context, or requires strict isolation of responsibilities/permissions;
- Multi-agent systems bring **higher token consumption**, coordination overhead, and new failure modes (information loss, duplicated work).

Q17: ***What role do agent frameworks (e.g. LangGraph, vendors' agent SDKs) play? Are they required?***

- Frameworks provide common capabilities: the agent loop, tool registration, state management, memory, tracing, multi-agent orchestration, human-in-the-loop, etc., which speed up prototyping.
- Downsides: abstraction layers may **hide the underlying prompts and responses**, making debugging harder and encouraging over-engineering.
- Recommendation: **implement directly with the model API first** (the core loop is only a few dozen lines of code), understand the underlying mechanics, then adopt a framework as needed; when using one, make sure you understand its underlying behavior.
