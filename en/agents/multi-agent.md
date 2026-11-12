# Multi-Agent Systems

Q1: ***What is a multi-agent system?***

- A system in which **multiple LLM agents collaborate** to complete a task; each agent may have a different role, prompt, tools, model and even permissions.
- Agents communicate and collaborate through message passing, shared state or tool calls.
- Motivations: division of labor and specialization, parallel speed-up, overcoming the limits of a single context window, and isolating responsibilities and permissions.

Q2: ***What are common multi-agent architectures?***

- **Orchestrator-worker**: a lead agent plans and splits the task, dispatches subtasks to multiple worker agents (often in parallel), and finally aggregates the results;
- **Supervisor/hierarchical**: a supervisor agent decides, based on the current state, which specialist agent to hand control to; this can be nested in multiple levels;
- **Handoff/routing**: agents directly transfer control of the conversation to a more suitable agent (e.g. in customer support, from a triage agent to a refund agent);
- **Sequential pipeline**: multiple agents process in a fixed order (e.g. researcher → writer → reviewer);
- **Debate/multi-role discussion**: multiple agents give views on the same question and critique each other, with a judge or consensus producing the conclusion;
- **Decentralized/network (swarm)**: agents communicate peer to peer, with no central controller.

Q3: ***How does the orchestrator-worker pattern work? What are its pros and cons?***

- Flow:
    1. The orchestrator analyzes the task, makes a plan and splits it into subtasks;
    2. It writes clear instructions for each subtask (goal, output format, available tools, boundaries) and launches worker agents;
    3. Workers execute in **their own independent contexts** and return distilled results;
    4. The orchestrator integrates the results, launches another round of subtasks if necessary, and finally outputs the answer.
- Pros: easy to parallelize, context isolation, the orchestrator keeps a global view;
- Cons: the orchestrator is a bottleneck and single point of failure; unclear subtask descriptions lead workers to duplicate work or go off-topic; high token consumption.

Q4: ***What is a subagent? Why is it said to provide "context isolation"?***

- A subagent is an agent launched by the main agent with its **own independent context window**, used to complete a well-bounded subtask (e.g. "find every place in the codebase that calls X").
- Context isolation: while exploring, a subagent may read many files, web pages and tool outputs; this **intermediate content lives only in the subagent's context**, and the main agent receives only the final distilled summary.
- Benefits:
    - The main agent's context stays clean and is not polluted by irrelevant details (mitigating context rot);
    - Multiple subagents can run in parallel;
    - Subagents can be configured with different tools, permissions or cheaper models.
- Costs: the main agent cannot see the details, so if the summary omits information, later decisions suffer; launching a subagent also has its own overhead.

Q5: ***How do agents communicate with each other?***

- **Message passing**: agents send requests and results to each other in natural language or structured messages (JSON);
- **Shared state/blackboard**: all agents read and write the same state object or shared store;
- **Shared file system**: subagents write detailed results to files and return only the file paths, avoiding information loss from "relaying" through the context;
- **Tool-style calls**: wrap one agent as a tool another agent can call (agent-as-tool);
- **Standard protocols**: e.g. A2A (Agent2Agent), used for agents from different vendors/frameworks to discover each other's capabilities and collaborate; MCP, by contrast, mainly connects agents to tools/data.

Q6: ***What are the cost and latency characteristics of multi-agent systems?***

- **Cost**: each agent has its own system prompt, tool definitions and context, so a multi-agent system's **token consumption is usually significantly higher than a single agent's** (possibly several times or more).
- **Latency**:
    - Running subtasks in parallel can **reduce total wall-clock time** (for breadth-oriented tasks);
    - But sequential multi-round coordination, handoffs and aggregation **add latency**.
- Multi-agent systems are therefore suited to **high-value, parallelizable, information-heavy** tasks; for simple or tightly coupled tasks that need closely shared context (such as making a single change in most coding tasks), a single agent is often more cost-effective.

Q7: ***When should you use multiple agents, and when not?***

- Suitable:
    - **Breadth-first, parallelizable** tasks (e.g. researching several independent directions at once);
    - The amount of information exceeds a single context window;
    - A clear division of labor requiring different specialized tools/permissions;
    - An independent perspective is needed for review (e.g. separating generator and reviewer).
- Not suitable:
    - Subtasks are **strongly interdependent and need to share a lot of context**;
    - Highly consistent output is required (agents deciding independently easily conflict, e.g. several agents writing the same code in parallel produce inconsistent styles);
    - Simple tasks that are sensitive to cost or latency.

Q8: ***What are typical failure modes of multi-agent systems?***

- **Unclear task descriptions**: vague instructions from the orchestrator cause workers to duplicate work, miss things or misunderstand;
- **Information lost in transmission**: the "telephone game" effect, where summaries drop key details;
- **Conflicting decisions**: parallel agents make contradictory implicit assumptions, leading to inconsistency when results are merged;
- **Over-spawning agents**: launching many subagents even for simple questions, wasting resources;
- **Loops and deadlocks**: agents handing off to each other back and forth or waiting on each other;
- **Role confusion**: agents not respecting their role boundaries;
- **Missing verification**: no agent is responsible for the final result, so errors are passed along layer by layer;
- **Error cascades**: one agent's erroneous output becomes other agents' input.

Q9: ***How can you improve the reliability of a multi-agent system?***

- The orchestrator's instructions to subagents should include: **a clear goal, output format, available tools and information sources, task boundaries, and when to stop**;
- Set the **scale of effort** according to task complexity (few calls for simple questions, more subagents for complex ones);
- Use **structured output** and shared files to reduce information loss;
- Add a dedicated **verification/review agent** or test-based checks;
- Full **tracing** that records each agent's inputs, outputs and tool calls;
- Support **checkpointing and resumption** for long-running systems, to avoid restarting from scratch after a failure;
- Iterate on the orchestrator's prompt through evaluation (many problems stem from how the orchestrator delegates).

Q10: ***What is multi-agent debate? What is it for?***

- Several agents (instances of the same model or different models) each give an answer, then **read each other's answers and rebut or revise them**; after several rounds, a judge agent or a vote produces the final answer.
- Purpose: can improve accuracy on some reasoning and factuality tasks and reduce the bias of a single model.
- Limitations:
    - High cost;
    - Multiple instances of the same model may **share the same biases** and tend to converge on a wrong answer (conformity effect);
    - Some studies point out that, under the same compute budget, its gains are not necessarily better than simpler methods such as self-consistency voting.

Q11: ***How should handoffs between agents be designed?***

- A handoff transfers **control of the conversation** from one agent to another, commonly used in multi-domain scenarios such as customer support.
- Design points:
    - Implement the handoff as a tool (e.g. `transfer_to_refund_agent`) and let the model decide when to call it;
    - Pass the **necessary context** during the handoff (user information, information already collected, a summary of the problem) so the user does not have to repeat themselves;
    - Define clear responsibility boundaries to prevent passing the buck back and forth;
    - Provide a path to hand back to the main agent or escalate to a human.

Q12: ***How is shared state managed in a multi-agent system?***

- Define a **single source of truth** for state (e.g. a database, shared file, or state object maintained by the orchestrator);
- Lock concurrent writes or use append-only logs to avoid race conditions;
- Each agent reads only the state needed for its task, avoiding context bloat;
- Version and log state changes to make tracing back and debugging easier.

Q13: ***How do you evaluate a multi-agent system?***

- **End-to-end results**: task success rate and answer quality (can be assessed with LLM-as-judge against a rubric);
- **Process metrics**: number of subagents, number of tool calls, token consumption, time taken, proportion of duplicated work;
- **Trajectory analysis**: check whether delegation was reasonable and whether information was lost in transmission;
- Because multi-agent systems are **highly non-deterministic** and the same input may take different paths on different runs, focus on whether the outcome achieves the goal rather than requiring a fixed sequence of steps;
- Compare against a single-agent baseline to confirm that the extra cost brings commensurate benefit.

Q14: ***How are permissions and security handled in a multi-agent system?***

- **Principle of least privilege**: each agent has only the tools and permissions needed for its responsibilities;
- **Isolate** agents that process untrusted input (e.g. browsing the web, reading email) from agents that hold high-privilege operations, passing only constrained structured data between them;
- Prevent prompt injection from **propagating** between agents: messages from other agents should also be treated as not fully trusted;
- Keep human approval for high-risk actions.
