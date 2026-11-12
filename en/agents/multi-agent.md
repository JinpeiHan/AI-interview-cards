# Multi-Agent Systems

Q1: ***What is a multi-agent system?***

- **A system in which several LLM agents collaborate on a task; each agent can have its own role, prompt, tools, model and permissions, and they cooperate via messages, shared state or tool calls.**
- Motivations: division of labor and specialization, parallel speedup, getting past a single context window, and isolating responsibilities and permissions.

Q2: ***What are common multi-agent architectures?***

- **The most common is orchestrator-worker: a lead agent splits the task among workers and aggregates; others are supervisor/hierarchical, handoff, sequential pipeline, debate and decentralized networks.**
- Supervisor/hierarchical: a supervisor decides which specialist agent gets control based on state, possibly nested.
- Handoff: hand the conversation directly to a better-suited agent, e.g. triage agent to refund agent.
- Sequential pipeline: researcher → writer → reviewer; debate: agents critique each other, and a judge or consensus decides.
- Decentralized (swarm): peer-to-peer communication with no central controller.

Q3: ***How does the orchestrator-worker pattern work? What are its pros and cons?***

- **The orchestrator analyzes the task, splits it into subtasks, writes clear instructions for each and launches workers; workers run in their own isolated contexts and return distilled results, which the orchestrator aggregates, starting another round if needed.**
- Subtask instructions should include the goal, output format, available tools and boundaries.
- Pros: easy parallelism, context isolation, and the orchestrator keeps a global view.
- Cons: the orchestrator is a bottleneck and single point of failure; unclear instructions cause duplication or off-topic work; high token usage.

Q4: ***What is a subagent? Why is it said to provide "context isolation"?***

- **A subagent is an agent launched by the main agent with its own context window to handle a well-bounded subtask; the many files and tool outputs it reads while exploring stay in its own context, and the main agent receives only a distilled summary.**
- Benefits: the main context stays clean (mitigating context rot), several subagents can run in parallel, and each can have different tools, permissions or a cheaper model.
- Costs: the main agent can't see details, so omissions in the summary hurt later decisions; launching subagents has overhead.

Q5: ***How do agents communicate with each other?***

- **The main ways are message passing, shared state (blackboard), a shared file system, wrapping agents as tools, and standard protocols.**
- Messages: natural language or structured JSON requests and results.
- Shared files: a subagent writes detailed results to a file and returns only the path, avoiding "telephone game" loss.
- Agent-as-tool: wrap one agent as a tool another can call.
- Protocols: A2A lets agents from different vendors/frameworks discover each other and collaborate; MCP mainly connects agents to tools and data.

Q6: ***What are the cost and latency characteristics of multi-agent systems?***

- **Token cost is usually much higher than a single agent (several times or more), since every agent has its own system prompt, tool definitions and context; latency depends on whether work can run in parallel.**
- Parallel execution on breadth-type tasks cuts total time, but sequential coordination, handoffs and aggregation add latency.
- So multi-agent suits high-value, parallelizable, information-heavy tasks; for tightly dependent work that shares a lot of context (like most coding changes), a single agent is more cost-effective.

Q7: ***When should you use multiple agents, and when not?***

- **Use them for breadth-first, parallelizable tasks, information that exceeds one context, clear division of tools or permissions, or when an independent review perspective is needed; avoid them when subtasks are tightly dependent, output must be highly consistent, or a simple task is cost- or latency-sensitive.**
- A typical independent-review example is separating the generator from the reviewer.
- A consistency counterexample: several agents writing the same code in parallel tend to clash in style and assumptions.

Q8: ***What are typical failure modes of multi-agent systems?***

- **The most common are unclear task descriptions and information lost in transmission; others are conflicting decisions, spawning too many agents, loops and deadlocks, role confusion, missing verification and error cascades.**
- Unclear descriptions: workers duplicate, miss or misread work; lost information: summaries drop key details.
- Conflicting decisions: parallel agents make contradictory implicit assumptions that clash when merged.
- Loops/deadlocks: agents hand off to or wait on each other; missing verification: no one owns the final result, so errors cascade.

Q9: ***How can you improve the reliability of a multi-agent system?***

- **The key is the orchestrator's delegation quality: instructions to subagents must state the goal, output format, tools and sources, task boundaries and when to stop.**
- Scale effort to task complexity, spawning few subagents for simple questions.
- Use structured output and shared files to reduce information loss, and add a dedicated verifier agent or test-based checks.
- Trace every agent's inputs, outputs and tool calls; support checkpoints and resumption for long runs.
- Iterate on the orchestrator's prompt with evals, since many problems come from how it delegates.

Q10: ***What is multi-agent debate? What is it for?***

- **Several agents each give an answer, then read each other's answers and rebut or revise; after a few rounds a judge or vote decides. It can improve accuracy on some reasoning and factual tasks.**
- Limitation: high cost.
- Instances of the same model may share the same biases and converge on a wrong answer through conformity.
- At equal compute, it doesn't necessarily beat simpler methods like self-consistency voting.

Q11: ***How should handoffs between agents be designed?***

- **Implement the handoff as a tool (e.g. `transfer_to_refund_agent`) that the model decides when to call, and pass the necessary context so the user doesn't have to repeat themselves.**
- Context passed includes user info, information already collected, and a summary of the issue.
- Draw clear responsibility boundaries to prevent passing the buck back and forth.
- Provide a path back to the main agent or to a human.

Q12: ***How is shared state managed in a multi-agent system?***

- **Have a single source of truth (a database, shared file, or a state object maintained by the orchestrator), and control concurrent writes.**
- Lock concurrent writes or use append-only logs to avoid races.
- Each agent reads only the state it needs, to avoid context bloat.
- Version and log state changes for traceability and debugging.

Q13: ***How do you evaluate a multi-agent system?***

- **Mainly on end-to-end results (task success rate, answer quality), supported by process metrics and trajectory analysis, and compared against a single-agent baseline to confirm the extra cost pays off.**
- Process metrics: number of subagents, tool calls, tokens, time, share of duplicated work.
- Trajectory analysis: whether delegation was sensible and whether information was lost in transmission.
- Multi-agent runs are highly nondeterministic, so judge whether the goal was reached rather than requiring fixed steps.

Q14: ***How are permissions and security handled in a multi-agent system?***

- **The core is least privilege and isolation: each agent gets only the tools and permissions its role needs, and agents handling untrusted input are isolated from high-privilege agents, exchanging only constrained structured data.**
- Prevent prompt injection from spreading between agents; messages from other agents are not fully trusted either.
- Keep human approval for high-risk actions.
