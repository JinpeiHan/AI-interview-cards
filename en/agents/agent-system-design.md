# Agent System Design

Q1: ***What is a general framework for answering agent system design questions?***

1. **Clarify requirements**: who the users are, what tasks to accomplish, success criteria, scale (QPS, concurrency), latency and cost requirements, risk level;
2. **Decide whether an agent is really needed**: use workflows wherever a workflow can solve the problem;
3. **Core architecture**: model choice, tools and data sources, context and memory, single agent or multiple agents;
4. **Reliability and safety**: guardrails, permissions, sandboxing, human-in-the-loop, error handling and fallbacks;
5. **Performance and cost**: caching, streaming output, model routing, parallelism;
6. **Evaluation and observability**: offline evaluation sets, online monitoring, feedback loop.

Q2: ***Design an e-commerce customer support agent***

- **Requirements**: answer questions about orders, shipping, returns/exchanges and products; perform actions such as looking up orders and initiating refunds; 24/7, low latency, compliant with company policy.
- **Architecture**:
    - **Routing/triage**: first identify the intent (small model or classifier); simple FAQs go through RAG for a direct answer, complex transactions go to an agent with tools, or are handed off by domain to different specialist agents;
    - **Tools**: `get_order`, `track_shipment`, `create_return`, `issue_refund`, `search_kb`, `escalate_to_human`;
    - **Knowledge**: policy documents and FAQs are retrieved via RAG so that answers are based on the latest policies;
    - **User context**: after authentication, load the user's information and recent orders; tools can only access **the current user's** data.
- **Guardrails and safety**:
    - Refunds above a threshold or outside policy require human approval;
    - Detect jailbreaks/injection on input, and check on output whether commitments (e.g. "we'll give you a full refund") exceed policy;
    - **Escalate to a human** when the user is upset, the issue remains unresolved after several attempts, or the user asks, attaching a conversation summary.
- **Evaluation**: evaluate with simulated users (similar to τ-bench) plus replays of real tickets, focusing on resolution rate, policy compliance rate, escalation rate, user satisfaction, and consistency across multiple runs.

Q3: ***Design a coding agent***

- **Requirements**: understand requirements within a code repository, modify code, run tests and submit changes; can run in an IDE/terminal/CI.
- **Core tools**: read files, search (grep/glob), edit files (based on exact string replacement or diffs), execute shell commands (run tests, build), version control operations.
- **Context strategy**:
    - Do not put the whole repository into the context up front; let the agent **explore on demand** through search;
    - Load project-level instruction files (e.g. build commands, coding conventions) at startup;
    - For long tasks, use a todo list and context compaction, or dispatch subagents for wide-ranging code searches.
- **Verification loop**: after modifying code, run tests, linting and type checks, and iterate based on failures; this is the key to why coding agents work well — **there is a verifiable feedback signal**.
- **Safety**: execute commands in a sandbox/container with restricted network access; dangerous commands (e.g. `rm -rf`, `git push --force`, deployment) require confirmation; do not expose production credentials; all changes are reversible via git and are merged only after human code review.
- **Evaluation**: SWE-bench-style tasks plus real internal issues, focusing on test pass rate, whether changes are minimal, and whether there is gaming behavior such as modifying tests.

Q4: ***Design a deep research agent***

- **Requirements**: for complex questions, search many web pages/documents, synthesize multiple sources, and output a long report with citations; minute-level run times are acceptable.
- **Architecture (orchestrator-worker)**:
    1. **Clarify**: confirm the research scope with the user first if necessary;
    2. **Plan**: the lead agent decomposes the question into several relatively independent sub-directions;
    3. **Parallel research**: launch a subagent for each direction; each performs multiple rounds of searching, reading and filtering in its own independent context, returning only distilled findings and sources;
    4. **Synthesize**: the lead agent aggregates the results, identifies information gaps, and launches a new round of research if necessary;
    5. **Write and cite**: generate the report, with a dedicated step verifying that each claim is supported by a source.
- **Key challenges**: judging source credibility, handling conflicting information, avoiding duplicate searches, high token cost, and **indirect prompt injection** in web content.
- **Engineering points**: persist intermediate results (to allow resuming from checkpoints), control the number of subagents and the search budget, and provide progress feedback for long tasks.
- **Evaluation**: factual accuracy, citation accuracy, coverage, source quality — usually using LLM-as-judge with a rubric, supplemented by human spot checks.

Q5: ***How do you reduce agent latency?***

- **Streaming output**: show text and progress to the user as early as possible (e.g. "Looking up your order..."), reducing perceived latency;
- **Parallelism**: parallel tool calls, parallel subagents, and guardrail checks running in parallel with the main model;
- **Model routing**: use small models for simple steps and cut unnecessary reasoning/thinking tokens;
- **Prompt caching**: cache stable prefixes, significantly reducing time to first token (TTFT);
- **Fewer round trips**: merge commonly used tool operations so that one tool call accomplishes more; implement deterministic steps in code rather than leaving them to the model;
- **Prefetching**: load data likely to be needed in advance (e.g. the user profile);
- **Result caching**: reuse results for identical or semantically similar queries (paying attention to freshness and personalization).

Q6: ***What is prompt caching? How is it used in agents?***

- The model server caches the computation (KV cache) corresponding to a **prompt prefix**; if later requests have **exactly the same prefix**, it can be reused directly, **reducing input cost and time to first token**.
- Agents are a natural fit: every loop iteration resends the system prompt, tool definitions and previous conversation history, only appending new content at the end.
- Design points:
    - **Put stable content first** (tool definitions → system prompt → long documents → conversation history), and changing content last;
    - Avoid putting content that changes every time, such as timestamps or random IDs, in the prefix;
    - Keep the tool list and its order stable; dynamically adding or removing tools invalidates the cache;
    - Note that caches have a time-to-live (TTL), and vendors differ in usage (explicit breakpoints or automatic caching);
    - Context compaction changes the prefix, so trade off compaction frequency against cache hit rate.

Q7: ***What fallback strategies should be designed in an agent system?***

- **Model level**: when the primary model times out, is rate-limited or fails, switch to a backup model or another vendor;
- **Tool level**: retry when a tool fails; if it still fails, tell the model to use an alternative tool or ask the user for help;
- **Task level**: when the agent cannot complete the task, return partial results with a clear explanation of what is unfinished, or escalate to a human;
- **Quality level**: when output fails validation (format, guardrails), regenerate or return a safe default response;
- **Cost/budget level**: when the budget is exceeded, stop and report progress rather than running indefinitely.
- All fallback behavior should be logged and monitored.

Q8: ***In system design, how do you design a guardrails system?***

- **Layered design**:
    - Input layer: authentication, rate limiting, jailbreak/injection detection, topic scope checks, PII detection;
    - Model layer: a clear system prompt and behavioral guidelines;
    - Action layer: tool argument validation, permission checks, human approval for high-risk actions, sandboxed execution;
    - Output layer: harmful content filtering, policy compliance checks, groundedness checks, format validation.
- **Trade-offs**: overly strict guardrails cause false refusals and hurt user experience; cheap, fast models can run checks in parallel, with heavier checks reserved for high-risk scenarios.
- Guardrail rules also need evaluation sets to measure false positive and false negative rates.

Q9: ***How do you choose models for an agent?***

- Trade off among **task difficulty, latency and cost**:
    - Orchestration/planning and complex reasoning: the most capable model (thinking can be enabled);
    - Large numbers of simple subtasks (classification, summarization, extraction): small, fast models;
- Focus on tool-calling ability, long-context ability and instruction following, not just general leaderboards;
- Compare models on your own evaluation set in terms of **success rate × cost × latency**;
- Keep **models swappable** in the design (abstract the model interface) to make upgrades and downgrades easy.

Q10: ***What must be considered for long-running (multi-hour) agent tasks?***

- **State persistence and checkpoints**: save progress periodically so the task can resume after a crash or interruption instead of starting over;
- **Context management**: compaction, external notes and subagents to avoid context overflow and quality degradation;
- **Progress visibility**: report progress to the user and allow inspection and intervention midway;
- **Asynchronous execution**: the task runs in the background and notifies the user on completion;
- **Budgets and timeouts**: prevent runaway behavior;
- **Environment consistency**: the sandbox environment is reproducible, and changes in external dependencies can be detected.

Q11: ***How do you design an agent's memory system to support personalization?***

- Distinguish **in-session memory** (context) from **cross-session memory** (user preferences, historical facts);
- At the end of a session, or via a memory tool, extract information worth remembering and store it in per-user isolated storage;
- At the start of a new session, retrieve relevant memories into the context, or let the agent query on demand;
- Handle memory conflicts and expiry, and allow users to view, modify and delete memories;
- Prevent malicious instructions from being injected via memory; memory content should be treated as data, not instructions.

Q12: ***How do you design the evaluation and iteration process for an agent?***

- Build an **offline evaluation set** (real tasks + edge cases + adversarial examples) that runs automatically on every change;
- Combine **deterministic checks, LLM-as-judge and human review**;
- **Trace all trajectories** online and monitor success rate, latency, cost and user feedback;
- Regularly sample and analyze failed trajectories to determine whether the problem lies in the prompt, tools, retrieval or model capability;
- Add new failure cases to the evaluation set, forming a "find issue → fix → regression test" loop;
- Use A/B tests or gradual rollouts to validate the effect of changes in production.

Q13: ***How do you design an agent that can operate a browser/computer (computer use)?***

- **Perception**: screenshots (vision model) and/or the accessibility tree and DOM structure; combining the two is usually more reliable;
- **Actions**: clicking, typing, scrolling, key presses, or higher-level browser operations (navigation, filling forms);
- **Challenges**: many steps, high latency, dynamically changing pages, coordinate localization errors, prompt injection in web pages;
- **Safety**: run in an isolated VM/browser environment, do not log in to sensitive accounts (or use dedicated accounts), restrict reachable domains, and require human confirmation for critical actions such as payments and form submissions;
- **Optimization**: prefer APIs when available, and use GUI operation only when there is no API.

Q14: ***How are multi-tenancy and scalability handled in agent systems?***

- **Isolation**: each tenant's/user's data, memory, retrieval indexes and sandbox environment are isolated from each other;
- **Rate limits and quotas**: limit request rate and token usage per tenant to prevent a single tenant from exhausting resources;
- **Stateless services + external state storage**: keep the agent orchestration service as stateless as possible, storing session state in a database/cache to allow horizontal scaling;
- **Queues and asynchrony**: run long tasks through a task queue rather than holding request connections;
- **Handle model API rate limits**: retries, backoff, and spreading load across regions/providers.
