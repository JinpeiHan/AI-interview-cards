# Agent System Design

Q1: ***What is a general framework for answering agent system design questions?***

- **First clarify requirements and decide whether an agent is really needed, then cover the core architecture, reliability and safety, performance and cost, and evaluation and observability.**
- Requirements: who the users are, what the task is, success criteria, scale, latency and cost targets, risk level.
- Use workflows for whatever parts they can handle.
- Architecture covers the model, tools and data sources, context and memory, single vs. multi-agent; safety covers guardrails, permissions, sandboxing, human-in-the-loop and fallbacks.
- Performance covers caching, streaming, routing and parallelism; finish with offline evals, online monitoring and the feedback loop.

Q2: ***Design an e-commerce customer support agent***

- **Start with triage routing: simple FAQs go to RAG for a direct answer, complex transactions go to a tool-using agent (or are handed off to domain specialists); tools can only access the current verified user's data, and high-risk actions need human approval.**
- Tools: `get_order`, `track_shipment`, `create_return`, `issue_refund`, `search_kb`, `escalate_to_human`; policies and FAQs via RAG so answers follow the latest policy.
- Guardrails: refunds above a threshold or outside policy need approval; detect jailbreaks/injection and check that promises in the output don't exceed policy.
- Hand off to a human, with a conversation summary, when the user is upset, the issue stays unresolved, or they ask for one.
- Evaluation: simulated users (τ-bench style) plus replayed real tickets, measuring resolution rate, policy compliance, escalation rate, satisfaction and consistency across runs.

Q3: ***Design a coding agent***

- **The core is a lean tool set (read files, search, edit, run shell, version control) plus a verification loop of "edit code → run tests → iterate on failures", all running in a sandbox.**
- Context: don't load the whole repo; explore on demand via search, load project instruction files at startup, and use to-do lists, compaction or subagents for broad searches on long tasks.
- Verifiable feedback (tests, lint, type checks) is what makes coding agents work well.
- Safety: sandboxed execution, restricted network, confirmation for dangerous commands (`rm -rf`, `git push --force`, deploys), no production credentials, git for rollback, and human code review before merge.
- Evaluation: SWE-bench-style tasks plus real internal issues, looking at test pass rate, minimal diffs, and gaming such as modifying tests.

Q4: ***Design a deep research agent***

- **Use orchestrator-workers: clarify scope if needed, have the lead agent split the question into independent directions, run subagents in parallel that search and read and return only distilled findings with sources, then aggregate, fill gaps, and write a cited report with a citation check.**
- Key challenges: source credibility, conflicting information, duplicate searches, high token cost, and indirect prompt injection from web pages.
- Engineering: persist intermediate results for resumption, cap the number of subagents and the search budget, and report progress on long runs.
- Evaluation: factual accuracy, citation accuracy, coverage and source quality, via LLM-as-judge with a rubric plus human spot checks.

Q5: ***How do you reduce agent latency?***

- **The most effective levers are streaming, parallelism and fewer model round trips, combined with model routing and prompt caching.**
- Streaming: show text and progress early (e.g. "Looking up your order..."), lowering perceived latency.
- Parallelism: parallel tool calls, parallel subagents, guardrails running alongside the main model.
- Small models for simple steps; prompt caching cuts time to first token.
- Merge common tool operations and implement deterministic steps in code; prefetch likely-needed data, and reuse results for identical or similar queries (mind freshness and personalization).

Q6: ***What is prompt caching? How is it used in agents?***

- **The server caches the computation (KV cache) for a prompt prefix, so later requests with exactly the same prefix reuse it, cutting input cost and time to first token; agents resend the system prompt, tool definitions and history every turn, so they benefit naturally.**
- Put stable content first (tool definitions → system prompt → long documents → conversation history) and changing content last.
- Keep timestamps and random IDs out of the prefix; keep the tool list and its order stable, since adding or removing tools invalidates the cache.
- Caches have a TTL and vendors differ (explicit breakpoints or automatic); compaction changes the prefix, so balance compaction frequency against hit rate.

Q7: ***What fallback strategies should be designed in an agent system?***

- **Have a fallback path at every layer, model, tool, task, quality and budget, and log and monitor every fallback.**
- Model layer: switch to a backup model or vendor on timeouts, rate limits or outages.
- Tool layer: retry first, then have the model use an alternative tool or ask the user.
- Task layer: if it can't finish, return partial results and state what is missing, or hand off to a human; quality layer: regenerate on failed validation or return a safe default reply.
- Budget layer: stop and report progress when over budget instead of running forever.

Q8: ***In system design, how do you design a guardrails system?***

- **Design in layers, input, model, action and output, each with its own checks, and balance strictness against false refusals.**
- Input: authentication, rate limiting, jailbreak/injection detection, topic scope, PII detection; model: a clear system prompt and behavior rules.
- Action: argument validation, permission checks, human approval for high-risk actions, sandboxed execution; output: harmful content, policy compliance, grounding and format checks.
- Run cheap fast models in parallel and reserve heavier checks for high-risk cases; guardrails also need eval sets to measure false positives and negatives.

Q9: ***How do you choose models for an agent?***

- **Trade off task difficulty, latency and cost: use the strongest model (with thinking) for orchestration, planning and complex reasoning, and small fast models for many simple subtasks.**
- Focus on tool calling, long context and instruction following rather than general leaderboards.
- Compare success rate, cost and latency on your own eval set.
- Abstract the model interface so models stay swappable for upgrades and downgrades.

Q10: ***What must be considered for long-running (multi-hour) agent tasks?***

- **Most important is state persistence and checkpoints, so an interruption resumes instead of restarting; next come context management and progress visibility.**
- Context: compaction, external notes and subagents to avoid overflow and quality decay.
- Run asynchronously in the background, notify on completion, and let users inspect and intervene midway.
- Set budgets and timeouts to prevent runaway runs; keep the sandbox reproducible and detect changes in external dependencies.

Q11: ***How do you design an agent's memory system to support personalization?***

- **Separate in-session memory (the context) from cross-session memory (preferences, historical facts): extract what's worth remembering at session end or via a memory tool, store it per user, and retrieve relevant memories at the start of a new session or let the agent query on demand.**
- Handle conflicting and stale memories.
- Let users view, edit and delete their memories.
- Treat memory content as data, not instructions, to prevent malicious instructions being injected.

Q12: ***How do you design the evaluation and iteration process for an agent?***

- **Build an offline eval set that runs automatically on every change, trace all trajectories online, and add new failures back into the eval set to close the "find → fix → regression test" loop.**
- The eval set includes real tasks, edge cases and adversarial examples, judged by deterministic checks, LLM-as-judge and human review.
- Monitor success rate, latency, cost and user feedback online.
- Sample failed trajectories to locate whether the problem is the prompt, tools, retrieval or model capability; validate changes online with A/B tests or staged rollouts.

Q13: ***How do you design an agent that can operate a browser/computer (computer use)?***

- **Perceive the page via screenshots plus the accessibility tree/DOM, output actions like click, type and scroll, run in an isolated VM or browser, and require human confirmation for key actions; prefer APIs whenever they exist.**
- Challenges: many steps, high latency, dynamic pages, coordinate errors, prompt injection from web pages.
- Safety: don't log into sensitive accounts or use dedicated ones, restrict accessible domains, and confirm payments and form submissions.

Q14: ***How are multi-tenancy and scalability handled in agent systems?***

- **Fully isolate each tenant's data, memory, indexes and sandboxes, apply per-tenant rate limits and quotas, and keep the orchestration service stateless for horizontal scaling.**
- Store session state in an external database or cache.
- Run long tasks asynchronously through a task queue instead of holding request connections.
- Handle model API rate limits with retry and backoff and multi-region/multi-vendor routing.
