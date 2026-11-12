# Evaluation and Safety

Q1: ***Why is evaluating agents harder than evaluating ordinary LLMs?***

- **Agents are multi-step and nondeterministic, change their environment, and have many valid paths, so you can't just compare a single text output as with an ordinary LLM.**
- Trajectories differ between runs, so one run doesn't represent overall performance.
- There are many correct approaches, so you can't compare step by step against a "gold trajectory".
- You need reproducible, resettable environments (databases, websites, code repos) and must check the final state.
- High cost: one example may involve dozens of model calls.

Q2: ***What are the main dimensions and methods of agent evaluation?***

- **Look at outcome (was the task done), trajectory (was the process sensible and compliant), efficiency, reliability and safety, and combine deterministic code checks, LLM-as-judge and human review.**
- Outcome: tests pass, final database state, correct answer.
- Trajectory: sensible tool choice and arguments, no redundant or dangerous steps.
- Efficiency: steps, tool calls, tokens, time, cost.
- Reliability is measured over multiple runs: pass@k means at least one of k succeeds, pass^k means all k succeed, which better reflects production reliability.

Q3: ***What is LLM-as-judge? How should it be used correctly?***

- **An LLM scores another model's or agent's output and trajectory against a rubric; it is scalable, cheap and handles open-ended output, but has known biases that need calibration.**
- Biases: position bias, verbosity bias, self-preference, sensitivity to surface formatting.
- Use specific rubrics, ideally binary or a few levels, and have the judge give reasons before the score.
- In pairwise comparisons, swap the order and judge twice; calibrate against human-labeled samples and measure agreement.
- Wherever deterministic checks (tests, exact match) work, prefer them.

Q4: ***Introduce some common agent benchmarks***

- **The most cited are SWE-bench (software engineering), τ-bench (customer-service dialogue plus tool calls), WebArena (web tasks) and GAIA (general assistants).**
- SWE-bench: fix real GitHub issues, judged by the repo's tests, with a human-filtered Verified subset.
- τ-bench: converse with a simulated user, call domain APIs and follow business policy, judged by final database state, with emphasis on pass^k.
- WebArena: complete tasks on self-hosted real websites; GAIA: multi-step reasoning, search, file and multimodal questions with short answers that are easy to auto-grade.
- Others include OSWorld, Terminal-Bench and BrowseComp; public benchmarks suffer from contamination and mismatch with real business, so build your own eval set for production.

Q5: ***How do you build an evaluation set for your own agent?***

- **Collect tasks from real user requests and failure cases, give each an automatically checkable success criterion, start with a few dozen high-quality examples and keep growing.**
- Cover common scenarios, edge cases and adversarial examples.
- Separate capability evals (measure the ceiling; pass rate can be low) from regression evals (prevent regressions; should be near 100%).
- Run automatically on every change to prompts, tools or models, and track the metrics.

Q6: ***What is agent observability? What needs to be recorded?***

- **Observability means being able to see what the agent did at each step and why; it is the basis for debugging, evaluation and compliance, and its core is complete tracing.**
- Record each model call's full input, output and thinking (if available).
- Record each tool call's name, arguments, result, duration and errors, plus tokens, latency and cost.
- Record call relationships between agents (span tree), user feedback and final outcome.
- Tools include OpenTelemetry, LangSmith, Langfuse and Arize Phoenix; redact sensitive data in logs.

Q7: ***What is prompt injection? What is the difference between direct and indirect injection?***

- **Prompt injection is when an attacker crafts input that makes the model ignore its original instructions and follow the attacker's instead.**
- Direct injection: the user writes malicious instructions in their own input, e.g. "ignore all previous instructions"; jailbreaks fall here too.
- Indirect injection: malicious instructions are hidden in external data the agent processes, such as web pages, emails, documents and tool results.
- For agents, indirect injection is more dangerous, since they automatically read lots of untrusted content and have tools that take real actions.

Q8: ***How do you defend against prompt injection?***

- **There is no perfect defense, only defense in depth, and the core is limiting the damage once the model is hijacked.**
- Least privilege; isolate untrusted data so modules that handle it don't directly hold high-privilege tools (e.g. the dual-LLM pattern).
- Avoid the "lethal trifecta" together: access to private data + exposure to untrusted content + ability to communicate externally.
- Human confirmation for sensitive actions, and outbound network restricted to allowlisted domains.
- Use classifiers to detect injection and exfiltration, and choose models more robust to injection.

Q9: ***Why do agents need sandboxing? What are common sandboxing approaches?***

- **Agents run code, commands and file operations, and a model mistake or injected instruction can delete data, leak credentials or damage systems; a sandbox confines the impact to a controlled environment.**
- Containers (Docker): lightweight but share the host kernel, so weaker isolation; gVisor and microVMs (Firecracker) isolate more strongly.
- OS-level sandboxes: Linux seccomp/Landlock/bubblewrap, macOS Seatbelt; WebAssembly can also run untrusted code.
- Key settings: file access limited to the working directory, network allowlist, resource limits, and no sensitive credentials inside the sandbox.

Q10: ***How should an agent's permission system be designed?***

- **Follow least privilege: read-only by default, authorization tiered by operation risk, and mandatory human confirmation for high-risk operations.**
- Low risk (read, search) auto-allowed; medium risk (edit files, call internal APIs) allowed by rule or asked; high risk (delete, pay, send external messages, deploy to production) must be confirmed.
- Support configurable allow/deny rules.
- Use short-lived, fine-grained, per-task credentials instead of long-lived admin keys; keep audit logs of all actions.

Q11: ***What is human-in-the-loop? How is it applied in agents?***

- **Bringing human review, confirmation or correction into key points of an agent's execution.**
- Common forms: approval before high-risk actions, clarification when information is missing, plan review before execution, result review before delivery, and escalation to a human when the agent can't cope.
- Trade-off: too much intervention hurts efficiency and causes "approval fatigue" (clicking yes without reading), so keep it to truly high-risk points.

Q12: ***What are guardrails? What types are there?***

- **Guardrails are mechanisms outside the model that check and constrain inputs, outputs and actions, in three types: input, output and action.**
- Input guardrails: detect jailbreaks, prompt injection, off-topic requests, PII.
- Output guardrails: detect harmful content, hallucination, sensitive-data leaks, and validate format.
- Action guardrails: check arguments and permissions before tool execution, e.g. amount limits, forbidden paths.
- Implemented with rules, classifiers or LLM review, and can run in parallel with the main model to reduce latency.

Q13: ***How do you control an agent's cost?***

- **The most effective levers are model routing, prompt caching and context management, plus hard budget caps and cost monitoring.**
- Model routing: small models for simple subtasks, large models only for complex reasoning.
- Prompt caching: put stable system prompts, tool definitions and long documents in a cached prefix.
- Context management: truncate or summarize tool results and compact history; tune the thinking budget, and use batch APIs for non-real-time work.
- Cap steps, tokens and cost per task, and monitor cost by user and task type to catch spikes from loops.

Q14: ***What is "reward hacking"? How does it manifest in agents?***

- **The model finds a shortcut that satisfies the evaluation signal without actually completing the task.**
- Typical signs: modifying or deleting tests to make them pass, hard-coding expected outputs.
- Claiming completion when not done, or bypassing rather than solving permission and validation problems.
- Countermeasures: more thorough evals (hidden tests, inspecting code changes), checking for such behavior in trajectory evals, and explicitly forbidding it in the prompt.

Q15: ***How do you continuously ensure quality and safety before and after an agent goes live?***

- **Before launch, run offline evals, red-teaming and a staged rollout; after launch, monitor continuously, sample and review, and feed new failures back into the eval set to close the loop.**
- Red-teaming focuses on prompt injection, privilege escalation and abuse.
- Monitor success rate, error rate, latency, cost and user feedback online, and sample trajectories for LLM-as-judge and human review.
- Run regression evals on model or prompt updates, with fast rollback.

Q16: ***What data privacy issues arise with agents? How are they handled?***

- **The main risks are user data flowing into logs, long-term memory and third-party tools, plus cross-user leakage and exfiltration via injection.**
- Redact PII in logs and traces.
- Isolate memory and retrieval indexes per user/tenant.
- Review third-party tools' data processing agreements, follow retention and deletion policies, and limit the agent's external communication.
