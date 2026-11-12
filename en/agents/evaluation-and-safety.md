# Evaluation and Safety

Q1: ***Why is evaluating agents harder than evaluating ordinary LLMs?***

- **Multi-step and non-deterministic**: the trajectory for the same task can differ on every run, so the result of one run does not represent overall performance;
- **Diverse paths**: there may be many correct ways to reach the goal, so you cannot simply compare step by step against a "reference trajectory";
- **Environment dependence**: reproducible environments (databases, web pages, code repositories) and state resets are required;
- **Side effects**: agents modify the environment, so you must check the final state, not just the text output;
- **High cost**: each evaluation example may involve dozens of model calls.

Q2: ***What are the main dimensions and methods of agent evaluation?***

- **Outcome evaluation**: whether the task was completed successfully, e.g. whether unit tests pass, whether the final database state is correct, whether the answer is correct;
- **Trajectory evaluation**: inspect the execution process — whether tool selection was correct, whether arguments were reasonable, whether there were redundant or dangerous steps, whether policies were followed;
- **Efficiency**: number of steps, number of tool calls, token consumption, time taken, cost;
- **Reliability/consistency**: success rate over multiple runs, e.g. **pass@k** (at least one success in $k$ attempts) and **pass^k** (all $k$ attempts succeed, which better reflects reliability in production);
- **Safety**: whether unauthorized or harmful actions were taken, and whether the agent was affected by prompt injection.
- Methods: a combination of code-based deterministic checks, LLM-as-judge, and human evaluation.

Q3: ***What is LLM-as-judge? How should it be used correctly?***

- Using an LLM to evaluate another model's or agent's output/trajectory against a **rubric**.
- Pros: scalable, cheap, and able to evaluate open-ended output.
- Known biases: **position bias** (preferring the answer shown first), **verbosity bias** (preferring longer answers), **self-preference** (preferring outputs from the same model family), and being swayed by surface formatting.
- Best practices:
    - Provide a **specific, actionable rubric**, ideally binary or a small number of grades;
    - Have the judge give its reasoning before the score;
    - In pairwise comparison, **swap the order** and judge twice;
    - **Calibrate** the judge against human-labeled samples and compute its agreement rate with human judgments;
    - Wherever deterministic checks are possible (tests, exact match), prefer them.

Q4: ***Introduce some common agent benchmarks***

- **SWE-bench**: issues from real GitHub repositories; the agent must modify code to fix the issue, and success is judged by the repository's tests; there are subsets such as the human-filtered Verified set. It measures **real-world software engineering ability**.
- **τ-bench (tau-bench)**: simulates customer service scenarios (e.g. retail, airline); the agent must converse with a **simulated user** and call domain APIs while following business policies; it is evaluated on the final database state and emphasizes consistency across multiple runs (pass^k).
- **WebArena**: web tasks completed in self-hostable realistic website environments (e-commerce, forums, code hosting, maps, etc.), evaluating **web navigation and operation ability**.
- **GAIA**: questions for general AI assistants requiring multi-step reasoning, web search, file handling, multimodality and more; answers are short and unambiguous for automatic scoring; relatively easy for humans but hard for AI.
- Others: OSWorld (operating system/desktop operation), Terminal-Bench (command-line tasks), BrowseComp (finding hard-to-locate information), etc.
- Note: public benchmarks suffer from **data contamination** and mismatch with real business needs; in production you should build **your own evaluation set**.

Q5: ***How do you build an evaluation set for your own agent?***

- Collect tasks from **real user requests and failure cases**, covering common scenarios, edge cases and adversarial examples;
- Define clear **success criteria** for each task (an automatically checkable final state or reference answer);
- Start with a small number (dozens) of high-quality examples and keep expanding as new failure modes are found;
- Distinguish **capability evals** (measuring what the agent can do; pass rates can be low) from **regression evals** (preventing existing capabilities from degrading; should pass close to 100%);
- Run them automatically whenever prompts, tools or models change, and track metric changes.

Q6: ***What is agent observability? What needs to be recorded?***

- Observability means being able to **see what the agent did at each step and why**; it is the foundation for debugging, evaluation and compliance.
- What to record (**tracing**):
    - The full input (system prompt, context), output, and thinking content (if available) of each model call;
    - The name, arguments, result, duration and errors of each tool call;
    - Token usage, latency, cost;
    - Call relationships between agents in a multi-agent system (span tree structure);
    - User feedback and the final outcome.
- Tools: OpenTelemetry (and its GenAI semantic conventions), LangSmith, Langfuse, Arize Phoenix, etc.
- Remember to **redact sensitive data** in logs.

Q7: ***What is prompt injection? What is the difference between direct and indirect injection?***

- Prompt injection is when an attacker crafts input that makes the model **ignore its original instructions and follow the attacker's instructions**.
- **Direct injection**: the user writes malicious instructions in their own input (e.g. "Ignore all previous instructions..."; jailbreaks also fall into this category);
- **Indirect injection**: malicious instructions are hidden in **external data** the agent processes, such as web pages, emails, documents, code comments and tool results. The agent is hijacked after reading them.
- For agents, **indirect injection is more dangerous**: agents automatically read large amounts of untrusted content and have tools that perform real actions.

Q8: ***How do you defend against prompt injection?***

- There is currently **no perfect defense**; multiple layers of defense in depth are needed:
    - **Least privilege**: give the agent only the tools and permissions needed for the task;
    - **Isolate untrusted data**: clearly mark external content as data rather than instructions; modules that process untrusted content do not directly hold high-privilege tools (e.g. the dual-LLM pattern: a quarantined model processes untrusted data and returns only constrained, structured results);
    - **Avoid the "lethal trifecta" occurring together**: access to private data + exposure to untrusted content + the ability to communicate externally (an exfiltration channel); risk is highest when all three are present;
    - **Human confirmation for sensitive operations**;
    - **Input/output detection**: use classifiers to detect injection and data exfiltration;
    - **Egress network restrictions**: limit reachable domains to prevent data from being sent out;
    - Use models that have been adversarially trained to be more robust to injection.

Q9: ***Why do agents need sandboxing? What are common sandboxing approaches?***

- Agents may execute code, commands and file operations; model mistakes or injected instructions could **delete data, leak credentials or damage systems**. A sandbox confines the blast radius to a controlled environment.
- Common approaches:
    - **Containers** (Docker, etc.): lightweight, but share the kernel with the host, so isolation is relatively weaker;
    - **gVisor, microVMs (e.g. Firecracker)**: stronger isolation;
    - **OS-level sandboxes**: e.g. Linux seccomp/Landlock/bubblewrap and macOS Seatbelt, restricting file system and network access;
    - **Browser sandboxes/WebAssembly**: for running untrusted code.
- Key configuration: **file system isolation** (access only to the working directory), **network isolation** (only allow-listed domains), resource limits (CPU, memory, time), and no sensitive credentials inside the sandbox.

Q10: ***How should an agent's permission system be designed?***

- **Principle of least privilege**: read-only by default, with additional authorization required for writing and execution;
- **Tier operations by risk**:
    - Low risk (reading, searching): allowed automatically;
    - Medium risk (editing files, calling internal APIs): allowed by rule or with a prompt;
    - High risk (deleting, payments, sending external messages, production deployment): human confirmation required;
- **Configurable allow/deny rules** (e.g. allow specific commands, deny access to specific directories);
- **Short-lived, fine-grained credentials** (e.g. limited-scope tokens issued per task), rather than giving the agent long-lived admin keys;
- **Audit logs**: every action is traceable.

Q11: ***What is human-in-the-loop? How is it applied in agents?***

- Introducing **human review, confirmation or correction** at key points during agent execution.
- Ways to apply it:
    - **Approval**: pause before high-risk operations and wait for user approval;
    - **Clarification**: proactively ask the user when information is insufficient or the request is ambiguous;
    - **Plan review**: have the user confirm the plan before execution;
    - **Result review**: human checks before delivery (e.g. code review before merging);
    - **Escalation**: hand off to a human when the agent cannot handle the situation.
- Trade-off: too much intervention lowers efficiency and causes "approval fatigue" (users click approve without reading), so it should be concentrated on genuinely high-risk points.

Q12: ***What are guardrails? What types are there?***

- Guardrails are mechanisms outside the model that **check and constrain** inputs, outputs and actions.
- Types:
    - **Input guardrails**: detect jailbreaks, prompt injection, off-topic requests, sensitive information (PII);
    - **Output guardrails**: detect harmful content, hallucinations, leakage of sensitive information, and validate formats;
    - **Action guardrails**: check arguments and permissions before tool execution (e.g. amount limits, forbidden paths);
- Implementation: rules (regex, allow lists), classification models, LLM review; they can **run in parallel** with the main model to reduce latency.

Q13: ***How do you control an agent's cost?***

- **Model routing**: use small/fast models for simple subtasks and large models only for complex reasoning;
- **Prompt caching**: put the stable system prompt, tool definitions and long documents in the prefix and cache them, reducing the cost and latency of repeated input;
- **Context management**: truncate/summarize tool results and compact history to reduce input tokens per turn;
- **Budget limits**: set a maximum number of steps, tokens and cost per task, and stop or ask for confirmation when exceeded;
- **Control reasoning effort**: adjust the thinking budget as needed;
- **Batching**: use batch APIs for non-real-time tasks;
- **Monitoring**: track cost by user and task type, and detect anomalies (e.g. cost spikes caused by loops).

Q14: ***What is "reward hacking"? How does it manifest in agents?***

- The model finds a **shortcut** that satisfies the evaluation signal without actually completing the task.
- Manifestations in agents:
    - **Modifying or deleting tests**, or hard-coding expected outputs, to make tests pass;
    - Claiming the task is complete when it is not;
    - Circumventing rather than solving permission or validation problems.
- Countermeasures: design more comprehensive evaluations (keep hidden tests, inspect code changes), specifically check for such behavior in trajectory evaluation, and explicitly prohibit it in the prompt.

Q15: ***How do you continuously ensure quality and safety before and after an agent goes live?***

- Before launch: offline evaluation sets, red-teaming (prompt injection, privilege escalation, misuse), gradual rollout;
- After launch:
    - Online tracing and monitoring (success rate, error rate, latency, cost, user feedback);
    - Sample online trajectories and review them with LLM-as-judge and humans;
    - Add newly discovered failure cases to the evaluation set, forming a **closed loop**;
    - Run regression evals when the model or prompts are updated, with the ability to roll back quickly if necessary.

Q16: ***What data privacy issues arise with agents? How are they handled?***

- Risks: user data ending up in logs, long-term memory, or third-party tools/MCP servers; cross-user memory leakage; data exfiltration via prompt injection.
- Handling: **redact PII** in logs and traces; isolate memory and retrieval indexes per user/tenant; review data processing agreements for third-party tools; follow data retention and deletion policies; restrict the agent's ability to communicate externally.
