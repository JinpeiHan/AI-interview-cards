# Tool Calling

Q1: ***What is function calling / tool calling? How does it work?***

- Tool calling is the ability of an LLM to **request a call to an external function in a structured format (usually JSON)**.
- Flow:
    1. The developer provides tool definitions in the request (name, description, JSON Schema of the parameters);
    2. The model decides whether a tool is needed; if so, it outputs a **tool call block** (tool name + arguments);
    3. The **application** parses and executes the function and sends the result back to the model as a **tool result message**;
    4. The model continues reasoning based on the result, possibly calling tools again or giving a final answer.
- Key point: **the model only produces the "intent to call" and does not execute code**; execution, permissions and security are the application's responsibility.

Q2: ***How do models learn to call tools?***

- Mainly through **post-training**: supervised fine-tuning (SFT) on conversation data containing tool definitions, tool calls and tool results, followed by reinforcement learning (RL) in real or simulated environments to optimize multi-step tool use.
- At inference time, tool definitions are injected into the system prompt/special template; the model outputs a structured call with special tokens, which the server parses into a tool call object in the API.
- Some APIs support **constrained decoding**, which guarantees that the output arguments strictly conform to the JSON Schema.

Q3: ***How should a good tool schema be designed?***

- **Name**: clear, verb-first, namespaced, e.g. `github_create_issue`, avoiding ambiguity;
- **Description**: explain what the tool does, **when to use/not use it**, what the parameters mean, what it returns, and its limitations, with examples where necessary;
- **Parameters**:
    - Use explicit types, `enum` and `required` fields to reduce free text;
    - Make parameter names self-explanatory (`user_id` rather than `id`);
    - Avoid making the model do complex format conversions (e.g. hand-writing long escaped strings);
- **Granularity**: design tools around **user intents/tasks** rather than simply wrapping every low-level API endpoint one by one; frequently combined operations can be merged into one tool;
- **Return value**: return concise, high-signal information meaningful to the model (e.g. names instead of internal UUIDs), and support pagination, filtering and truncation.

Q4: ***What is MCP (Model Context Protocol)? What problem does it solve?***

- MCP is an **open protocol** introduced by Anthropic in late 2024 to standardize connections between LLM applications and external tools and data sources; it is often likened to "a USB-C port for AI applications".
- The problem it solves: without a common standard, connecting $M$ applications to $N$ tools requires $M \times N$ custom integrations; with MCP, each application implements a client once and each tool implements a server once, reducing integration cost to $M + N$.
- It is now widely supported by major model vendors, IDEs and agent frameworks.

Q5: ***What are MCP's architecture and core concepts?***

- **Architecture**:
    - **Host**: the AI application the user interacts with, such as a chat client, IDE or agent;
    - **Client**: the component inside the host that maintains a one-to-one connection with a given server;
    - **Server**: a program that exposes capabilities, such as an MCP server for GitHub, a database or the file system.
- **Communication**: based on **JSON-RPC 2.0**; transports include local **stdio** and remote **Streamable HTTP** (remote scenarios usually use OAuth for authorization).
- **Primitives provided by servers**:
    - **Tools**: functions the model can call (model-controlled);
    - **Resources**: readable data/context, such as file contents (application-controlled);
    - **Prompts**: reusable prompt templates (user-controlled).
- **Capabilities provided by clients**: e.g. **Sampling** (lets the server request a generation from the host's model), **Elicitation** (requests additional information from the user), and Roots (tells the server what scope it may access).

Q6: ***How does MCP relate to ordinary function calling?***

- Function calling is a **model-level** capability: how the model outputs structured tool calls;
- MCP is an **application-integration-level** protocol: how tools are discovered, described, invoked and transported.
- At runtime, the host fetches the server's tool list via MCP (`tools/list`) and converts it into tool definitions for the model API; after the model emits a tool call, the host forwards it to the server via MCP (`tools/call`) for execution. **The two are complementary.**

Q7: ***What are the security risks of using MCP?***

- **Tool poisoning**: a malicious server embeds hidden instructions in tool descriptions to influence model behavior;
- **Prompt injection**: content returned by tools (web pages, emails, issues) contains malicious instructions;
- **Excessive permissions**: a server holding high-privilege credentials is induced to perform destructive actions or leak data;
- **Rug pull**: a server changes its tool definitions after the user has approved it;
- **Data exfiltration**: combining a tool that reads private data with a tool that can send data externally.
- Defenses: only use servers from trusted sources, least privilege, human confirmation for sensitive operations, alerts on tool-definition changes, isolation and review of outputs.

Q8: ***What is structured output? What are the ways to implement it?***

- Making the model output results that **strictly conform to a predefined format** (usually a JSON Schema), so programs can parse them.
- Implementation approaches:
    - **Prompt constraints**: give the format and examples in the prompt; simplest, but correctness is not guaranteed;
    - **JSON mode**: guarantees valid JSON, but not conformance to a specific schema;
    - **Schema-constrained decoding (strict mode)**: masks invalid tokens during decoding using a grammar/finite-state machine, **guaranteeing schema conformance**;
    - **Piggybacking on tool calling**: define a "submit result" tool and have the model output structured data by calling it;
    - **Post-hoc validation**: validate with Pydantic or similar; on failure, feed the error back to the model and retry.
- Note: strong constraints may slightly hurt reasoning quality; you can let the model reason first and then output the structured result.

Q9: ***How should tool call errors be handled?***

- **Distinguish error types**:
    - Argument errors/validation failures: return a **clear, actionable error message** to the model as the tool result (e.g. "parameter date must be in YYYY-MM-DD format") so the model can fix it itself;
    - Transient errors (timeouts, rate limits, 5xx): do **exponential backoff retries** at the runtime level, without the model needing to know;
    - Permanent errors (no permission, resource not found): tell the model explicitly so it can change approach or ask the user for help.
- Mark errors as errors (e.g. `is_error`) instead of letting the runtime crash;
- Set a **maximum number of retries** to prevent the model from repeatedly retrying with the same arguments;
- For operations with side effects, ensure **idempotency** on retry (e.g. using idempotency keys) to avoid duplicate orders or duplicate emails.

Q10: ***What are parallel tool calls? What should you watch out for?***

- The model outputs **multiple independent tool calls** in one response; the runtime can execute them concurrently and return all results together, **reducing round trips and latency**.
- Typical scenarios: querying the weather in several cities at once, reading several files at once, issuing several searches at once.
- Caveats:
    - Only **mutually independent** calls can run in parallel; dependent calls must be serial;
    - Each result must be matched to its call via the **tool_call_id**;
    - Parallel write operations with side effects raise race-condition and ordering issues; parallel calls can be disabled in the API if necessary;
    - The prompt can encourage the model to make parallel calls for independent operations.

Q11: ***When there are very many tools (dozens to thousands), how do you get the model to choose correctly?***

- Problem: tool definitions consume a large number of context tokens, and too many similar tools reduce selection accuracy.
- Methods:
    - **Tool retrieval/dynamic loading**: select relevant tools for the current task via vector retrieval or keyword matching, and put only those into the context (e.g. provide a "tool search" meta-tool that loads full definitions on demand);
    - **Hierarchical routing**: first choose a category/subagent, then choose a tool within the category;
    - **Merging and pruning**: merge tools with overlapping functions and remove rarely used ones;
    - **Namespacing**: use prefixes to distinguish services, e.g. `slack_`, `jira_`;
    - **Code execution instead**: let the agent write code that calls APIs (exposing tools as a code library) rather than putting all tool definitions into the context.
- Continuously measure tool-selection accuracy with an evaluation set.

Q12: ***What does the `tool_choice` parameter do?***

- It controls whether/how the model uses tools; common values:
    - **auto**: the model decides whether to call a tool (default);
    - **required/any**: the model must call at least one tool;
    - **Specific tool**: force a call to a particular tool (often used for structured extraction);
    - **none**: tool calls are disallowed.
- Note: for some models that support extended thinking, forcing a tool call may be incompatible with thinking mode; check the specific API documentation.

Q13: ***What if a tool result is too long?***

- **Truncate, paginate and filter** at the tool level, and tell the model the result was truncated and how to get more (e.g. provide an `offset` parameter);
- Provide a "response verbosity" parameter (e.g. `concise`/`detailed`) so the model can choose as needed;
- Write large results to a file/store first, return only a summary and a reference path, and let the model read on demand;
- Use a small model to summarize the result before returning it;
- Clear old tool results from the history that are no longer needed (tool result clearing).

Q14: ***How do you evaluate and improve tool design?***

- Build **evaluation tasks close to real usage**, and measure task success rate, number of tool calls, error rate and token consumption;
- Read the agent's **full trajectories** to see where the model misunderstood tools (wrong tool, wrong arguments, redundant calls);
- Revise tool descriptions, parameters and return formats based on these observations, and compare iterations;
- You can even have a model analyze failed trajectories and suggest improvements to tool descriptions.

Q15: ***How should "human approval" in tool calling be designed?***

- Tier by risk: read-only operations run automatically; **irreversible or high-impact operations** (deleting, transferring money, sending emails, deploying) require user confirmation;
- The confirmation UI should show the **specific arguments** and impact, not a vague "allow or not";
- Support permission policies such as "allow once/always allow this kind of operation", balancing security and user experience;
- When rejected, return the rejection reason to the model as the tool result so it can adjust its approach.

Q16: ***What are the pros and cons of tool calling vs. code execution (letting the model write and run code)?***

- **Tool calling**:
    - Pros: explicit interfaces, easy to apply permissions and validation, good security;
    - Cons: every call needs a model round trip, complex data processing requires many calls, and all intermediate results pass through the context.
- **Code execution (code as action)**:
    - Pros: highly expressive — a single piece of code can combine multiple calls, loops and filtering, and **intermediate data stays in the execution environment, saving context tokens**;
    - Cons: requires a secure **sandbox**, and behavior is harder to audit and restrict.
- In practice they are often combined: tools for routine operations, and a code execution environment for complex data processing.
