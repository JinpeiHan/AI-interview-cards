# Tool Calling

Q1: ***What is function calling / tool calling? How does it work?***

- **Tool calling is the ability of an LLM to request an external function call in a structured format (usually JSON): the model only outputs the "intent to call", and the application does the actual execution.**
- The developer provides tool definitions (name, description, JSON Schema for the parameters).
- When needed, the model outputs a tool name and arguments; the application runs the function and sends the result back as a tool result message.
- The model continues from the result, possibly calling more tools or giving a final answer; execution, permissions and security all live on the application side.

Q2: ***How do models learn to call tools?***

- **Mainly through post-training: SFT on conversations containing tool definitions, calls and results, then reinforcement learning in real or simulated environments to optimize multi-step tool use.**
- At inference, tool definitions are injected into the system prompt or a special template; the model emits a call with special tokens, which the server parses into an API tool call object.
- Some APIs support constrained decoding to guarantee the arguments strictly match the JSON Schema.

Q3: ***How should a good tool schema be designed?***

- **Use a clear name, a description that says when to use and not use the tool, explicitly typed parameters, and design around user tasks rather than low-level API endpoints.**
- Name: verb first, namespaced, e.g. `github_create_issue`.
- Parameters: explicit types, `enum`, `required`, self-explanatory names (`user_id` not `id`), and no complex format conversions for the model.
- Granularity: frequently combined operations can be merged into one tool.
- Return values: concise and high-signal (names rather than internal UUIDs), with pagination, filtering and truncation.

Q4: ***What is MCP (Model Context Protocol)? What problem does it solve?***

- **MCP is an open protocol introduced by Anthropic in late 2024 that standardizes how LLM applications connect to external tools and data sources, often called "USB-C for AI apps".**
- Without a standard, M apps connecting to N tools need M×N custom integrations; with MCP, each app implements a client once and each tool a server once, so only M+N.
- It is now widely supported by major model vendors, IDEs and agent frameworks.

Q5: ***What are MCP's architecture and core concepts?***

- **MCP uses a Host-Client-Server architecture: the Host is the AI app, each Client inside it holds a one-to-one connection to a Server, and Servers expose tools and data.**
- Communication is JSON-RPC 2.0, over local stdio or remote Streamable HTTP (remote usually with OAuth).
- Servers offer three primitives: Tools (model-controlled functions), Resources (app-controlled data), Prompts (user-controlled templates).
- Clients also offer capabilities: Sampling (Server uses the Host's model), Elicitation (ask the user for more input), Roots (declare accessible scope).

Q6: ***How does MCP relate to ordinary function calling?***

- **They are complementary: function calling is the model-level ability to output tool calls, while MCP is the application-level protocol for how tools are discovered, described, invoked and transported.**
- At runtime the Host gets the Server's tools via `tools/list` and converts them into model API tool definitions.
- When the model emits a tool call, the Host forwards it to the Server via `tools/call`.

Q7: ***What are the security risks of using MCP?***

- **The main risk is that both third-party Servers and tool outputs can manipulate the model, while Servers often hold high-privilege credentials.**
- Tool poisoning: a malicious Server hides instructions in tool descriptions; prompt injection: tool results such as web pages or emails contain malicious instructions.
- Over-privilege leading to destructive actions; rug pull: a Server changes tool definitions after the user approved them.
- Data exfiltration: combining a tool that reads private data with one that can send data out.
- Defenses: only trusted Servers, least privilege, human confirmation for sensitive actions, alerts on tool-definition changes.

Q8: ***What is structured output? What are the ways to implement it?***

- **It makes the model output results that strictly follow a predefined format (usually a JSON Schema); the most reliable method is constraining decoding so the output is guaranteed to match.**
- Prompt constraints: give the format and examples; simplest but no guarantee. JSON mode: guarantees valid JSON but not schema conformance.
- Schema-constrained decoding (strict mode): masks invalid tokens during decoding, guaranteeing schema conformance.
- Borrow tool calling (a "submit result" tool), or validate afterwards with Pydantic and feed errors back for a retry.
- Hard constraints can slightly hurt reasoning, so let the model reason first, then output the structured result.

Q9: ***How should tool call errors be handled?***

- **Handle each error type differently, and return errors to the model as tool results instead of crashing the runtime.**
- Argument errors: return a clear, actionable message (e.g. "date must be YYYY-MM-DD") so the model can fix it.
- Transient errors (timeouts, rate limits, 5xx): retry with exponential backoff in the runtime, invisibly to the model.
- Permanent errors (no permission, resource missing): tell the model clearly so it changes approach or asks the user.
- Cap the number of retries; make side-effecting operations idempotent so retries don't double-order or double-email.

Q10: ***What are parallel tool calls? What should you watch out for?***

- **The model outputs several independent tool calls in one reply; the runtime runs them concurrently and returns all results together, cutting round trips and latency.**
- Typical cases: weather for several cities, reading several files, firing several searches.
- Only calls with no dependencies can run in parallel, and each result must be matched to its call by `tool_call_id`.
- Parallel writes with side effects risk race conditions; disable parallel calls in the API if needed.

Q11: ***When there are very many tools (dozens to thousands), how do you get the model to choose correctly?***

- **The key is not to put every tool definition in the context, but to retrieve and load relevant tools dynamically per task.**
- Tool retrieval / dynamic loading: select tools by vector or keyword search, or provide a "search tools" meta-tool that loads definitions on demand.
- Hierarchical routing: pick a category or subagent first, then a tool within it.
- Merge overlapping tools, remove rarely used ones, and namespace by service (`slack_`, `jira_`).
- Agents can also write code against APIs instead of loading every tool definition; track selection accuracy with an eval set.

Q12: ***What does the `tool_choice` parameter do?***

- **It controls whether and how the model uses tools: auto (model decides, the default), required/any (must call at least one), a specific tool (forced, common for structured extraction), none (no tool calls).**
- On some models with extended thinking, forcing a tool call may be incompatible with thinking mode; check the API docs.

Q13: ***What if a tool result is too long?***

- **Truncate, paginate and filter at the tool level, and tell the model the result was truncated and how to get more.**
- Offer a verbosity parameter (`concise`/`detailed`) so the model chooses as needed.
- Write large results to a file and return only a summary and path for on-demand reading, or summarize with a small model first.
- Clear old tool results from history once they are no longer needed.

Q14: ***How do you evaluate and improve tool design?***

- **Evaluate on tasks close to real usage, read full trajectories to find where the model misunderstands tools, then revise descriptions, parameters and return formats and compare iterations.**
- Metrics: task success rate, number of tool calls, error rate, token usage.
- Typical issues: wrong tool chosen, bad arguments, redundant calls.
- You can also have the model analyze failed trajectories and suggest better tool descriptions.

Q15: ***How should "human approval" in tool calling be designed?***

- **Tier by risk: read-only operations run automatically, while irreversible or high-impact operations (delete, transfer money, send email, deploy) require user confirmation.**
- The confirmation UI shows the concrete arguments and impact, not a vague "allow?".
- Support policies like "allow once / always allow" to balance safety and UX.
- On rejection, return the reason as a tool result so the model can adjust.

Q16: ***What are the pros and cons of tool calling vs. code execution (letting the model write and run code)?***

- **Tool calling has clear interfaces, easy permission checks and validation, and is safer; code execution is more expressive and saves tokens by keeping intermediate data in the execution environment, but needs a sandbox and is harder to audit.**
- Tool calling's downside: one model round trip per call, many calls for complex data processing, and all intermediate results pass through the context.
- Code execution can combine multiple calls, loops and filtering in one snippet.
- In practice combine them: tools for routine operations, code execution for complex data processing.
