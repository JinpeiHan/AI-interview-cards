# RAG and Memory

Q1: ***What is RAG (retrieval-augmented generation)? Why is it needed?***

- **RAG retrieves relevant content from an external knowledge base before generation and puts it into the prompt, so the LLM answers based on that content.**
- It works around the model's knowledge cutoff and gives access to private or domain data the model doesn't know.
- It reduces hallucination and lets the answer cite sources.
- Compared with fine-tuning, updating knowledge just means updating the index: cheap, controllable and deletable.

Q2: ***Describe a typical RAG pipeline***

- **It has an offline indexing part and an online query part: offline, documents are chunked, embedded and stored; online, relevant chunks are retrieved, reranked and passed with the question to the LLM.**
- Offline: parse documents → clean and chunk → embed → store in a vector database (optionally with a keyword index and metadata).
- Online: rewrite the query → vector + keyword retrieval of top-K → rerank and filter → put into the prompt and generate an answer with citations.

Q3: ***What are common chunking strategies? How do you choose the chunk size?***

- **Common strategies are fixed-length with overlap, structural splitting by paragraphs or headings, semantic chunking, and contextual chunking where an LLM adds a context prefix to each chunk; size usually starts at a few hundred tokens and is tuned on an eval set.**
- Chunks that are too small lack meaning; too large adds noise, dilutes the embedding and uses up context.
- Structure-aware splitting (Markdown headings, code functions, table rows) usually beats hard cuts.
- "Retrieve small, return large" (parent-document retrieval) balances retrieval precision and context completeness.

Q4: ***What role does the embedding model play in RAG? How do you choose one?***

- **The embedding model maps text to dense vectors so that semantically similar texts are close in vector space, enabling semantic retrieval.**
- It is usually a bi-encoder: queries and documents are encoded separately, so document vectors can be precomputed offline and retrieval is fast.
- Choose by domain and language fit, max input length, vector dimension, multilingual ability, and above all retrieval evaluation on your own data rather than leaderboards.
- Fine-tune on domain data if needed.
- Key formula: $\cos(q,d)=\frac{q\cdot d}{\|q\|\|d\|}$

Q5: ***How is vector retrieval accelerated? What is ANN?***

- **Exact nearest-neighbor search compares against every vector, which is too slow at scale, so approximate nearest neighbor (ANN) trades a little recall for a large speedup.**
- HNSW: a hierarchical graph index; fast with high recall, but memory-heavy.
- IVF: cluster first and search only the nearest clusters; PQ: compress vectors to save memory, often combined with IVF.
- Common tools: FAISS, Milvus, Qdrant, Weaviate, pgvector, Elasticsearch, etc.

Q6: ***What is hybrid search? Why is it needed?***

- **Hybrid search combines keyword retrieval (e.g. BM25) with vector retrieval because they complement each other: vectors handle semantics and paraphrase, BM25 handles exact matches like proper nouns, IDs and code.**
- Fusion can be a weighted sum of normalized scores, or more commonly RRF.
- RRF depends only on ranks and needs no score calibration, so it is simple and robust.
- Key formula: $score(d)=\sum_{i}\frac{1}{k+rank_i(d)}$

Q7: ***What is reranking? Why does it improve results?***

- **A fast retriever first recalls many candidates (e.g. top 50-100), then a finer model rescores them to pick the top-K: a two-stage "recall then precision" setup.**
- Usually a cross-encoder: query and document are fed in together with full cross-attention, which is far more accurate than a bi-encoder.
- It is expensive, so it is only applied to a few candidates; an LLM can also be used as the reranker.

Q8: ***What query-side optimizations are there?***

- **The core idea is to turn the user's raw question into queries better suited for retrieval: rewrite, expand, decompose and filter.**
- Query rewriting: turn a colloquial, multi-turn question with references into a standalone query.
- Multi-query and decomposition: retrieve with several phrasings and merge, or split complex questions into sub-questions.
- HyDE: have the LLM write a hypothetical answer and retrieve with its embedding; step-back: retrieve more abstract background first.
- Metadata filtering by time, source or permissions.

Q9: ***How do you evaluate a RAG system?***

- **Evaluate the retrieval and generation layers separately, so you can tell whether a failure is "not retrieved" or "retrieved but used badly".**
- Retrieval: Recall@K, Precision@K, MRR, nDCG.
- Generation: faithfulness (supported by retrieved content), answer relevance, correctness, citation accuracy.
- Method: a labeled QA eval set, combining human review and LLM-as-judge (frameworks like Ragas implement common metrics).

Q10: ***What is agentic RAG? How does it differ from traditional RAG?***

- **Traditional RAG is a fixed pipeline that retrieves once and generates; agentic RAG makes retrieval a tool, and the model decides whether to retrieve, what, from which source, and whether the results are enough.**
- If not enough, it can rewrite queries, do multi-hop retrieval, and verify and synthesize results.
- Pros: handles multi-hop, complex questions; cons: higher latency and cost, less predictable.
- In settings like code, agents often use just-in-time retrieval such as grep and file browsing instead of, or alongside, a vector index.

Q11: ***With long-context models, is RAG still necessary?***

- **Yes; the two are complementary.**
- Knowledge bases are usually far larger than the context window, and stuffing everything in is expensive and slow.
- With very long contexts, models use middle information poorly (Lost in the Middle), and noise hurts reasoning.
- RAG makes access control, real-time updates and citation easy; long context lowers the precision bar for retrieval, so you can retrieve more and let the model filter.

Q12: ***How should an agent's long-term memory be designed?***

- **Design four stages well, writing, storage, retrieval, and updating/forgetting, while ensuring privacy and safety.**
- Writing: store user preferences, key facts, task conclusions and lessons; the agent can write via a memory tool, or a background process can extract and summarize after a session.
- Storage: vector store, key-value/relational database, knowledge graph or file notes, chosen by data shape.
- Retrieval: score by relevance, recency and importance; updating: resolve conflicts, dedupe and merge, set expiry.
- Safety: users can view and delete memories; prevent malicious instructions injected via memory and cross-user leakage.

Q13: ***What is context window management? What are common strategies?***

- **As the agent loop runs, history and tool results pile up until they overflow the window or degrade performance, so you must actively control what stays in context.**
- Sliding window: keep only the last N turns; simple but loses early information.
- Compaction: near the limit, summarize history and continue in a fresh context from the summary.
- Clear old tool results; write progress and plans to external notes (e.g. `NOTES.md`) and read them back when needed.
- Subagents isolate heavy exploration and return only distilled results; store only references (paths, URLs) and load on demand.

Q14: ***What information should be kept during context compaction? What are the risks?***

- **Keep the original goal and constraints, key decisions and their reasons, current progress and next steps, open issues, and important file paths and identifiers; redundant tool output and resolved intermediate attempts can go.**
- The main risk is information loss: the summary drops a detail that later turns out to matter, or the model redoes work already done.
- Mitigation: keep the raw logs/files for lookup, list must-keep items explicitly in the compaction prompt, and tune the strategy with evals.

Q15: ***What are common problems in RAG, and how are they solved?***

- **Common problems by stage: not retrieved, ranked too low, retrieved but misused, poor document parsing, outdated or conflicting knowledge, and permission leaks.**
- Not retrieved: better chunking, hybrid search, query rewriting, fine-tuned embeddings; ranked too low: add a reranker.
- Misused: prompt the model to answer from the context with citations, and reduce noisy chunks.
- Poor parsing: use layout analysis, OCR or multimodal models for tables, images and multi-column PDFs; outdated: add time metadata and prefer the latest version.
- Permissions: filter by user permissions (document-level ACL) at retrieval time.

Q16: ***What is GraphRAG? What scenarios is it suited for?***

- **GraphRAG uses an LLM to extract entities and relations from documents into a knowledge graph, runs community detection and hierarchical summarization, and retrieves using both the graph structure and the summaries.**
- It suits global, cross-document questions (e.g. "what are the main themes of these reports") and multi-hop relational reasoning.
- Downsides: building the graph is expensive and updates are complex; for simple fact lookup, ordinary vector RAG is usually enough.
