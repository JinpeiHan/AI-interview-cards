# RAG and Memory

Q1: ***What is RAG (retrieval-augmented generation)? Why is it needed?***

- RAG (Retrieval-Augmented Generation) **retrieves relevant content from an external knowledge base** before generation, places the retrieved results into the prompt, and lets the LLM answer based on that content.
- Problems it solves:
    - The model's knowledge has a **cutoff date** and cannot access the latest information;
    - The model does not know **private/domain data**;
    - It reduces **hallucinations** and can provide **source citations**;
    - Compared with fine-tuning, updating knowledge only requires updating the index — cheap, controllable and deletable.

Q2: ***Describe a typical RAG pipeline***

- **Offline indexing**:
    1. Load and parse documents (PDF, HTML, tables, etc.);
    2. Clean and **chunk** them;
    3. Encode each chunk into a vector with an embedding model;
    4. Store the vectors in a vector database (a keyword inverted index and metadata can be built at the same time).
- **Online querying (retrieval & generation)**:
    1. Query rewriting/expansion;
    2. Retrieve the top-K candidates (vector retrieval + keyword retrieval);
    3. **Rerank** and filter;
    4. Concatenate the retrieved chunks with the question into the prompt, and have the LLM generate an answer with citations.

Q3: ***What are common chunking strategies? How do you choose the chunk size?***

- Strategies:
    - **Fixed-length chunking**: split by token count, with **overlap** to avoid cutting through meaning;
    - **Recursive/structural chunking**: recursively split along natural boundaries such as paragraphs, headings and sentences;
    - **Semantic chunking**: determine boundaries from changes in the embedding similarity of adjacent sentences;
    - **Document-structure-aware**: split by structure such as Markdown headings, code functions or table rows;
    - **Contextual chunking**: use an LLM to generate, for each chunk, a prefix describing its position/meaning within the whole document, then embed and index it, mitigating the problem of chunks losing their context.
- Chunk size trade-offs:
    - **Too small**: incomplete meaning, lack of context;
    - **Too large**: introduces noise, dilutes the embedding, and occupies context.
- In practice, start from a few hundred tokens and tune with an evaluation set; you can also use **"retrieve small chunks, return large chunks"** (e.g. parent-document retrieval) to balance retrieval precision and contextual completeness.

Q4: ***What role does the embedding model play in RAG? How do you choose one?***

- An embedding model maps text to dense vectors so that **semantically similar texts are close in vector space**, typically measured by cosine similarity $\cos(q,d)=\frac{q\cdot d}{\|q\|\|d\|}$.
- A **bi-encoder (two-tower)** architecture is usually used: queries and documents are encoded separately, and document vectors can be precomputed offline, making retrieval fast.
- Selection considerations: domain and language fit, maximum input length, vector dimension (storage and speed), multilingual ability, and retrieval evaluation results on your own data (not just public leaderboards);
- If necessary, **fine-tune the embedding model** on domain data.

Q5: ***How is vector retrieval accelerated? What is ANN?***

- Exact nearest-neighbor search has complexity $O(N)$, which is too slow at scale, so **approximate nearest neighbor (ANN)** algorithms are used, trading a small loss in recall for a large speedup.
- Common indexes:
    - **HNSW**: based on hierarchical navigable small-world graphs; fast with high recall, but memory-hungry;
    - **IVF**: cluster first, and at query time search only the nearest few clusters;
    - **PQ (product quantization)**: compresses vectors to save memory, often combined with IVF.
- Common tools: FAISS, Milvus, Qdrant, Weaviate, pgvector, Elasticsearch/OpenSearch, etc.

Q6: ***What is hybrid search? Why is it needed?***

- It combines **sparse retrieval (keywords, e.g. BM25)** with **dense retrieval (vectors)**.
- Reasons:
    - Vector retrieval is good at semantic matching and paraphrases, but weaker at exact matching of **proper nouns, IDs, code and rare words**;
    - BM25 is good at exact keyword matching but does not understand semantics.
- Fusion methods:
    - **Weighted sum** (requires score normalization);
    - **RRF (Reciprocal Rank Fusion)**: $score(d)=\sum_{i}\frac{1}{k+rank_i(d)}$, which depends only on ranks, needs no score calibration, and is simple and robust.

Q7: ***What is reranking? Why does it improve results?***

- First use a fast retriever to recall many candidates (e.g. top 50–100), then use a more fine-grained model to rescore them and select the top K to send to the LLM.
- A **cross-encoder** is commonly used: the query and document are **concatenated** and fed into the model together for full cross-attention, which judges relevance far more accurately than a bi-encoder, but is computationally expensive and only suitable for a small number of candidates.
- An LLM can also be used for reranking (listwise/pointwise).
- In essence it is a **two-stage "recall then precision ranking"**: the first stage optimizes recall, the second optimizes precision.

Q8: ***What query-side optimizations are there?***

- **Query rewriting**: rewrite colloquial, multi-turn questions containing references into self-contained retrieval queries;
- **Multi-query**: generate several differently phrased queries, retrieve for each and merge the results;
- **Query decomposition**: split a complex question into several sub-questions and retrieve for each;
- **HyDE**: have the LLM generate a hypothetical answer first and retrieve with its embedding;
- **Metadata filtering**: filter by fields such as time, source and permissions;
- **Step-back prompting**: first retrieve more abstract background knowledge.

Q9: ***How do you evaluate a RAG system?***

- **Retrieval level**: Recall@K, Precision@K, MRR, nDCG — measuring whether relevant documents are retrieved and ranked near the top;
- **Generation level**:
    - **Faithfulness/groundedness**: whether the answer is supported by the retrieved content, with no hallucination;
    - **Answer relevance**: whether it answers the question;
    - **Correctness**: compared against a reference answer;
    - **Citation accuracy**: whether citations actually support the corresponding claims;
- Methods: build a labeled QA evaluation set and combine human evaluation with LLM-as-judge; frameworks such as Ragas implement common metrics.
- **Diagnose separately** whether a failure is in retrieval or in generation.

Q10: ***What is agentic RAG? How does it differ from traditional RAG?***

- **Traditional RAG**: a fixed pipeline that retrieves once per query and uses the results directly for generation.
- **Agentic RAG**: retrieval becomes a **tool** of the agent, and the model decides:
    - Whether retrieval is needed, what to retrieve, and which data source to use (vector store, SQL, web search, grep over a codebase, etc.);
    - Whether the information is sufficient based on the results, and if not, **rewrites the query and does multi-hop retrieval**;
    - How to verify and synthesize the retrieved results.
- Pros: handles multi-hop, complex questions and is highly adaptive; cons: higher latency and cost, and less predictable behavior.
- In scenarios such as code, agents often use **"just-in-time retrieval" like grep/file browsing** to replace or complement a prebuilt vector index.

Q11: ***With long-context models, is RAG still necessary?***

- Yes, but the two are **complementary**:
    - Knowledge bases are often far larger than the context window;
    - Stuffing large amounts of content into the context is **costly and slow**;
    - With very long contexts, the model's ability to use information in the middle degrades (the **"Lost in the Middle"** phenomenon), and noise interferes with reasoning;
    - RAG makes **access control, real-time updates and citation tracing** easier.
- Long context lowers the precision requirement on retrieval: you can retrieve more and larger chunks and let the model filter them itself.

Q12: ***How should an agent's long-term memory be designed?***

- **Writing**: decide what to remember — user preferences, key facts, task conclusions, lessons learned; the agent can write actively via a "memory tool", or a background process can **extract and summarize** after the session ends.
- **Storage**: vector stores (semantic retrieval), key-value/relational databases (structured facts), knowledge graphs (entity relations), files (e.g. Markdown notes).
- **Retrieval**: score by a combination of relevance, recency and importance, and bring the results back into the context.
- **Updating and forgetting**: handle conflicts and stale information (new facts overwrite old ones), deduplicate, merge, and set expiration policies.
- **Privacy and security**: users can view and delete memories; prevent malicious instructions from being injected via memory and prevent cross-user leakage.

Q13: ***What is context window management? What are common strategies?***

- As the agent loop runs, conversation history and tool results keep accumulating and will eventually exceed the context window or degrade performance, so they must be actively managed.
- Common strategies:
    - **Sliding window**: keep only the last N turns; simple but loses early information;
    - **Summarization/compaction**: when the context approaches its limit, have the model summarize the history into a concise summary and continue working in a new context seeded with it;
    - **Tool result clearing**: delete or replace old tool results that have already been processed;
    - **External memory/notes**: the agent writes progress, plans and key findings to files (e.g. `NOTES.md`, a to-do list) and reads them back when needed;
    - **Subagents**: hand subtasks requiring heavy exploration to subagents that return only distilled results;
    - **On-demand retrieval**: keep only references (file paths, URLs) and load the content when needed.

Q14: ***What information should be kept during context compaction? What are the risks?***

- Keep: the user's original goal and constraints, key decisions made and the reasons for them, current progress and the next steps, unresolved issues/errors, and important file paths or identifiers.
- Discard: redundant raw tool output, intermediate attempts that have been resolved, duplicated content.
- Risks:
    - **Information loss**: the summary omits details that only later turn out to be important;
    - Over-compression causes the model to redo work it has already done;
- Mitigation: keep retrievable raw records (logs/files), explicitly list what must be kept in the compaction prompt, and tune the compaction strategy through evaluation.

Q15: ***What are common problems in RAG, and how are they solved?***

- **Nothing relevant is retrieved**: improve chunking, use hybrid search, rewrite queries, fine-tune embeddings;
- **Retrieved but ranked low**: add a reranker;
- **Retrieved but the model ignores/misuses it**: improve the prompt by requiring answers grounded in the context with citations; reduce noisy chunks;
- **Poor document parsing quality**: tables, images and multi-column PDFs need specialized parsing (layout analysis, OCR, multimodal models);
- **Knowledge conflicts/staleness**: add time metadata and prefer the latest version;
- **Permission leakage**: filter by user permissions at the retrieval stage (document-level ACLs).

Q16: ***What is GraphRAG? What scenarios is it suited for?***

- GraphRAG first uses an LLM to **extract entities and relations from documents to build a knowledge graph**, runs community detection and hierarchical summarization on the graph, and combines graph structure and summaries at retrieval time.
- Suitable scenarios: **global, cross-document synthesis questions** (e.g. "what are the main themes of these reports?") and multi-hop relational reasoning.
- Downsides: building the graph is expensive and updating it is complex; for simple fact lookups, traditional vector RAG is usually sufficient.
