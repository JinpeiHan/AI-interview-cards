# ML System Design and MLOps

## Lifecycle and Training

Q1: ***Describe an end-to-end machine learning lifecycle. What does MLOps focus on?***

- Typical stages:
    1. **Problem definition**: business goal → ML objective → metrics (mapping between offline metrics and online KPIs), constraints (latency, cost, compliance)
    2. **Data**: collection, labeling, cleaning, validation, feature engineering, data versioning
    3. **Training and experimentation**: baseline → iteration, hyperparameter search, experiment tracking
    4. **Evaluation**: offline metrics, slice analysis, fairness, robustness
    5. **Deployment**: packaging, registration, gradual rollout
    6. **Monitoring and feedback**: data/model/business monitoring, collecting new labels → retraining
- The core of MLOps: make this loop **reproducible, automated, observable and reversible**; unlike DevOps, an ML system depends on three kinds of assets — **code, data and models** — and can "silently degrade" without any code change.
- Maturity (following Google's levels): Level 0 manual scripts → Level 1 automated training pipeline + continuous training → Level 2 CI/CD for the pipeline itself.
- Classic follow-up: Sculley et al. 2015 "Hidden Technical Debt in ML Systems" — model code is a tiny part of the system; glue code, pipeline jungles, configuration debt and feedback loops are the main risks.

Q2: ***What is the difference between offline (batch) training and online (incremental/streaming) training? How do you choose?***

- **Offline batch training**: periodically (daily/weekly) train from scratch or warm-start on full or sliding-window data.
    - Pros: stable, easy to reproduce, evaluate and roll back; cons: poor freshness, cannot react quickly to distribution changes
- **Online/incremental training**: continuously update on streaming data (e.g. FTRL, streaming embedding updates), taking effect within minutes or even seconds.
    - Pros: high freshness, suitable for fast-changing domains like news, ads and short video, and for cold-starting new IDs
    - Cons: vulnerable to dirty data/attacks, **catastrophic forgetting**, harder to evaluate and roll back, complex system (needs sample joining, real-time features)
- Common compromise: **daily full training + hourly/minute-level incremental updates**, with automatic validation gates (stop pushing if metrics look abnormal) and rollback-able snapshots.
- Decision criteria: speed of distribution change, proportion of new entities, whether the gain from freshness justifies the engineering cost — quantify the loss from **model staleness** with offline backtests of "model from N days ago vs latest model".

## Features and Data

Q3: ***What is a feature store? What problems does it solve?***

- A system that centrally manages feature **definition, computation, storage and serving** (e.g. Feast, Tecton, in-house systems at large companies).
- Components:
    - **Offline store** (warehouse/data lake, e.g. Hive, Parquet, BigQuery): used to generate training sets, supports historical lookback
    - **Online store** (low-latency KV, e.g. Redis, Cassandra, DynamoDB): used for online inference, millisecond reads of the latest values
    - **Feature registry**: feature metadata, owners, lineage, versions
    - Batch/streaming computation pipelines that write to both stores (materialization)
- Problems solved:
    - **Training-serving inconsistency** (one feature definition drives both sides)
    - **Feature reuse**, avoiding duplicate development across teams
    - **Point-in-time correct joins**
    - Feature discovery, governance and monitoring

Q4: ***What is training-serving skew? What causes it? How do you avoid it?***

- Definition: the data/processing logic the model sees during training differs from what it sees during online inference, leading to good offline metrics but poor online performance.
- Sources:
    - **Different code paths**: features computed offline in Python/SQL but reimplemented online in Java/C++, with subtle differences (missing-value handling, bucket boundaries, tokenizer version)
    - **Different data freshness**: training uses complete T+1 data, while online only has partial real-time data
    - **Feature leakage**: training uses information that is only available after the prediction time
    - **Distribution differences**: training samples are filtered/sampled while online sees all traffic; or the model itself influences the data (feedback loops)
    - Dependency versions and preprocessing parameters (normalization mean/variance) not packaged with the model
- Avoidance:
    - One feature codebase/feature store serving both sides; **package preprocessing into the model** (e.g. TF Transform, versioning the tokenizer together with the model)
    - **Log-and-wait**: log the features actually used at inference time and use them directly as training samples, eliminating skew at the root
    - **Consistency checks** before launch: for the same batch of requests, compare offline vs online computed features field by field and compare predictions
    - Monitor the difference between online feature distributions and training distributions

Q5: ***What is point-in-time correctness? How do you prevent feature leakage/label leakage?***

- When building training samples, each sample's features must be the values actually available **before the event time $t$**, not values updated after $t$.
- Example: predicting whether a user places an order at time $t$, using the **current** snapshot of "user's order count in the last 7 days" (which includes orders after $t$) is leakage; offline AUC is inflated and the model collapses online.
- Approach:
    - Store features with historical versions keyed by (entity, timestamp) and do an **as-of join** when generating training sets: take the feature value at $\max\{t_f : t_f\le t_{event}\}$
    - Account for **feature availability delay**: if a feature is only produced at T+1, require $t_f\le t_{event}-\text{delay}$ in the join
    - Split train/validation/test **by time**, not randomly
    - Inspect "too good to be true" features (a single feature with extremely high AUC is often a leakage signal)

```sql
-- As-of join sketch: for each event, take the latest feature value before the event time
SELECT e.user_id, e.event_ts, e.label, f.feat_value
FROM events e
LEFT JOIN LATERAL (
    SELECT feat_value FROM user_features f
    WHERE f.user_id = e.user_id AND f.feature_ts <= e.event_ts
    ORDER BY f.feature_ts DESC LIMIT 1
) f ON TRUE;
```

Q6: ***How do you perform data quality checks?***

- **Schema validation**: fields, types, value ranges, enum values, non-null constraints.
- **Statistical validation**: missing rate, mean/quantiles, cardinality, uniqueness, distribution difference vs the previous batch/training set.
- **Business rules**: e.g. price > 0, clicks ≤ impressions, timestamps not in the future.
- **Volume checks**: sudden drops/spikes in row count, upstream delays, missing partitions.
- **Label quality**: annotation agreement (Cohen's kappa), noisy-label detection, sudden changes in positive/negative ratio.
- Tools: Great Expectations, TFDV (TensorFlow Data Validation), Deequ, Pandera, dbt tests.
- Where to place them: at data ingestion, at feature pipeline output, before training (**gate**: abort training if validation fails), at the online inference request entry.
- Principle: data issues are the most common root cause of production incidents; checks must be **automated + alerting + blocking**, not just dashboards.

## Deployment and Serving

Q7: ***What model serving patterns exist? Which scenarios suit each?***

| Pattern | How | Latency | Use cases |
| --- | --- | --- | --- |
| **Batch prediction** | Scheduled job scores all entities, results written to KV/table | Hourly/daily freshness, extremely fast reads | User profiles, churn prediction, offline recommendation candidates, marketing lists |
| **Online real-time inference** | Request-triggered, synchronous call to the model service (REST/gRPC) | Milliseconds | Search ranking, risk decisions, ad CTR, chat |
| **Streaming inference** | Consume event streams (Kafka/Flink), near-real-time scoring | Seconds | Real-time transaction fraud monitoring, IoT anomaly detection, real-time feature updates |
| **On-device inference** | Model deployed to phones/edge devices | No network overhead | Privacy-sensitive, offline availability, ultra-low latency |

- Common hybrid: **batch precomputation + lightweight online re-ranking** (e.g. retrieval generated offline, ranking online).
- Serving tools: TorchServe, Triton Inference Server, TF Serving, BentoML, KServe/Seldon (K8s), vLLM/SGLang (LLMs).
- Design considerations: throughput vs latency (dynamic batching), autoscaling, cold start, multiple models sharing a GPU, fallback strategy (return rule-based/cached results on model timeout).

Q8: ***How do you optimize model inference latency?***

- **Model level**:
    - **Knowledge distillation**: train a small model (student) on soft labels from a large model (teacher), $\mathcal L=\alpha\,\mathrm{CE}(y,p_s)+(1-\alpha)T^2\,\mathrm{KL}(p_t^{T}\Vert p_s^{T})$
    - **Quantization**: FP32 → FP16/BF16 → INT8/INT4 (PTQ post-training quantization, QAT quantization-aware training; LLMs commonly use GPTQ, AWQ, FP8)
    - **Pruning**: only structured pruning (channels/layers/attention heads) yields real speedups
    - More efficient architectures, fewer features, early exit
- **Runtime/compilation**: ONNX Runtime, TensorRT, `torch.compile`, operator fusion, FlashAttention, KV cache (LLMs).
- **System level**:
    - **Caching**: result cache (popular requests), feature cache, embedding cache, prompt/prefix cache for LLMs
    - **Dynamic batching**, concurrency control, GPU utilization optimization
    - **Cascade architecture**: retrieval → pre-ranking → ranking; later stages use heavier models on fewer candidates
    - Precomputation (item tower computed offline + ANN search), asynchronous parallel feature fetching
- Look at **P99 tail latency** rather than the mean; profile first to locate the bottleneck (often feature fetching/network I/O rather than model compute).

Q9: ***What should a model registry and versioning record?***

- A model registry is the "artifact repository + state machine" for models (MLflow Model Registry, W&B Registry, SageMaker, Vertex AI, etc.).
- Each model version should record:
    - Model files and **signature** (input/output schema), dependency environment (Docker image/conda lock file)
    - **Lineage**: training code commit, data version/snapshot, feature list and versions, hyperparameters, random seed
    - Evaluation metrics (overall + key slices), model card (intended use, limitations, fairness)
    - State: `Staging` → `Production` → `Archived`, plus approver and launch time
- Purpose: traceability (online issues can be traced back to the data and code used), **one-click rollback**, audit and compliance, multiple versions in parallel (A/B).
- Data versioning tools: DVC, lakeFS, time travel in Delta Lake/Iceberg.

Q10: ***What are CI/CD/CT in ML? How do they differ from traditional software?***

- **CI (continuous integration)**: beyond code unit tests, also test **data** and **models**:
    - Unit tests for feature logic, data schema validation, smoke tests of the training pipeline on a small sample (it runs, loss decreases)
    - Model quality tests: no worse than baseline on a fixed eval set, no regression on key slices, inference latency/memory within budget
- **CD (continuous delivery/deployment)**: automatically package the model service, deploy to staging, push to production after shadow/canary validation; supports automatic rollback.
- **CT (continuous training)**: specific to ML; **automatically retrain** on a schedule or trigger (new data, drift, metric drop), automatically evaluate, and automatically register/push when gates are passed.
- Differences from traditional software:
    - Test results are **statistical** (metric thresholds), not deterministic pass/fail
    - Data must be validated; the artifact (model) depends on data, not just code
    - What you deploy is a "training pipeline", not just a model
- Tools: Kubeflow Pipelines, Airflow, Argo, TFX, Metaflow, GitHub Actions + MLflow.

Q11: ***What is the difference between shadow deployment, canary release and A/B testing?***

- **Shadow deployment (dark launch)**: the new model receives mirrored production traffic and produces predictions, but **does not affect users**; compare prediction differences, latency, error rate and feature consistency against the old model. Suitable for high-risk domains (risk control, pricing); the downside is it cannot measure the effect on user behavior.
- **Canary release**: route a small fraction (1%→5%→25%→100%) of real traffic to the new model, monitor system and guardrail metrics, and automatically roll back on anomalies. The goal is **reducing release risk**, not precisely measuring impact.
- **A/B testing**: randomly assign users to control/treatment groups and compare business metrics with statistical tests given sufficient sample size and duration; the goal is **causal evaluation of impact**.
    - Key points: choose the randomization unit (user rather than request, to avoid inconsistent experience for the same user), power analysis for sample size, predefine the primary metric and **guardrail metrics**, avoid peeking, beware of novelty effects and network effects (social/two-sided marketplaces need randomization by region or time slice)
    - Variance reduction: CUPED
- Others: **interleaving** for ranking models, with higher sensitivity; **multi-armed bandits** reduce regret during exploration.
- Typical sequence: offline evaluation → shadow → canary → A/B → full rollout.

## Monitoring and Retraining

Q12: ***What should be monitored in a production ML system? What is the difference between data drift and concept drift?***

- Monitoring layers:
    - **System**: QPS, latency (P50/P99), error rate, resource utilization
    - **Data**: input feature missing rate, distribution shift, schema changes, upstream data delays
    - **Model**: prediction distribution (mean, positive rate), confidence distribution, accuracy/AUC/calibration once labels arrive
    - **Business**: CTR, conversion rate, GMV, default rate, etc.
- Drift types (input $X$, label $Y$):
    - **Data drift/covariate shift**: $P(X)$ changes, $P(Y|X)$ stays the same (e.g. an influx of a new user segment)
    - **Label/prior shift**: $P(Y)$ changes (e.g. fraud rate rises)
    - **Concept drift**: $P(Y|X)$ changes; the correct output for the same input changes (e.g. post-pandemic spending habits, fraudsters changing tactics)
- Data drift does not necessarily hurt performance (drift in unimportant features doesn't matter); concept drift can only be confirmed with new labels, and is usually detected via performance metrics or proxy metrics.

Q13: ***How are PSI and KS computed? How are they used for drift detection?***

- **PSI (Population Stability Index)**: bin the feature (or model score) into $B$ bins based on the reference distribution, with reference proportion $e_i$ and current proportion $a_i$:
$$\mathrm{PSI}=\sum_{i=1}^{B}(a_i-e_i)\ln\frac{a_i}{e_i}$$
    - It is a symmetrized KL divergence; rule-of-thumb thresholds: $<0.1$ stable, $0.1\sim0.25$ minor change worth watching, $>0.25$ significant drift
    - Empty bins need smoothing; the binning scheme (equal-frequency) affects the result
- **KS statistic**: maximum difference between two empirical CDFs $D=\sup_x|F_1(x)-F_2(x)|$; the KS test gives a p-value; suited to continuous variables.
    - In risk control, KS also refers to **model discrimination**: the maximum gap between cumulative score distributions of good and bad samples
- Others: chi-square test, JS divergence, Wasserstein distance for categorical features; for high-dimensional/embedding data use MMD or a domain classifier (train a classifier to distinguish new vs old data; the higher the AUC, the larger the drift).
- In practice: with large samples statistical tests are almost always "significant", so focus on **effect size** (PSI, distances) and set alert thresholds in light of feature importance.

```python
import numpy as np

def psi(expected, actual, bins=10, eps=1e-6):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + eps
    a = np.histogram(actual, edges)[0] / len(actual) + eps
    return float(np.sum((a - e) * np.log(a / e)))
```

Q14: ***When should model retraining be triggered?***

- **Scheduled**: fixed daily/weekly cadence, simple and predictable, suited to slowly changing distributions.
- **Data-volume triggered**: enough new labeled data has accumulated.
- **Drift triggered**: PSI of inputs or predictions exceeds a threshold.
- **Performance triggered**: online metrics (once labels arrive) fall below a threshold, or business KPIs drop.
- **Event triggered**: upstream data/feature changes, business rule changes, major external events.
- Trade-off: retraining cost vs staleness loss; use offline backtests to estimate the "model decay curve over time" to set the frequency.
- Safety measures: a retrained model must pass a **champion/challenger** comparison against the production model and slice checks before being pushed; retraining triggered by a data anomaly can actually hurt (find the root cause first; dirty data must be fixed first).

Q15: ***What problems do label delay and feedback loops cause? How do you handle them?***

- **Label delay**: true labels take a long time to appear (loan default takes months, conversions may be delayed by days).
    - Problems: online performance cannot be evaluated promptly; recent samples are mislabeled as negative in training data (no conversion yet ≠ will not convert)
    - Handling: monitor with **proxy metrics/early signals** (clicks, first repayment); **delayed feedback modeling** (e.g. DFM modeling conversion delay, inserting samples as negative first then correcting with importance weighting); define a label maturity window; use drift monitoring as a leading indicator
- **Feedback loops**: model predictions influence subsequent training data.
    - Recommender systems: only recommended items get impressions and clicks → **exposure bias/popularity bias** reinforces itself, "the rich get richer"
    - Risk control: rejected applications never get default labels → **selection bias** (requires reject inference)
    - In settings like predictive policing they can amplify social biases
- Handling: keep a small amount of **random exploration traffic** (ε-greedy/bandit), **inverse propensity weighting (IPS)** for debiasing, log impressions and propensities at scoring time, causal inference methods, long-term holdout groups.

## Experiments and Engineering

Q16: ***What should experiment tracking record? How do you ensure reproducibility?***

- **Experiment tracking** (MLflow, Weights & Biases, Neptune, TensorBoard) records:
    - Code version (git commit + uncommitted diff), config/hyperparameters
    - Data version, feature list, data split method
    - Metric curves, final metrics, slice metrics, artifacts (models, plots, prediction files)
    - Environment: library versions, hardware, random seeds
- Elements of **reproducibility**:
    - **Code**: version control + config files (Hydra, etc.); don't rely on hidden notebook state
    - **Data**: immutable snapshots/versions (DVC, lakeFS, table time travel), record queries and time points
    - **Environment**: containerization, pinned dependency versions
    - **Randomness**: fix seeds (Python/NumPy/PyTorch/CUDA), `torch.use_deterministic_algorithms(True)`; non-deterministic GPU kernels and distributed communication order can still cause small differences
- Realistic goal: **statistical reproducibility** (consistent mean and variance across seeds) often matters more than bit-exact reproducibility.

Q17: ***How do you manage the cost of ML systems?***

- **Training cost**:
    - Choose an appropriately sized model; use small-scale experiments and scaling curves first
    - Mixed precision, gradient checkpointing, Spot/preemptible instances + checkpoint resumption
    - Incremental training instead of full retraining, less wasteful hyperparameter search (Bayesian optimization, early-stopping-based Hyperband/ASHA)
- **Inference cost** (often far exceeds training over the long run):
    - Quantization, distillation, batching, GPU sharing (MIG, multiple models per GPU), traffic-based **autoscaling**, serverless inference
    - Caching, cascades (a cheap model filters first, only hard samples go to the large model)
    - LLMs: model routing (small models handle easy requests), prompt compression, prefix cache, limiting output length
- **Data/storage cost**: feature lifecycle management, retiring unused features, hot/cold tiered storage.
- **Governance**: cost attribution and tagging by team/model, budget alerts; treat "cost per request" as a launch metric alongside accuracy and latency.

## LLMOps

Q18: ***What is special about LLMOps compared with traditional MLOps?***

- Assets shift from "trained weights" to a combination of **prompts, retrieval data, tool definitions, model provider versions and fine-tuned adapters**; many teams don't train models but orchestrate them.
- **Prompt versioning**: manage prompts as code (git / prompt registry), recording the template, variables, model version, temperature and other parameters; every change must pass evaluation before launch; note that upstream model upgrades also change behavior, so pin model snapshot versions.
- **Eval sets**:
    - Build a golden set covering core scenarios, edge cases and historical failures, continually augmented from production logs
    - Evaluation methods: exact match/programmatic checks (JSON format, unit tests), similarity to reference answers, **LLM-as-a-judge** (must be calibrated against human labels; beware position bias, length bias, self-preference), human evaluation
    - Used as **regression test** gates in CI
- **Guardrails**: input side (prompt injection detection, PII filtering, topic restrictions), output side (harmful content, format validation, factuality/citation checks, permission and argument validation for tool calls), human review where needed.
- **Tracing**: record the full chain of each request — prompt, retrieval results, tool calls, intermediate steps, token usage, latency, cost (LangSmith, Langfuse, Arize Phoenix, OpenTelemetry GenAI semantic conventions, etc.), used to debug multi-step agent failures and to build new eval samples.
- **Monitoring**: token cost, latency (time to first token, TTFT), refusal rate, hallucination rate, user feedback (thumbs up/down), retrieval hit rate.
- **RAG-specific**: index versioning, keeping the embedding model version consistent with the index, document update pipelines, retrieval quality evaluation (Recall@K, context relevance, faithfulness).

Q19: ***How do you continuously improve an LLM application after launch?***

- **Data flywheel**: production traces + user feedback → discover failure modes → cluster and attribute (retrieval missed? unclear prompt? insufficient model capability?) → add to the eval set → fix → regression test → release.
- Improvement levers in increasing cost: prompt optimization / few-shot → better retrieval (chunking, reranking, hybrid search) → tool and workflow design → fine-tuning (SFT, LoRA, preference optimization DPO) → switch to a stronger model.
- Online experiments: A/B test prompt or model versions, tracking task completion rate, human takeover rate and cost.
- Risk control: run the eval set as regression before a provider model upgrade; keep a fallback model; require human confirmation for high-risk actions (write operations, payments).

## Interview Answer Framework

Q20: ***How should you answer an ML system design interview?***

- A generic framework (proactively drive the pace and budget time within 45 minutes):
    1. **Clarify requirements (~5 minutes)**: business goal, user scale, QPS, latency budget, data availability, cold start, compliance constraints; translate the business goal into an **ML problem** (classification/ranking/regression/generation)
    2. **Metrics**: offline metrics (AUC, NDCG, Recall@K, calibration) + online metrics (CTR, retention, GMV) + guardrail metrics (latency, complaint rate), and explain how they align
    3. **Data and labels**: data sources, how labels are obtained (explicit/implicit feedback, human annotation), positive/negative definitions, sampling, label delay and bias
    4. **Feature engineering**: user/item/context/cross features, real-time vs batch features, embeddings
    5. **Model**: start from a simple baseline (rules, LR, GBDT), then deep models, explaining trade-offs; for large-scale settings give a **multi-stage architecture** (retrieval → pre-ranking → ranking → re-ranking)
    6. **Training**: loss function, class imbalance, training frequency, offline evaluation method (time-based split)
    7. **Serving**: batch/online, latency optimization, caching, feature store, fallback
    8. **Evaluation and launch**: shadow → canary → A/B
    9. **Monitoring and iteration**: drift, retraining, feedback loops, fairness and safety
- Bonus points: proactively discuss **trade-offs** (accuracy vs latency vs cost), back-of-the-envelope estimates (QPS × candidate count → compute), failure modes and fallbacks, give an end-to-end solution first and then deep-dive into one module.
- Common pitfalls: jumping straight to model choice, ignoring data and label sources, not mentioning metrics and monitoring, only covering offline and not online.

Q21: ***Using "design a fraudulent transaction detection system" as an example, briefly describe the key design points***

- **Requirements**: real-time blocking (latency budget e.g. < 100ms), extreme imbalance (fraud rate often far below 1%), cost of false blocks (user experience) vs cost of misses (financial loss).
- **Metrics**: PR-AUC, recall at a fixed false positive rate, blocked amount; online look at fraud loss rate and false-block complaint rate.
- **Data and labels**: chargebacks/manual review results as labels, with **label delay** of weeks; blocked transactions have no true labels (selection bias), so keep a small fraction released or manually reviewed.
- **Features**: real-time window statistics (transaction count and amount in the past 1 minute/1 hour), device/IP/geo consistency, merchant risk, graph features (number of accounts sharing a device, ring associations); stream processing (Flink) + online feature store, ensuring point-in-time correctness.
- **Model**: rule engine as a safety net + GBDT (strong baseline, interpretable) + optional GNN/sequence models; score → decision thresholds (approve/verify/block) set by a cost matrix.
- **Serving**: synchronous online scoring, parallelized feature fetching, fall back to rules on timeout.
- **Launch and monitoring**: shadow run compared with the rule system → canary; monitor score distribution PSI, sudden changes in block rate, adversarial drift (fraudsters adapting to the new model), frequent retraining.

Q22: ***What does a common MLOps tool stack look like? How do you choose?***

- **Orchestration**: Airflow, Kubeflow Pipelines, Argo Workflows, Dagster, Prefect, Metaflow.
- **Data/features**: Spark, Flink, dbt; Feast, Tecton; DVC, lakeFS, Delta Lake/Iceberg.
- **Experiments and registry**: MLflow, Weights & Biases, Neptune.
- **Training**: Ray (Train/Tune), PyTorch DDP/FSDP, DeepSpeed, cloud platforms (SageMaker, Vertex AI, Azure ML).
- **Serving**: Triton, TorchServe, BentoML, KServe, Ray Serve; for LLMs vLLM, SGLang, TensorRT-LLM.
- **Monitoring**: Prometheus + Grafana (system), Evidently, WhyLabs, Arize, NannyML (data/model); for LLMs Langfuse, LangSmith, Phoenix.
- Selection principles: team size and maturity (small teams should prefer managed services and few tools), integration with existing infrastructure (K8s, cloud vendor), avoid over-engineering — first solve the three most painful problems: reproducibility, monitoring and rollback.
