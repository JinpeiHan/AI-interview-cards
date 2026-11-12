# ML System Design and MLOps

## Lifecycle and Training

Q1: ***Describe an end-to-end machine learning lifecycle. What does MLOps focus on?***

- **An ML project roughly goes through problem definition → data → training and experimentation → evaluation → deployment → monitoring and feedback, then retrains on new data, closing the loop. MLOps is about making this loop reproducible, automated, observable and reversible.**
- Problem definition means translating the business goal into an ML objective and metrics, and pinning down constraints such as latency, cost and compliance.
- Difference from DevOps: an ML system depends on three kinds of assets — code, data and models — and can "silently degrade" without any code change.
- Maturity (Google's levels): Level 0 manual scripts → Level 1 automated training pipeline + continuous training → Level 2 CI/CD for the pipeline itself.
- A frequently cited point ("Hidden Technical Debt in ML Systems"): model code is only a small part; glue code, pipelines, configuration and feedback loops are the main risks.

Q2: ***What is the difference between offline (batch) training and online (incremental/streaming) training? How do you choose?***

- **Offline training periodically retrains on a batch of data, daily or weekly; it is stable, easy to reproduce and easy to roll back, but cannot keep up with distribution changes. Online training updates continuously on a data stream and takes effect within minutes, so it is very fresh, but it is easily thrown off by dirty data, hard to evaluate and roll back, and the system is more complex.**
- Online training suits fast-changing domains with many new IDs, like news, ads and short video; another downside is catastrophic forgetting.
- Common compromise: daily full training + hourly/minute-level incremental updates, with automatic validation gates (stop pushing if metrics look abnormal) and rollback-able snapshots.
- Decision criteria: how fast the distribution changes, how many new entities appear, and whether the gain from freshness justifies the engineering cost; offline backtests of "model from N days ago vs latest model" quantify the loss from model staleness.

## Features and Data

Q3: ***What is a feature store? What problems does it solve?***

- **A feature store is a system that centrally manages feature definitions, computation, storage and online serving (e.g. Feast, Tecton). Its main job is to remove inconsistency between the features used in training and those used online, while also making features reusable and ensuring training sets are built point-in-time correctly.**
- Offline store (data warehouse/data lake): generates training sets and supports historical lookback.
- Online store (low-latency KV such as Redis): serves the latest feature values in milliseconds at inference time.
- Feature registry: records metadata, owners, lineage and versions for discovery and governance; batch/streaming pipelines write the same definitions into both stores.

Q4: ***What is training-serving skew? What causes it? How do you avoid it?***

- **Training-serving skew means the data or processing logic the model sees in training differs from what it sees in online inference, so offline metrics look good but online results are poor. The most fundamental fix is to use the same feature logic on both sides, or to log the features actually used online and train on them directly.**
- Common causes: separate offline and online code (inconsistent missing-value handling, bucketing, tokenizer versions); different data freshness; feature leakage; filtered training samples versus full online traffic; preprocessing parameters not packaged with the model.
- Prevention: serve both sides from a feature store, and package preprocessing into the model (e.g. version the tokenizer together with the model).
- Log-and-wait: persist the features actually used at inference time and use them directly for training, eliminating skew at the root.
- Before launch, compare offline and online features and predictions on the same batch of requests; after launch, keep monitoring the distribution gap between the two.

Q5: ***What is point-in-time correctness? How do you prevent feature leakage/label leakage?***

- **Point-in-time correctness requires every training sample to use only feature values that were actually available before the moment the event happened. Using values updated after the event is feature leakage: offline AUC is inflated and the model collapses online.**
- Example: predicting whether a user orders at some moment, but using a current snapshot of "orders in the last 7 days" that includes later orders.
- Approach: store features with history keyed by (entity, timestamp), and build training sets with an as-of join that takes the latest value before the event time.
- Also account for feature availability delay: for a feature produced T+1, shift the join time further back.
- Split train/validation/test by time; be suspicious of any single feature that looks "too good to be true", which is often a leakage signal.

Q6: ***How do you perform data quality checks?***

- **Data quality checks should cover schema, statistical distribution, business rules, volume and label quality, with automated validation where data lands, where features are produced, before training and at the online entry point; failures should alert or even block, not just show up in a report.**
- Schema and business rules: fields, types, value ranges, non-null; e.g. price > 0, clicks ≤ impressions, timestamps not in the future.
- Statistics and volume: missing rate, quantiles, cardinality, distribution difference from the previous batch or the training set; sudden row-count changes, upstream delay, missing partitions.
- Label quality: annotation agreement, noisy labels, sudden shifts in positive/negative ratio.
- Tools: Great Expectations, TFDV, Deequ, Pandera, dbt tests; data problems are the most common root cause of production incidents.

## Deployment and Serving

Q7: ***What model serving patterns exist? Which scenarios suit each?***

- **There are four main patterns: batch prediction (scheduled offline scoring, for user profiles and marketing lists), online real-time inference (request-triggered, milliseconds, for search ranking, risk control and ads), streaming inference (consumes event streams, seconds, for real-time fraud monitoring) and on-device inference (deployed on the device, for privacy-sensitive and offline use). Real systems often mix them.**

| Pattern | Freshness/latency | Typical scenarios |
| --- | --- | --- |
| Batch prediction | Hourly/daily, very fast reads | User profiles, churn prediction, offline recommendation candidates |
| Online real-time inference | Milliseconds | Search ranking, risk decisions, ad CTR |
| Streaming inference | Seconds | Transaction fraud monitoring, IoT anomaly detection |
| On-device inference | No network overhead | Privacy-sensitive, offline availability |

- Common hybrid: retrieval generated in offline batch, ranking computed online.
- Tools: Triton, TorchServe, TF Serving, BentoML, KServe; for LLMs, vLLM and SGLang.
- Design points: throughput vs latency (dynamic batching), autoscaling, sharing GPUs across models, falling back to rules or cached results on timeout.

Q8: ***How do you optimize model inference latency?***

- **Profile first to find the bottleneck, then work on three levels: at the model level make the model smaller and faster (distillation, quantization, pruning), at the runtime level use compilation and kernel optimizations, and at the system level use caching, batching, cascades and precomputation. Look at P99 tail latency, and remember the bottleneck is often feature fetching and network I/O rather than model compute.**
- Model level: knowledge distillation; quantization to FP16/INT8/INT4 (post-training or quantization-aware training); only structured pruning gives real speedups; early exit.
- Runtime: ONNX Runtime, TensorRT, `torch.compile`, operator fusion, FlashAttention, KV cache for LLMs.
- System level: result/feature/embedding caches, dynamic batching, a retrieval → pre-ranking → ranking cascade, item embeddings precomputed offline + ANN search, fetching features in parallel.

Q9: ***What should a model registry and versioning record?***

- **A model registry is an "artifact repository + state machine" for models. Each version should record the model file and its input/output signature, the runtime environment, full lineage (code, data, features, hyperparameters, seed), evaluation results and its current stage, so that production issues can be traced back and rolled back with one click.**
- Lineage: training code commit, data snapshot, feature list and versions, hyperparameters, random seed.
- Evaluation: overall and key-slice metrics, plus a model card (intended use, limitations, fairness).
- Stage transitions: Staging → Production → Archived, with approver and launch time; supports auditing and running multiple versions for A/B.
- Tools: MLflow, W&B, SageMaker, Vertex AI; for data versioning, DVC, lakeFS, Delta Lake/Iceberg time travel.

Q10: ***What are CI/CD/CT in ML? How do they differ from traditional software?***

- **CI tests data and models in addition to code; CD automatically packages and deploys the model service with rollback support; CT (continuous training) is specific to ML: it automatically retrains and evaluates when new data, drift or a metric drop occurs, and ships automatically once the gates pass. The biggest differences from traditional software are that tests are statistical threshold checks, the deliverable depends on data, and what gets deployed is the whole training pipeline.**
- CI: unit tests for feature logic, schema validation, small-sample smoke training (runs end to end, loss goes down), model no worse than baseline, no regression on key slices, latency and memory within budget.
- CD: deploy to staging, validate with shadow or canary, then promote to production, with automatic rollback on anomalies.
- Tools: Kubeflow Pipelines, Airflow, Argo, TFX, Metaflow, GitHub Actions + MLflow.

Q11: ***What is the difference between shadow deployment, canary release and A/B testing?***

- **They serve different purposes: a shadow deployment runs the new model on live traffic without affecting users, to verify correctness and performance; a canary release gradually shifts a small share of real traffic to the new model, to reduce release risk; an A/B test splits traffic randomly and uses statistical tests, to measure the causal effect on business metrics.**
- Shadow deployment suits high-risk settings like risk control and pricing, but cannot measure the impact on user behavior.
- Canary goes 1% → 5% → 25% → 100%, monitoring guardrail metrics and rolling back automatically on anomalies.
- A/B essentials: randomize by user, compute sample size up front, predefine primary and guardrail metrics, do not peek, watch for novelty and network effects; CUPED reduces variance.
- Ranking models can also use interleaving (more sensitive); typical sequence: offline evaluation → shadow → canary → A/B → full rollout.

## Monitoring and Retraining

Q12: ***What should be monitored in a production ML system? What is the difference between data drift and concept drift?***

- **Monitor four layers: system (latency, error rate), data (feature missingness and distributions), model (prediction distribution, performance once labels arrive) and business (CTR, conversion, etc.). Data drift means the input distribution changed but the input-to-output relationship did not; concept drift means the correct answer for the same input has changed.**
- Data drift example: an influx of a new user population; it does not necessarily hurt performance, and drift in unimportant features can be ignored.
- Label drift: the proportion of labels itself changes, e.g. the fraud rate rises.
- Concept drift examples: spending habits changing after the pandemic, fraudsters switching tactics; it can only be confirmed with new labels and is usually detected through performance or proxy metrics.

Q13: ***How are PSI and KS computed? How are they used for drift detection?***

- **PSI bins a feature or model score by the baseline distribution and compares the share of each bin between baseline and current; it is a symmetrized KL divergence. KS is the maximum gap between two empirical cumulative distributions. Both quantify "how far the current distribution is from the baseline", and an alert fires when a threshold is exceeded.**
- PSI rules of thumb: < 0.1 stable, 0.1–0.25 worth watching, > 0.25 significant drift; smooth empty bins, and equal-frequency binning is common.
- KS suits continuous variables and gives a p-value; in risk control, KS also refers to how well a model separates good from bad samples.
- Other methods: chi-square or JS divergence for categorical features, Wasserstein distance; for high-dimensional embeddings, MMD or a classifier trained to tell old data from new (higher AUC means more drift).
- With large samples statistical tests are almost always "significant", so look at effect size and set thresholds with feature importance in mind.
- Key formula: $\mathrm{PSI}=\sum_{i}(a_i-e_i)\ln\frac{a_i}{e_i}$ ($e_i$, $a_i$ are the baseline and current shares of bin $i$)

Q14: ***When should model retraining be triggered?***

- **Common triggers are: a schedule, accumulating enough new labels, detected drift, a drop in online performance, and events such as upstream data or business rule changes. Frequency is a trade-off between retraining cost and staleness loss, and can be chosen from an offline backtest that plots how the model decays over time.**
- Scheduled triggers are simple and predictable, suited to distributions that change smoothly.
- Drift triggers (e.g. PSI over threshold) and performance triggers (metric or KPI drop) are more targeted.
- Safety: a new model must pass a champion/challenger comparison with the production model and slice checks before it is pushed.
- If drift comes from a data anomaly, find the root cause and fix the dirty data first; blind retraining can do harm.

Q15: ***What problems do label delay and feedback loops cause? How do you handle them?***

- **Label delay means true labels take a long time to arrive (loan default can take months), so performance cannot be evaluated in time and recent samples get mislabeled as negatives; a feedback loop means the model's decisions shape future training data, so bias reinforces itself. The first is handled with proxy metrics and delayed-feedback modeling, the second with reserved exploration traffic and debiasing weights.**
- Label delay: monitor with early signals (clicks, first repayment); model conversion delay with correction and importance weighting; set a label maturity window; use drift monitoring as a leading indicator.
- Feedback loop examples: in recommendation only recommended items get clicks, causing "rich get richer" exposure bias; in risk control rejected applications never get default labels, causing selection bias (needs reject inference).
- Feedback loop handling: keep a little random exploration traffic (ε-greedy/bandit), inverse propensity weighting (IPS), log exposures and scoring propensities, keep a long-term holdout group.

## Experiments and Engineering

Q16: ***What should experiment tracking record? How do you ensure reproducibility?***

- **Experiment tracking should record code version, config and hyperparameters, data version and split method, all metrics and artifacts, plus the runtime environment and random seed. Reproducibility rests on four things: versioned code, data snapshots, containerized environments and fixed randomness.**
- Tools: MLflow, Weights & Biases, Neptune, TensorBoard.
- Code: git commit + config files (e.g. Hydra), do not rely on hidden notebook state; data: immutable snapshots (DVC, lakeFS, time travel).
- Randomness: fix Python/NumPy/PyTorch/CUDA seeds and enable deterministic algorithms; non-deterministic GPU kernels and distributed communication can still cause tiny differences.
- Realistic goal: consistent mean and variance across seeds (statistical reproducibility) usually matters more than bit-exact results.

Q17: ***How do you manage the cost of ML systems?***

- **Cost splits into training, inference and data storage, and over the long run inference is usually the most expensive. The approach: for training, use an appropriately sized model and cheap compute; for inference, make the model smaller, batch more, cache more and filter with cheap models first; attribute cost by team and model, and treat cost per request as a launch metric alongside accuracy and latency.**
- Training: run small-scale experiments first to see scaling trends; mixed precision, gradient checkpointing, preemptible instances + checkpoint resume; incremental training, early-stopping hyperparameter search (Hyperband/ASHA).
- Inference: quantization, distillation, batching, GPU sharing, autoscaling, cascades; for LLMs also model routing, prompt compression, prefix cache, limiting output length.
- Data: retire unused features, hot/cold tiered storage; for governance, set budget alerts.

## LLMOps

Q18: ***What is special about LLMOps compared with traditional MLOps?***

- **The assets LLMOps manages are no longer mainly self-trained weights but a combination of prompts, retrieval data, tool definitions, model provider versions and fine-tuned adapters; many teams orchestrate models rather than train them. So the focus shifts to prompt versioning, eval sets as regression tests, input/output guardrails, end-to-end tracing and cost/quality monitoring.**
- Manage prompts as code, recorded together with model version and parameters, and evaluate changes before release; pin model snapshots, because provider upgrades also change behavior.
- Eval sets: cover core scenarios, edge cases and past failures, continually topped up from production; use programmatic checks, comparison with reference answers, LLM-as-a-judge (calibrate against humans, watch for position and length bias) and human evaluation.
- Guardrails: on input, defend against prompt injection and filter PII; on output, check harmful content, format, factuality, and the permissions and arguments of tool calls.
- Tracing and monitoring: record prompts, retrieval results, tool calls, tokens and latency (e.g. TTFT); RAG also needs versioning of the index and embedding model, and evaluation of retrieval quality and faithfulness.

Q19: ***How do you continuously improve an LLM application after launch?***

- **Through a data flywheel: find failure cases in production traces and user feedback, attribute them (retrieval missed, unclear prompt, or insufficient model capability), add them to the eval set, fix, run regression tests and release. Try fixes in order of increasing cost.**
- Improvement order: prompt optimization / few-shot → better retrieval (chunking, reranking, hybrid search) → tool and workflow design → fine-tuning (SFT, LoRA, DPO) → a stronger model.
- Run A/B tests online on prompt or model versions, tracking task completion rate, human takeover rate and cost.
- Risk control: run regressions before a provider upgrade; keep a fallback model; add human confirmation for high-risk actions (writes, payments).

## Interview Answer Framework

Q20: ***How should you answer an ML system design interview?***

- **Drive the pace with a fixed framework: first clarify requirements and translate the business goal into an ML problem, then cover metrics, data and labels, features, model, training, serving, launch evaluation and monitoring/iteration in turn. Give an end-to-end design first, then go deep on one module, and keep discussing the trade-offs between accuracy, latency and cost.**
- Clarify (about 5 minutes): business goal, scale and QPS, latency budget, data availability, cold start and compliance.
- Metrics: offline (AUC, NDCG, calibration), online (CTR, retention) and guardrail metrics, and explain how they align.
- Model: start from baselines like rules, LR or GBDT before deep models; at large scale use a multi-stage retrieval → pre-ranking → ranking → re-ranking architecture.
- Bonus points: back-of-the-envelope estimates (QPS × candidates), failure modes and fallbacks; common mistakes: picking a model right away, ignoring data and labels, not mentioning metrics and monitoring, only talking about offline.

Q21: ***Using "design a fraudulent transaction detection system" as an example, briefly describe the key design points***

- **This is a real-time, extremely imbalanced problem where both false blocks and missed fraud are costly. The core design: use real-time window statistics and graph features, score online with GBDT backed by a rule engine, and set thresholds from a cost matrix to decide allow, verify or block; at the same time handle label delay and selection bias, and keep monitoring adversarial drift.**
- Metrics: PR-AUC, recall at a fixed false positive rate, blocked amount; online, fraud loss rate and false-block complaint rate.
- Labels: chargebacks and manual review outcomes, delayed by weeks; blocked transactions have no true label, so keep a small share released or manually reviewed.
- Features: transaction count and amount over the last minute/hour, device/IP/geo consistency, merchant risk, graph features such as shared devices; use Flink + an online feature store with point-in-time correctness.
- Serving and launch: latency budget e.g. < 100ms, fetch features in parallel, fall back to rules on timeout; shadow against the rule system first, then canary, monitor score PSI and sudden changes in block rate, retrain frequently.

Q22: ***What does a common MLOps tool stack look like? How do you choose?***

- **The stack divides by stage: orchestration, data and features, experiments and registry, training, serving and monitoring, each with several mainstream options. Choose based on team size and existing infrastructure, avoid over-engineering, and solve the three most painful problems first: reproducibility, monitoring and rollback.**
- Orchestration: Airflow, Kubeflow Pipelines, Argo, Dagster, Prefect, Metaflow; data/features: Spark, Flink, dbt, Feast, Tecton, DVC, lakeFS, Delta Lake/Iceberg.
- Experiments and registry: MLflow, W&B, Neptune; training: Ray, PyTorch DDP/FSDP, DeepSpeed, cloud platforms (SageMaker, Vertex AI, Azure ML).
- Serving: Triton, TorchServe, BentoML, KServe, Ray Serve; for LLMs, vLLM, SGLang, TensorRT-LLM.
- Monitoring: Prometheus + Grafana (system), Evidently, WhyLabs, Arize, NannyML (data/model); for LLMs, Langfuse, LangSmith, Phoenix. Small teams should favor managed services.
