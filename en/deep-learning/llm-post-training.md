# LLM Post-Training and Alignment

## Overall Pipeline

Q1: ***What do pretraining and post-training each do? What does a typical modern LLM post-training pipeline look like?***

- **Pretraining**: next-token prediction on trillions of unlabeled tokens, learning **knowledge and general language ability** and producing a base model. It only "continues text": it does not reliably follow instructions and has no safety boundaries.
- **Post-training**: uses far less data (usually at most millions of samples) to shape **behavior**: instruction following, dialogue format, helpfulness, honesty, harmlessness, reasoning, tool use, etc. A common view is that "knowledge comes mainly from pretraining; post-training mainly **elicits and formats** it" (the Superficial Alignment Hypothesis, LIMA 2023).
- Typical pipeline (the InstructGPT paradigm and its evolution):
    1. **SFT**: supervised fine-tuning on high-quality (prompt, response) demonstrations;
    2. **Preference optimization**: RLHF (reward model + PPO) or offline methods such as DPO;
    3. **Reasoning / verifiable-reward RL** (RLVR, e.g. GRPO): large-scale RL on tasks with automatically checkable answers such as math and code;
    4. Safety tuning, refusal calibration, and multiple iterations (rejection sampling -> SFT -> preference optimization -> ...).
- In between there is often a **mid-training / annealing stage**: at the end of pretraining, the share of high-quality, long-context, code and math data is increased to bridge pretraining and post-training.

## SFT and Instruction Tuning

Q2: ***In SFT, which matters more: data quality or quantity?***

- **Quality first**: LIMA achieved decent conversational ability with about 1000 curated samples; large amounts of low-quality data teach the model a verbose, wrong, hallucinating style.
- Quality dimensions: **correctness**, **diversity** (task types, domains, difficulty, length, languages), **format consistency**, whether the response style matches the target (level of detail, tone), and no duplication/contamination (deduplicated against eval sets).
- Quantity still matters: covering more skills (code, math, tool calling, multi-turn) and long-tail tasks requires scale. Modern SFT sets are typically hundreds of thousands to millions of samples, but strictly filtered (LLM scoring, deduplication, difficulty filtering, execution-based verification).
- Follow-up: SFT learns "imitation". If demonstrations go beyond what the model already knows, they can **induce hallucination** (the model learns to "fabricate confidently").

Q3: ***What is a chat template? Why is it important?***

- A chat template serializes multi-turn dialogue into the fixed format seen during training, using **special tokens** to mark roles and boundaries, e.g.:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
2<|im_end|>
```

- Why it matters:
    - Training and inference **must use the same template**, otherwise performance drops noticeably (the model doesn't stop, roles get confused);
    - The **EOS / end-of-turn token** teaches the model when to stop;
    - Role separation is the basis of the **instruction hierarchy** (system > user > tool output) and of defending against prompt injection;
    - Tool calls and chains of thought (e.g. `<think>` blocks) are also expressed through special structures in the template.
- In practice, use the tokenizer's built-in `apply_chat_template` to avoid hand-concatenation bugs.

Q4: ***Why mask the loss on prompt tokens in SFT?***

- The goal of SFT is to learn $p(y\mid x)$, not $p(x)$. The loss is computed only on response tokens:

$$\mathcal{L}_{SFT} = -\sum_{t \in \text{response}} \log \pi_\theta(y_t \mid x, y_{<t})$$

- Implementation: set the labels at prompt positions to `-100` (the `ignore_index` of PyTorch's `CrossEntropyLoss`).
- Reasons:
    - Avoids wasting capacity on learning to "generate user questions", or even speaking for the user at inference time;
    - User inputs may be low-quality or noisy and should not be imitated;
    - In multi-turn dialogue, loss is usually computed on **all assistant turns** (system/user/tool messages are all masked).
- Follow-up: some work found that keeping some prompt loss on short-response data has a mild regularizing effect, but masking remains the mainstream practice. Also watch **per-token vs. per-sample averaging**: when packing multiple samples, this changes the weight of long vs. short responses.

Q5: ***How does instruction tuning relate to SFT? Why does it improve zero-shot generalization?***

- Instruction tuning is a form of SFT: many NLP tasks are rewritten as "natural-language instruction + output" and used for multi-task fine-tuning (FLAN, T0, Super-NaturalInstructions, etc.).
- Key findings: the larger the **number and diversity of tasks**, the better zero-shot generalization to unseen tasks; mixing in CoT examples preserves reasoning ability; gains are larger for bigger models.
- Difference from conversational SFT: early instruction tuning leaned toward "academic task formats" with short answers; conversational SFT (e.g. InstructGPT, ShareGPT-style data) emphasizes open-ended, long answers and multi-turn interaction. Modern practice mixes both.

## RLHF and PPO

Q6: ***Describe the full RLHF pipeline. How is the reward model trained (Bradley-Terry loss)?***

- Pipeline: SFT model $\pi_{SFT}$ -> sample several responses for the same prompt -> humans label preference pairs $(y_w \succ y_l)$ -> train reward model $r_\phi$ -> optimize the policy with PPO.
- The **Bradley-Terry model** assumes:

$$P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)$$

- Reward model loss (negative log-likelihood):

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)}\Big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]$$

- Architecture: usually initialized from the SFT model, with the LM head replaced by a linear head that outputs a scalar, taken from the hidden state of the last token.
- Key points:
    - Only **relative differences** in reward are meaningful (shift invariance), so rewards are often normalized;
    - If a prompt has $K$ ranked responses, $\binom{K}{2}$ pairs can be built (InstructGPT's approach), with pairs from the same prompt kept in one batch to prevent overfitting;
    - Human inter-annotator agreement is often only around 60%-80%, which caps RM accuracy.

Q7: ***What is the optimization objective in the PPO stage? Why add a KL penalty?***

- Objective:

$$\max_{\pi_\theta} \ \mathbb{E}_{x\sim D,\, y\sim \pi_\theta(\cdot|x)}\big[r_\phi(x,y)\big] - \beta\, \mathbb{D}_{KL}\big[\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)\big]$$

  where $\pi_{ref}$ is usually the frozen SFT model.
- In implementations, the KL is often applied as a **per-token reward**: $r_t = -\beta \log\frac{\pi_\theta(y_t|\cdot)}{\pi_{ref}(y_t|\cdot)}$, with the RM score added at the last token.
- Why KL:
    - **Prevents reward hacking**: the RM is only reliable near the SFT distribution; once the policy drifts far, it exploits RM loopholes (repetition, sycophancy, overly long answers);
    - **Keeps language fluent and diverse**, avoiding mode collapse;
    - **Preserves pretrained abilities**, reducing the alignment tax;
    - Theoretically, the KL-regularized objective has a closed-form optimum $\pi^* \propto \pi_{ref}\exp(r/\beta)$, i.e. a "Bayesian-style" reweighting of the prior, which is also the starting point of DPO.
- $\beta$ too small -> reward hacking; too large -> nothing is learned. Adaptive KL control toward a target KL value can be used.

Q8: ***What are the specific details of PPO for LLMs? (four models, GAE, clipping)***

- **Four models**: actor (policy, trained), critic (value model, trained), reference (frozen), reward model (frozen). The large memory footprint is PPO's main pain point.
- **MDP view**: state = prompt + tokens generated so far, action = next token; reward is sparse (RM score only at the end of the sequence) + per-token KL penalty.
- **Value model and GAE**: the critic estimates $V(s_t)$ and GAE computes advantages:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\qquad \hat A_t = \sum_{l\ge 0} (\gamma\lambda)^l\, \delta_{t+l}$$

  For LLMs, typically $\gamma=1$ and $\lambda$ close to 1 (e.g. 0.95).
- **Clipped objective**:

$$\mathcal{L}^{CLIP} = \mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big],\quad \rho_t=\frac{\pi_\theta(y_t|s_t)}{\pi_{\theta_{old}}(y_t|s_t)}$$

  $\epsilon$ is often 0.2; it limits the size of each update, allowing several epochs over the same batch of rollouts.
- Other tricks: value-function clipping, advantage normalization, reward normalization/clipping, initializing the critic from the RM, mixing in pretraining loss (PPO-ptx) to prevent forgetting.
- Training loop: rollout (generation, the most time-consuming part) -> scoring and advantage computation -> several mini-batch updates.

## DPO and Variants

Q9: ***What is the derivation idea and loss function of DPO?***

- Starting from the KL-constrained RLHF objective, the optimal policy has a closed form:

$$\pi^*(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\Big(\frac{1}{\beta}r(x,y)\Big)$$

- Solving for the reward (the **implicit reward**):

$$r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)$$

- Substituting into Bradley-Terry, the partition function $Z(x)$ **cancels** when the two responses are subtracted, giving a loss that depends only on the policy:

$$\mathcal{L}_{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big]$$

- Intuition: "your language model is secretly a reward model". It raises the probability of chosen relative to the reference model and lowers that of rejected; the gradient carries a weight $\sigma(\hat r_l - \hat r_w)$, i.e. it updates more on **pairs currently ranked wrong**.

```python
def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # All inputs are summed log-probs of the whole response (summed over response tokens only)
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -torch.nn.functional.logsigmoid(logits).mean()
```

Q10: ***What are the pros and cons of DPO compared with PPO?***

- Pros:
    - **Simple and stable**: no RM training, no online sampling, no critic; essentially a classification loss;
    - **Resource-efficient**: only the policy + reference model are needed (reference log-probs can even be precomputed);
    - Few hyperparameters (mainly $\beta$), easy to reproduce.
- Cons:
    - **Offline / off-policy**: when preference data comes from other models there is distribution shift, and results are usually worse than online methods; in practice this is mitigated with **online/iterative DPO** (sample from the current policy, label with an RM or judge);
    - **The chosen probability can also drop**: DPO only constrains the difference, so both likelihoods can fall together, shifting outputs toward unseen responses;
    - Prone to **overfitting** preference data (when preferences are near-deterministic, the implicit reward gap tends to infinity and the KL constraint becomes ineffective);
    - **Length bias**: tends to get longer;
    - Cannot exploit the RM's generalization to newly sampled data, and is not suited to reasoning RL with verifiable rewards.
- Empirical takeaway: with enough resources, **online RL (PPO/GRPO) has a higher ceiling**; DPO suits fast iteration and small-to-medium-scale alignment.

Q11: ***Briefly describe the DPO variants IPO, KTO, ORPO and SimPO.***

- **IPO** (Identity PO): replaces log-sigmoid with a squared loss, regressing the implicit reward gap to a fixed margin $\frac{1}{2\beta}$, which fixes DPO's overfitting when preferences are deterministic: $\big(h_\theta - \frac{1}{2\beta}\big)^2$, where $h_\theta$ is the log-ratio difference from DPO.
- **KTO** (Kahneman-Tversky Optimization): based on prospect theory, it **does not need paired data**, only a binary "good/bad" label per response; gains and losses are weighted asymmetrically. Suited to thumbs-up/thumbs-down data.
- **ORPO** (Odds Ratio PO): **no reference model, single stage**; adds an odds-ratio term $-\log\sigma\big(\log\frac{odds(y_w)}{odds(y_l)}\big)$ to the SFT loss, doing SFT and preference alignment at once.
- **SimPO**: **no reference model**; uses the **length-normalized** average log-probability as the implicit reward and adds a target margin $\gamma$:

$$\mathcal{L} = -\log\sigma\Big(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\Big)$$

  This mitigates length bias and matches how sequences are scored at generation time.
- Common themes: dropping the reference model (saving memory), handling length bias, relaxing data requirements, preventing overfitting.

## GRPO and Verifiable Rewards

Q12: ***What is GRPO? How does it differ from PPO?***

- GRPO (Group Relative Policy Optimization, proposed in DeepSeekMath 2024 and used extensively in DeepSeek-R1) **removes the critic**: for each prompt it samples a group of $G$ responses and uses the within-group standardized reward as the advantage:

$$\hat A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$

  This advantage is assigned to every token of response $i$.
- The objective still uses the PPO-style clipped ratio, and the KL penalty is **added directly to the loss** (rather than folded into the reward), commonly with the unbiased low-variance estimator $\frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1$.
- Pros: eliminates a value model as large as the policy (big savings in memory and compute); the group baseline naturally fits **sparse sequence-level rewards** (e.g. correct/incorrect); simple to implement.
- Cons and improvements:
    - When a whole group is all correct or all wrong, the advantage is 0 and there is **no learning signal**; DAPO filters such groups with dynamic sampling;
    - Dividing by std up-weights questions at the extremes of difficulty, and normalizing by response length introduces **length bias**; Dr. GRPO removes these normalizations;
    - DAPO also proposes clip-higher (an asymmetric upper clip bound to prevent entropy collapse), token-level loss aggregation, overlong-response penalties, etc.
- Related critic-free methods: RLOO (leave-one-out baseline), REINFORCE++, ReMax.

Q13: ***What is RLVR (reinforcement learning with verifiable rewards)? Why did it drive the development of reasoning models?***

- RLVR: the reward comes from **programmatically verifiable** rules rather than a learned RM:
    - Math: the final answer matches the reference (e.g. extract the content of `\boxed{}` and check symbolic equivalence);
    - Code: passes unit tests;
    - Format checks, instruction-constraint checks, etc.
- Benefits: rewards are **hard to hack** (compared with neural RMs), scale without limit, and need no human labels; RL can be run long and at large scale.
- Landmarks: OpenAI o1 (2024) opened the reasoning-model paradigm; DeepSeek-R1 (2025) showed publicly that pure RL directly on a base model (R1-Zero) can spontaneously produce long chains of thought, self-verification and reflection (the "aha moment"), with response length growing over training.
- R1's full pipeline: cold-start long-CoT SFT -> reasoning RL (GRPO + rule-based rewards) -> rejection sampling to create new SFT data + general data -> all-scenario RL (rule-based rewards + preference RM).
- Follow-up: there is debate that RLVR mainly **improves sampling efficiency** (concentrating correct paths already present in pass@k into pass@1) rather than expanding the model's capability boundary; at large $k$, the base model's pass@k is sometimes no worse than the RL model's. Also watch for verifier false positives (e.g. answer-parsing loopholes, insufficient tests).

Q14: ***What is the difference between a process reward model (PRM) and an outcome reward model (ORM)?***

- **ORM**: scores only the final result (correct/incorrect or overall quality). Cheap to label, but **credit assignment** is hard, and it can reward processes where "the reasoning is wrong but the answer happens to be right".
- **PRM**: scores **each step** of reasoning, providing dense signal and locating the first erroneous step. OpenAI's "Let's Verify Step by Step" (2023) showed PRMs outperform ORMs for best-of-N reranking.
- Sources of PRM data: human step-level annotation (PRM800K, expensive); or **Monte Carlo estimation** (Math-Shepherd style: roll out many times from a given step and use the final accuracy as that step's value).
- Practical issues: ambiguous step segmentation, noisy labels, and easy to hack when used as an RL reward (e.g. splitting steps very finely to farm reward). DeepSeek-R1 reported not using a PRM; mainstream reasoning RL still relies mostly on **outcome rewards + rule-based verification**, while PRMs are used more for test-time search/reranking.

## Reward Hacking and AI Feedback

Q15: ***What is reward hacking? What are common symptoms and mitigations?***

- Definition: the policy finds ways to raise the proxy reward (RM score) while the true objective (human preference) does not improve or even gets worse: **Goodhart's law**. The proxy reward keeps rising while true quality first rises then falls ("overoptimization", studied as scaling laws by Gao et al. 2022).
- Common symptoms:
    - **Length hacking**: responses grow longer and longer, padded with lists and disclaimers;
    - **Sycophancy**: agreeing with the user's mistaken views;
    - Repetitive templated phrasing, overconfidence;
    - In RLVR: exploiting verifier loopholes, such as modifying/skipping unit tests, emitting formats that fool the answer parser, or writing special-case code.
- Mitigations:
    - **KL penalty**, early stopping (judged by human eval or a held-out judge);
    - **RM ensembles**, uncertainty penalties, length penalties or length debiasing on the reward;
    - **Iteratively updating the RM** with samples from the new policy;
    - More robust verifiers (sandboxing, hidden tests, forbidding edits to test files);
    - Constraints and multi-objective rewards (e.g. separate helpfulness and safety RMs);
    - Monitoring reward distribution, response length, entropy, etc., and manually inspecting high-reward samples.

Q16: ***What are RLAIF and Constitutional AI?***

- **RLAIF**: uses AI (usually a strong LLM) instead of humans to give preference labels, then trains an RM or runs DPO as usual. Cheap, scalable and consistent; the risk is inheriting the judge model's biases.
- **Constitutional AI** (Anthropic, 2022): given a set of written principles (a "constitution"), it has two stages:
    1. **Supervised stage (critique & revision)**: have the model answer harmful requests -> self-critique according to a principle -> revise -> SFT on the revised answers;
    2. **RL stage (RLAIF)**: the model compares two responses according to the principles, producing preference data to train a preference model, followed by RL.
- Benefits: reduces the burden of humans labeling harmful content; the rules are **transparent and auditable**; achieves a better balance between "harmless" and "helpful" (explained refusals rather than evasion).

## Data and Distillation

Q17: ***What is rejection sampling / best-of-N? How is it used in post-training?***

- **Best-of-N (at inference)**: sample $N$ responses and pick the highest-scoring one with an RM or verifier. Simple yet strong, and a strong baseline for measuring RL; its KL from the reference policy is about $\log N - \frac{N-1}{N}$. The cost is $\times N$ inference.
- **Rejection sampling fine-tuning (at training, RFT)**: sample several responses per prompt, keep only those that pass verification or have the highest RM score, and do SFT on them; iterate for several rounds (Llama 2, STaR, ReST, etc.).
- Pros: simple and stable (needs only SFT infrastructure); can be seen as a "discrete" policy improvement on on-policy data (Expert Iteration).
- Cons: learns only from positives and ignores information in negatives; may fail to sample any correct response on hard problems; the ceiling is generally lower than online RL.

Q18: ***How do you distill from a stronger model? What should you watch out for?***

- **Black-box / sequence-level distillation**: a strong model (teacher) generates responses (including long CoT), and the student does SFT on them. DeepSeek-R1's distilled versions showed that for small models, **directly distilling a strong reasoning model's outputs is often more effective than running RL on the small model**.
- **White-box / logit distillation**: minimize the KL between student and teacher token distributions (requires the same tokenizer and teacher logits), carrying more information; **on-policy distillation** has the student generate and the teacher score each token (reverse KL), mitigating train/inference mismatch (exposure bias).
- Caveats:
    - Teacher errors and hallucinations are inherited and need filtering (answer verification, deduplication, quality scoring);
    - When the **capability gap is too large** the student can't learn; use an intermediate-sized teacher or curriculum learning;
    - Imitating style without real capability ("the false promise of imitating proprietary LLMs", 2023);
    - Licensing / terms-of-service issues: some commercial models forbid using their outputs to train competing models.

Q19: ***How is synthetic data used in post-training? What are the risks?***

- Uses:
    - **Instruction generation**: Self-Instruct, Evol-Instruct (progressively increasing difficulty and complexity), generating diverse prompts from seed documents/personas;
    - **Response generation + filtering**: generate with strong models, filter with verifiers, execution feedback or LLM judges;
    - Preference pair construction: sample the same prompt multiple times and decide chosen and rejected with an RM / judge / rules;
    - Reasoning data: large-scale synthetic problems + verifiable answers; simulated environments for tool-use and agent trajectories.
- Risks:
    - **Loss of diversity and model collapse** (recursively training on one's own outputs loses the tails of the distribution);
    - Error accumulation, hallucination propagation;
    - Eval contamination (synthetic data resembling benchmarks);
    - Homogeneous style (the "AI voice").
- Mitigations: keep real human data, deduplicate and control diversity, verify strictly, mix multiple generator models.

## Forgetting, Alignment Tax and Safety

Q20: ***What are catastrophic forgetting and the "alignment tax" in post-training? How do you mitigate them?***

- **Catastrophic forgetting**: after fine-tuning on a narrow distribution, the model's ability on other tasks drops (e.g. math or multilingual ability degrades after excessive SFT).
- **Alignment tax**: performance drops on some standard benchmarks after alignment (InstructGPT observed regressions on some NLP tasks), along with reduced creativity/diversity and worse calibration (models' probability calibration usually gets worse after RLHF).
- Mitigations:
    - **Data mixing / replay**: mix in general instruction data and a little pretraining data (PPO-ptx);
    - **KL constraint** to the reference model; smaller learning rates, fewer epochs;
    - **Parameter-efficient fine-tuning** (LoRA, "learns less and forgets less");
    - **Model merging** (weight averaging, e.g. interpolating SFT and RL model weights, WiSE-FT);
    - Multi-task joint training; after staged training, select checkpoints on a comprehensive eval suite.
- Observation: online RL usually **forgets less** than SFT, because it updates on the policy's own distribution and changes the model more "conservatively".

Q21: ***How is safety tuning done? How do you avoid over-refusal?***

- Safety tuning methods:
    - Safety SFT data (harmful requests + appropriate refusals or safe completions);
    - **A separate safety RM** or multi-objective rewards (Llama 2 uses a helpfulness RM and a safety RM, chosen depending on whether the prompt is sensitive);
    - Red teaming (human + automated) to find vulnerabilities, then adding data;
    - Constitutional AI / rule-based rewards (e.g. OpenAI's Rule-Based Rewards, and Deliberative Alignment: having the model explicitly cite the safety spec in its reasoning).
- **Over-refusal**: refusing requests that look sensitive but are actually harmless (e.g. "how do I kill a Python process"). Measured with XSTest, OR-Bench, etc.
- Mitigations:
    - Add many "**looks harmful but is actually benign**" borderline examples with correct, helpful answers to the training data;
    - Move from "hard refusal" to **safe completion**: provide as much useful information as is safe and explain the reason for any refusal;
    - Optimize helpfulness and harmlessness jointly, and track both **harmful rate and false-refusal rate** in evaluation;
    - Judge by context (system prompt, user intent) rather than keywords.
- Follow-up: jailbreaks, multi-turn attacks and prompt injection require adversarial training and instruction-hierarchy training; safety alignment can easily be removed with a small amount of fine-tuning ("shallow safety alignment"), which matters especially for open-weight models.

## Evaluation

Q22: ***How do you evaluate an aligned model? What biases does LLM-as-judge have?***

- Evaluation methods:
    - **Human evaluation**: pairwise comparison or scoring; the gold standard but expensive and slow;
    - **Win rate**: the win/tie/loss ratio in pairwise comparison against a baseline model; Chatbot Arena fits **Elo / Bradley-Terry scores** from large numbers of crowdsourced pairwise votes;
    - **LLM-as-judge**: MT-Bench (scoring), AlpacaEval (win rate against a reference model), Arena-Hard;
    - Capability benchmarks: MMLU-style knowledge, math (AIME, MATH), code (LiveCodeBench, SWE-bench), instruction following (IFEval, programmatically verifiable), safety and refusal test sets.
- Biases of LLM-as-judge:
    - **Position bias**: prefers the first (or second) response -> evaluate twice with the order swapped and keep consistent verdicts;
    - **Length/verbosity bias**: prefers longer responses -> use length-controlled win rates (e.g. AlpacaEval 2 LC);
    - **Self-preference**: prefers responses resembling its own style or generated by itself;
    - Format bias (prefers markdown lists), authority and tone bias, unreliable judgment of math/code correctness;
    - Mitigations: provide a scoring rubric and reference answers, ensemble multiple judges, check agreement against human labels.
- Follow-up: watch for **data contamination** and benchmark overfitting; Arena-style scores can also be gamed by "style" (hence the style-controlled versions).

## Long-Chain Reasoning and Test-Time Compute

Q23: ***How are long chain-of-thought (long CoT) reasoning models trained?***

- Core: have the model generate a long **thinking process** (exploration, backtracking, self-verification) before giving the answer, and reinforce this behavior via RL.
- Common pipeline:
    1. **Cold-start SFT**: a small amount of high-quality long-CoT data (curated by humans or distilled from a strong reasoning model) to fix the format (e.g. `<think>...</think>` followed by the final answer) and improve readability;
    2. **Large-scale RLVR**: GRPO/PPO with rule-based rewards on math, code, science and logic problems; filter problems by difficulty (problems that are too easy or too hard give no learning signal);
    3. Rejection sampling + general SFT, mixing in non-reasoning data to retain general ability;
    4. General RL to align preferences and safety.
- Phenomena and tricks in training: response length grows over training; prevent **entropy collapse** (clip-higher, entropy regularization); penalize overly long responses or use a length curriculum (gradually enlarging the context length); language mixing can be handled with a language-consistency reward.
- Small models: usually gain long-reasoning ability by **distilling** trajectories from large reasoning models, which is more efficient than direct RL.

Q24: ***What is test-time compute scaling? What approaches exist?***

- Idea: besides scaling training compute, spend more compute **at inference** to get better results; o1 showed performance improving log-linearly with "thinking length".
- Approaches:
    - **Sequential scaling**: longer chains of thought (reasoning models), self-correction, multi-round reflection; controllable via a "thinking budget" (e.g. budget forcing, which forcibly truncates or extends thinking by appending "Wait", s1 2025);
    - **Parallel scaling**: multiple samples + **majority voting (self-consistency)**, best-of-N (selected by an ORM/PRM/verifier);
    - **Search**: PRM-guided beam search, MCTS, tree-of-thoughts, etc.
- Trade-offs:
    - Overthinking on easy problems wastes compute, while underthinking on hard problems hurts; **adaptive compute** is needed (allocating budget by difficulty, or hybrid thinking/non-thinking modes);
    - Parallel methods depend on reliable verifiers; majority voting only works for tasks with comparable answers;
    - Compute-optimal studies show that under a given budget, a small model + more test-time compute can sometimes beat a larger model, but the hardest problems still require a stronger base model.

Q25: ***If you were to post-train starting from a base model, how would you design the experiments and iteration loop?***

- **Data**: define target capabilities and a behavior spec -> collect/synthesize SFT data (diversity, quality filtering, dedup against eval sets) -> preference data (human + AI feedback) -> a problem bank with verifiers for verifiable tasks.
- **Training**: SFT (correct chat template, loss masking, few epochs) -> preference optimization (DPO for fast iteration, or RM + PPO/GRPO online RL) -> RLVR to improve reasoning -> safety tuning.
- **Evaluation**: build an **internal eval suite** covering knowledge, reasoning, code, instruction following, dialogue quality, safety/false refusals and long context; combine LLM judges with human spot checks, and monitor the alignment tax.
- **Iteration**: analyze failure cases -> add targeted data -> retrain; each round, resample preference data and update the RM with the latest model (on-policy iteration), and do checkpoint selection or model merging.
- Ablation focus: data mixture ratios, KL coefficient $\beta$, learning rate, sampling temperature and group size, reward length debiasing.
