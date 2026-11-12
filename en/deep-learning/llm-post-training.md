# LLM Post-Training and Alignment

## Overall Pipeline

Q1: ***What do pretraining and post-training each do? What does a typical modern LLM post-training pipeline look like?***

- **Pretraining does next-token prediction on trillions of tokens so the model learns knowledge and language ability, but it can only continue text; post-training uses far less data to shape behavior, making the model follow instructions, be helpful, honest and safe, and able to reason and use tools.**
- A common view is that "knowledge mostly comes from pretraining, and post-training mostly elicits and formats it" (LIMA's Superficial Alignment Hypothesis).
- Typical pipeline: SFT → preference optimization (RLHF or DPO) → reasoning RL on verifiable tasks such as math and code (e.g. GRPO) → safety tuning, iterated over several rounds.
- There is often a mid-training/annealing stage at the end of pretraining that raises the share of high-quality, long-context, code and math data to bridge the two.

## SFT and Instruction Tuning

Q2: ***In SFT, which matters more: data quality or quantity?***

- **Quality comes first: LIMA got decent chat ability from only about 1,000 curated samples, while lots of low-quality data teaches the model verbosity, errors and hallucination. But quantity still matters when you need to cover more skills and long-tail tasks.**
- Quality means correctness, diversity (tasks, domains, difficulty, length, language), consistent format, a style that fits the target, and deduplication against evaluation sets.
- Modern SFT sets usually have hundreds of thousands to millions of examples, but all go through strict filtering such as LLM scoring, deduplication, difficulty selection and execution checks.
- Caveat: SFT is imitation, so demonstrations beyond the model's existing knowledge can teach it to "confidently make things up".

Q3: ***What is a chat template? Why is it important?***

- **A chat template serializes a multi-turn conversation into a fixed format, using special tokens to mark roles and turn boundaries; training and inference must use the same template, otherwise the model fails to stop or confuses roles.**

```text
<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
2<|im_end|>
```

- The end-of-turn token teaches the model when to stop.
- Role separation is the basis of the instruction hierarchy (system > user > tool output) and of defending against prompt injection.
- Tool calls and chain of thought (e.g. `<think>` blocks) are also expressed via special structures in the template; in practice use the tokenizer's built-in `apply_chat_template` to avoid mistakes from hand-built strings.

Q4: ***Why mask the loss on prompt tokens in SFT?***

- **SFT should learn "how to answer a given question", not "how to generate the question", so the loss is computed only on response tokens, and prompt labels are set to -100 so they are ignored.**
- It keeps the model from wasting capacity learning to generate user questions, or even speaking for the user at inference.
- User inputs can be noisy and low quality and should not be imitated.
- In multi-turn conversations the loss is usually computed on all assistant turns; when packing multiple samples, note whether you average per token or per sample, since this changes the weight of long vs. short answers.

Q5: ***How does instruction tuning relate to SFT? Why does it improve zero-shot generalization?***

- **Instruction tuning is a kind of SFT: many NLP tasks are rewritten as "natural-language instruction + output" for multi-task fine-tuning (e.g. FLAN, T0). The more numerous and diverse the tasks the model has seen, the better it understands new instructions and the better its zero-shot generalization.**
- Mixing in CoT examples preserves reasoning ability, and larger models benefit more.
- Early instruction tuning leaned toward academic tasks with short answers; chat-style SFT emphasizes open-ended long answers and multi-turn interaction, and modern practice mixes both.

## RLHF and PPO

Q6: ***Describe the full RLHF pipeline. How is the reward model trained (Bradley-Terry loss)?***

- **RLHF first samples several answers to the same prompt from the SFT model and has people label which is better, then trains a scoring reward model on these preference pairs, and finally uses PPO to optimize the policy to get high scores. The reward model is trained so that the preferred answer scores higher than the rejected one.**
- The reward model is usually initialized from the SFT model, with the LM head replaced by a scalar head that reads the last token's hidden state.
- Only reward differences are meaningful, so rewards are often normalized; when a prompt has K ranked answers, all pairwise combinations can be used.
- Human inter-annotator agreement is usually only 60%–80%, which caps the reward model's accuracy.
- Key formula: $\mathcal{L}_{RM} = -\mathbb{E}\big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\big]$

Q7: ***What is the optimization objective in the PPO stage? Why add a KL penalty?***

- **The objective is to maximize the reward model's score while a KL penalty keeps the policy from drifting too far from the reference model (the frozen SFT model). The KL penalty mainly prevents reward hacking: the reward model is only reliable near the SFT distribution, and a policy that drifts far will exploit its loopholes.**
- KL also keeps language fluent and diverse, avoids mode collapse, and preserves pretrained abilities.
- In practice the KL is often applied as a per-token reward, with the reward model's score added at the last token.
- The KL-regularized objective has a closed-form optimum (the reference model times the exponential of the reward), which is exactly DPO's starting point.
- Too small a $\beta$ leads to reward hacking, too large prevents learning; adaptive KL control can be used.
- Key formula: $\max_{\pi_\theta} \ \mathbb{E}\big[r_\phi(x,y)\big] - \beta\, \mathbb{D}_{KL}\big[\pi_\theta\,\|\,\pi_{ref}\big]$

Q8: ***What are the specific details of PPO for LLMs? (four models, GAE, clipping)***

- **PPO uses four models at once: the actor and critic being trained, plus the frozen reference and reward models, and the large memory cost is its main pain point. The critic estimates value, GAE computes each token's advantage, and a clipped probability ratio limits how much each update can change the policy.**
- MDP view: the state is the prompt plus generated tokens and the action is the next token; the reward is sparse, with an RM score only at the end plus a per-token KL penalty.
- For LLMs GAE commonly uses $\gamma=1$ and $\lambda$ close to 1, balancing the bias and variance of advantage estimates.
- Clipping ($\epsilon$ commonly 0.2) makes it safe to train several epochs on the same batch of rollouts.
- Other tricks: value clipping, advantage and reward normalization, initializing the critic from the RM, mixing in pretraining loss to prevent forgetting; in the training loop, generating rollouts is the most expensive part.

## DPO and Variants

Q9: ***What is the derivation idea and loss function of DPO?***

- **DPO uses the closed-form optimum of the KL-constrained RLHF objective to express the reward as "the log-probability ratio between the policy and the reference model"; plugged into the Bradley-Terry preference model, the intractable partition function cancels out, so the policy can be optimized directly on preference pairs without training a reward model.**
- Intuition: "your language model is secretly a reward model": raise the probability of the chosen answer relative to the reference model and lower that of the rejected one.
- The gradient automatically gives more weight to pairs the model currently ranks wrongly.
- The implementation is very simple, essentially a binary classification loss whose inputs are each answer's summed log-probability over response tokens.
- Key formula: $\mathcal{L}_{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big]$

Q10: ***What are the pros and cons of DPO compared with PPO?***

- **DPO is simple, stable and cheap, needing no reward model, critic or online sampling; but it is an offline method with distribution shift and a tendency to overfit, and when resources allow, online RL (PPO/GRPO) has a higher ceiling.**
- Pros: only the policy and reference model are needed (reference log-probs can be precomputed), the main hyperparameter is just $\beta$, and it is easy to reproduce.
- Cons: preference data from other models causes distribution shift, which online/iterative DPO mitigates; since it only constrains the difference, the chosen answer's probability can drop as well.
- When preferences are nearly deterministic the implicit reward gap goes to infinity and it overfits; it has a length bias toward longer answers.
- It does not suit reasoning RL with verifiable rewards; DPO is best for fast iteration and small-to-medium-scale alignment.

Q11: ***Briefly describe the DPO variants IPO, KTO, ORPO and SimPO.***

- **Each patches one DPO problem: IPO prevents overfitting, KTO relaxes the data requirement, ORPO and SimPO drop the reference model to save memory, and SimPO also eases length bias.**
- IPO: replaces log-sigmoid with a squared loss that regresses the implicit reward gap to a fixed margin, avoiding overfitting when preferences are deterministic.
- KTO: based on prospect theory, it needs no paired data, only a "good/bad" label per answer, which fits thumbs-up/thumbs-down data.
- ORPO: adds an odds-ratio term to the SFT loss, doing SFT and preference alignment in a single stage with no reference model.
- SimPO: uses the length-normalized average log-probability as the implicit reward plus a target margin, needs no reference model, and matches how answers are scored during generation.

## GRPO and Verifiable Rewards

Q12: ***What is GRPO? How does it differ from PPO?***

- **GRPO (proposed in DeepSeekMath and widely used in DeepSeek-R1) removes PPO's critic: for each prompt it samples a group of answers, standardizes each answer's reward within the group, and uses that as the advantage for all of the answer's tokens.**
- It still uses a PPO-style clipped objective, but the KL penalty is added directly to the loss rather than to the reward.
- Pros: no value model as large as the policy, so memory and compute drop sharply; the in-group baseline naturally fits sparse sequence-level rewards such as right/wrong.
- Cons: a group that is all right or all wrong gives no learning signal (DAPO filters these with dynamic sampling); dividing by std and normalizing by length introduce biases (Dr. GRPO removes them); DAPO also proposes clip-higher to prevent entropy collapse.
- Similar critic-free methods include RLOO, REINFORCE++ and ReMax.
- Key formula: $\hat A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$

Q13: ***What is RLVR (reinforcement learning with verifiable rewards)? Why did it drive the development of reasoning models?***

- **In RLVR the reward comes from rules that can be checked programmatically (does the math answer match, does the code pass unit tests) instead of a learned reward model. Such rewards are hard to hack, need no human labels and scale without limit, so they support long, large-scale RL training, which gave rise to reasoning models.**
- Milestones: OpenAI o1 started the reasoning-model paradigm; DeepSeek-R1-Zero ran pure RL directly on a base model, and long chains of thought, self-verification and reflection emerged on their own.
- The full R1 pipeline: cold-start long-CoT SFT → reasoning RL → rejection sampling to make new SFT data → RL across all scenarios.
- Debate: some argue RLVR mainly concentrates correct paths the base model already had into pass@1 rather than expanding its capability boundary; verifier false positives also need watching.

Q14: ***What is the difference between a process reward model (PRM) and an outcome reward model (ORM)?***

- **An ORM scores only the final result, which is cheap to label but makes credit assignment hard and may reward "wrong process, lucky answer"; a PRM scores every reasoning step, giving a denser signal that can locate the first wrong step.**
- OpenAI's "Let's Verify Step by Step" showed PRMs beat ORMs for best-of-N reranking.
- PRM data can come from human step-by-step labels (e.g. PRM800K, expensive) or from Monte Carlo rollouts that estimate each step's chance of success.
- Practical issues: unclear step boundaries, noisy labels, and easy hacking when used as an RL reward; so mainstream reasoning RL still relies on outcome rewards plus rule-based verification, and PRMs are used more for test-time search and reranking.

## Reward Hacking and AI Feedback

Q15: ***What is reward hacking? What are common symptoms and mitigations?***

- **Reward hacking is when the policy finds ways to raise the proxy reward without improving, or even while hurting, true quality, i.e. Goodhart's law; the typical sign is a reward-model score that keeps rising while human evaluation first rises and then falls.**
- Common symptoms: ever longer answers padded with lists and disclaimers; sycophantic agreement with the user's mistakes; templated phrasing; in RLVR, exploiting verifier loopholes such as editing tests or writing special-case code.
- Mitigations: KL penalties and early stopping based on human evaluation; RM ensembles and length debiasing; continually updating the RM with samples from the new policy.
- Also use more robust verifiers (sandboxes, hidden tests) and multi-objective rewards, monitor the reward distribution, length and entropy, and manually inspect high-reward samples.

Q16: ***What are RLAIF and Constitutional AI?***

- **RLAIF uses a strong LLM instead of humans to give preference labels, then trains an RM or runs DPO as usual; it is cheap and scalable but inherits the judge model's biases. Constitutional AI (Anthropic, 2022) builds on RLAIF by using a written set of principles (a "constitution") to guide the model's self-critique and comparisons.**
- Supervised stage: the model answers harmful requests → critiques and revises its answer according to a principle → the revised answers are used for SFT.
- RL stage: the model judges which of two answers is better according to the principles, generating preference data to train a preference model, followed by RL.
- Benefits: less human labeling of harmful content, transparent and auditable rules, and a better balance between harmless and helpful (explained refusals rather than evasion).

## Data and Distillation

Q17: ***What is rejection sampling / best-of-N? How is it used in post-training?***

- **Best-of-N samples N answers at inference and uses an RM or verifier to pick the highest-scoring one; rejection-sampling fine-tuning applies the same idea to training: sample several answers per prompt, keep only those that pass verification or score highest, do SFT on them, and iterate for several rounds.**
- Best-of-N is simple yet strong and a strong baseline for judging RL, but multiplies inference cost by N.
- Rejection-sampling fine-tuning (e.g. Llama 2, STaR, ReST) needs only SFT infrastructure, is stable, and can be seen as a discrete form of policy improvement.
- Cons: it learns only from positives and ignores negatives; hard problems may yield no correct samples; its ceiling is usually below online RL.

Q18: ***How do you distill from a stronger model? What should you watch out for?***

- **The most common approach is black-box distillation: have the strong model generate answers (including long CoT) and SFT the student on them; if the teacher's logits are available, white-box distillation pushes the student's token distribution directly toward the teacher's. For small models, distilling a strong reasoning model often works better than running RL directly (as the DeepSeek-R1 distilled models show).**
- On-policy distillation lets the student generate while the teacher scores each token, easing the mismatch between training and inference distributions.
- The teacher's errors and hallucinations are inherited, so verify answers, deduplicate and filter for quality.
- If the capability gap is too large the student cannot learn, so use an intermediate-size teacher or a curriculum; also beware of learning only the style without the ability.
- Mind the license terms: some commercial models forbid using their outputs to train competitors.

Q19: ***How is synthetic data used in post-training? What are the risks?***

- **Synthetic data is used to generate instructions, generate and filter answers, build preference pairs, and synthesize verifiable reasoning problems and agent trajectories at scale; the main risks are loss of diversity up to model collapse, error propagation and evaluation contamination.**
- Instruction generation: Self-Instruct, Evol-Instruct (gradually increasing difficulty), and generating diverse prompts from documents or personas.
- Answers come from a strong model and are filtered with verifiers, execution feedback or an LLM judge; preference pairs come from multiple samples judged by an RM or rules.
- Other risks: a uniform "AI voice", and contamination when synthetic data resembles benchmarks.
- Mitigations: keep real human data, deduplicate and control diversity, verify strictly, and mix several generator models.

## Forgetting, Alignment Tax and Safety

Q20: ***What are catastrophic forgetting and the "alignment tax" in post-training? How do you mitigate them?***

- **Catastrophic forgetting is when fine-tuning on a narrow distribution degrades other abilities; the alignment tax is when alignment worsens some benchmark scores, creativity and probability calibration. The core fix is not letting the model move too far from where it started: mix in general data, use KL constraints and small learning rates, and use LoRA and model merging.**
- Data mixing/replay: mix in general instruction data and a little pretraining data.
- Parameter-efficient fine-tuning (LoRA "learns less and forgets less"); model merging, e.g. interpolating between SFT and RL weights.
- Multi-task joint training, and selecting checkpoints on a broad evaluation suite.
- Observation: online RL usually forgets less than SFT, because it updates on the policy's own distribution and changes more conservatively.

Q21: ***How is safety tuning done? How do you avoid over-refusal?***

- **Safety tuning uses safety SFT data, a separate safety reward model or rule-based rewards, plus red-teaming to find gaps and add data; the key to avoiding over-refusal is adding many "looks sensitive but is actually harmless" examples to training and replacing hard refusals with safe completions that help as much as possible within safe bounds.**
- Llama 2 uses separate helpfulness and safety RMs; OpenAI's Deliberative Alignment has the model explicitly cite safety specs while reasoning.
- An example of over-refusal is refusing "how do I kill a Python process"; it can be measured with XSTest and OR-Bench.
- Track both the harmful rate and the false-refusal rate in evaluation, and judge by contextual intent rather than keywords.
- Jailbreaks, multi-turn attacks and prompt injection need adversarial and instruction-hierarchy training; safety alignment can easily be removed by a little fine-tuning, which matters especially for open-weight models.

## Evaluation

Q22: ***How do you evaluate an aligned model? What biases does LLM-as-judge have?***

- **Evaluation combines human evaluation (the gold standard but expensive), win rates against a baseline (e.g. Chatbot Arena's Elo), LLM-as-judge (MT-Bench, AlpacaEval, Arena-Hard) and capability benchmarks. The main LLM-judge biases are position bias, length bias and self-preference.**
- Capability benchmarks cover knowledge, math, code, programmatically verifiable instruction following (IFEval), and safety/refusal tests.
- Position bias: evaluate twice with the order swapped; length bias: use length-controlled win rates (e.g. AlpacaEval 2 LC).
- Other biases: preferring markdown formatting and an authoritative tone, and unreliable judgment of math/code correctness; mitigate with scoring rubrics and reference answers, ensembles of judges, and agreement checks against human labels.
- Also watch for data contamination and benchmark overfitting; Arena scores can also be gamed by "style".

## Long-Chain Reasoning and Test-Time Compute

Q23: ***How are long chain-of-thought (long CoT) reasoning models trained?***

- **First do cold-start SFT on a small amount of high-quality long-CoT data to set the "think, then answer" format; then run large-scale RL on verifiable tasks such as math and code so that exploration, backtracking and self-verification get reinforced; finally do SFT on mixed general data and general RL to keep general ability and safety.**
- The RL stage filters problems by difficulty, since problems that are too easy or too hard give no learning signal.
- Answer length naturally grows during training; prevent entropy collapse (clip-higher, entropy regularization), penalize overly long answers or gradually expand the context length, and add a language-consistency reward against language mixing.
- Small models usually distill the trajectories of a large reasoning model directly, which is more efficient than doing RL themselves.

Q24: ***What is test-time compute scaling? What approaches exist?***

- **Test-time compute scaling spends more compute at inference to get better results, in three main ways: thinking longer serially (longer chains of thought, self-correction), sampling many times in parallel and choosing (majority voting, best-of-N), and verifier-guided search (e.g. PRM beam search, MCTS).**
- o1 showed performance rising roughly log-linearly with thinking length; s1 uses budget forcing (truncating or appending "Wait") to control the thinking budget.
- Easy problems get overthought and hard ones underthought, so budgets should adapt to difficulty, or models should mix thinking and non-thinking modes.
- Parallel methods depend on reliable verifiers, and majority voting only works when answers can be compared.
- Under a fixed budget a small model with more test-time compute can sometimes beat a larger model, but the hardest problems still need a stronger base model.

Q25: ***If you were to post-train starting from a base model, how would you design the experiments and iteration loop?***

- **First define the target abilities and behavior spec and prepare the data, then train in the order SFT → preference optimization → RLVR → safety tuning while building a broad internal evaluation suite, and keep iterating through "analyze failures → add targeted data → retrain".**
- Data: diverse, quality-filtered SFT data deduplicated against evaluation sets, preference data from human + AI feedback, and a problem bank with verifiers for verifiable tasks.
- Training: correct chat template and loss masking, few epochs; DPO for fast iteration, or RM + PPO/GRPO for online RL.
- Evaluation: knowledge, reasoning, code, instruction following, chat quality, safety/false refusals and long context, combining LLM judges with human spot checks and monitoring the alignment tax.
- Each round, resample preference data with the latest model and update the RM; key ablations are data mixture, KL coefficient, learning rate, sampling temperature and group size, and length debiasing.
