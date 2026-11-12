# Statistics and A/B Testing

## Hypothesis Testing Basics

Q1: ***Describe the general framework of hypothesis testing***

- **Hypothesis testing is a proof by contradiction: assume there is no effect (the null hypothesis $H_0$), then ask how unusual the observed data would be under that assumption; if unusual enough, reject $H_0$.**
- Steps: state $H_0$/$H_1$ and choose one- or two-sided → fix the significance level $\alpha$ in advance (commonly 0.05) → pick a test statistic (z, t, chi-square) → compute the p-value → reject $H_0$ if $p<\alpha$.
- A non-significant result only means "fail to reject $H_0$", not "accept $H_0$".
- $\alpha$, the sample size and the test must be decided before looking at the data; otherwise it is p-hacking.

Q2: ***What does a p-value actually mean? What are the common misreadings?***

- **The p-value is the probability, assuming $H_0$ is true, of seeing data at least as extreme as what was observed. It measures how incompatible the data are with $H_0$; it is not the probability that $H_0$ is true.**
- Misreading 1: the p-value is the probability that $H_0$ is true -- that is $P(H_0\mid\text{data})$, which needs a prior.
- Misreading 2: a smaller p means a bigger effect -- the p-value depends on both effect size and sample size; with enough data a tiny effect is significant, so always report the effect size and confidence interval too.
- Misreading 3: $p>0.05$ means no difference -- the test may simply be underpowered.
- Under $H_0$ the p-value is uniform on $[0,1]$, which is exactly why repeated peeking and multiple comparisons create false positives.

Q3: ***Explain Type I error, Type II error and statistical power***

- **A Type I error is calling an effect when there is none (false positive, probability $\alpha$); a Type II error is missing an effect that exists (false negative, probability $\beta$); power $1-\beta$ is the probability of detecting a real effect.**

| | $H_0$ true | $H_1$ true |
|---|---|---|
| Reject $H_0$ | Type I error ($\alpha$) | Correct (power $1-\beta$) |
| Do not reject $H_0$ | Correct | Type II error ($\beta$) |

- A/B tests usually require 80% power.
- Larger effects, more samples, lower variance and a looser $\alpha$ all increase power.
- At a fixed sample size $\alpha$ and $\beta$ trade off; reducing both requires more samples or lower variance.

Q4: ***How should a confidence interval be interpreted? How does it relate to hypothesis testing?***

- **A 95% confidence interval means: if we repeated the sampling many times and built an interval the same way each time, about 95% of those intervals would cover the true parameter. It does not mean "the true value has a 95% chance of lying in this interval" -- that is the Bayesian credible interval.**
- In the frequentist view the parameter is fixed; a specific interval either contains it or it does not.
- Duality: a two-sided test is significant at level $\alpha$ exactly when the $1-\alpha$ confidence interval excludes 0.
- A confidence interval carries more information than a single p-value: it shows both how big the effect is and how uncertain it is.

Q5: ***What is the Central Limit Theorem (CLT)? Why does it matter for A/B testing?***

- **The CLT says that for i.i.d. samples with finite variance, the sample mean is approximately normal once the sample is large enough. So even if metrics like revenue or time spent are highly skewed, their difference in means can still be analyzed with z/t tests and confidence intervals.**
- Heavy-tailed metrics (a few "whale" users) converge slowly and need larger samples, or capping / log transforms.
- The CLT requires independent observations; randomizing by user but analyzing by session/page view makes observations correlated, so use the delta method or cluster-robust standard errors.
- The CLT fails when the variance is infinite (e.g. the Cauchy distribution).

## Common Tests

Q6: ***What is the difference between a z-test and a t-test? What is Welch's t-test?***

- **A z-test is used when the variance is known or the sample is large, and the statistic is normal; a t-test estimates the variance from the sample, so the statistic follows a heavier-tailed t distribution. It is more accurate for small samples, and the two are nearly identical for large ones.**
- Proportion metrics such as conversion rate commonly use the two-proportion z-test.
- Student's t-test assumes equal variances in both groups; Welch's t-test drops that assumption and corrects the degrees of freedom.
- In practice default to Welch: it loses almost no power when variances are equal and is more robust when they are not.
- Paired t-test: for before/after measurements on the same subjects, run a one-sample t-test on the differences.
- Key formula (Welch): $t=\dfrac{\bar x_B-\bar x_A}{\sqrt{s_A^2/n_A+s_B^2/n_B}}$

Q7: ***When is the chi-square test used?***

- **The chi-square test is for categorical data: a test of independence (e.g. is "group × converted" independent?) and a goodness-of-fit test (do observed counts match expected proportions? The SRM check in A/B testing is this case).**
- It compares observed and expected counts; the bigger the gap, the bigger the statistic.
- When expected counts are small (usually below 5), use Fisher's exact test instead.
- For a 2×2 table the chi-square test is equivalent to the two-sided two-proportion z-test ($\chi^2=z^2$).
- Key formula: $\chi^2=\sum\frac{(O_i-E_i)^2}{E_i}$

Q8: ***When would you use the Mann-Whitney U test?***

- **When the sample is small, strongly skewed, has outliers or is ordinal, so normality is a bad assumption, use the Mann-Whitney U test to compare two groups. It pools and ranks both groups and compares rank sums; it is a non-parametric test.**
- It tests whether one group tends to be larger than the other; it can be read as a median difference only when both distributions have the same shape.
- Downside: it does not tell you "how much the mean went up", while the business usually cares about means.
- With large samples the t-test is already robust thanks to the CLT; other alternatives are the bootstrap or a t-test after a log transform.

Q9: ***What is the bootstrap? When is it used?***

- **The bootstrap treats the sample as the population: resample it with replacement, at the same size, thousands of times, compute the statistic each time, and use the distribution of those results to estimate the standard error and confidence interval.**
- It is used for statistics with no simple analytic variance, such as medians, quantiles, ratio metrics and AUC.
- The simplest interval takes the 2.5% and 97.5% quantiles of the bootstrap results (percentile interval); BCa is more accurate.
- Resample by the randomization unit (by user, not by event), otherwise the variance is underestimated.
- It works poorly for extreme quantiles, tiny samples and heavy tails; it is also expensive, so large platforms often use the delta method instead.

## A/B Test Design

Q10: ***How do you choose metrics for an A/B test? What are north-star and guardrail metrics?***

- **The north-star metric (OEC) is the core metric reflecting long-term product value, such as retention or engaged time per user; guardrail metrics are metrics that must not get worse. Usually a short-term, measurable proxy of the north star is chosen as the primary decision metric for the experiment.**
- Guardrails come in two kinds: business guardrails (revenue, retention, complaint rate) and trust guardrails (SRM, load latency, crash rate, which check that the experiment itself is reliable).
- A good metric is sensitive (moves within the experiment window), aligned with long-term value, low-variance and hard to game; evaluate it with past experiments or A/A tests.
- When short-term metrics conflict with long-term value (e.g. more ads raise revenue but hurt retention), calibrate with a long-term holdout.

Q11: ***How do you choose the randomization unit?***

- **Usually randomize by user so each user gets a consistent experience. The rule is that the randomization unit should be the same as or coarser than the analysis unit, and when users influence each other, switch to a coarser unit (region, cluster, time slice).**
- Randomizing by user but computing CTR per page view makes observations dependent; the naive variance is too small, so use the delta method or a user-level bootstrap.
- Finer units (session, page view) give more samples and more power, but may create inconsistent experiences and leakage.
- Logged-out users can only be split by cookie/device, which has the one-person-many-devices problem.

Q12: ***How do you calculate the sample size for an A/B test? What is MDE?***

- **The MDE (minimum detectable effect) is the smallest effect the experiment can reliably detect at a given $\alpha$ and power; the business should set it as "the smallest lift worth shipping". Sample size is proportional to the metric variance and inversely proportional to the square of the MDE.**
- Halving the MDE quadruples the sample size, which is why variance reduction (e.g. CUPED) directly shortens experiments.
- For a proportion metric the variance is $p(1-p)$. Example: a 10% baseline conversion rate and a 0.5 percentage-point absolute lift need about 58,000 users per group.
- Key formula (rule of thumb for $\alpha=0.05$, 80% power, per group): $n\approx\dfrac{16\sigma^2}{\delta^2}$

Q13: ***How long should an experiment run? What are novelty and primacy effects?***

- **Duration is roughly "required sample size ÷ daily traffic", but it should cover at least full weekly cycles (usually 1–2 weeks or more), avoid holidays and big promotions, and run long enough for novelty/primacy effects to fade.**
- Novelty effect: users try a new feature more out of curiosity, then it wears off, so the effect is overestimated.
- Primacy effect (change aversion): existing users dislike the new version at first and recover after adapting, so the effect is underestimated.
- Detection: plot the treatment effect over time; look at new vs. existing users separately (new users are unaffected); analyze cohorts by first-exposure day; use a long-term holdout if needed.

Q14: ***What is the "peeking" problem? How does sequential testing solve it?***

- **Peeking means checking the p-value repeatedly during the experiment and stopping as soon as it is significant. Under $H_0$ the p-value wanders randomly, so the more you look, the more likely you hit $p<0.05$ once, and the real false positive rate is far above 5%.**
- Simplest fix: compute the sample size up front and only look at the end.
- Group sequential tests: pre-plan a few interim analyses and allocate $\alpha$ with a spending function (e.g. O'Brien-Fleming, very strict early and close to the usual threshold at the end).
- Always-valid inference (mSPRT, confidence sequences): look any time while keeping the error rate controlled, at the cost of slightly lower power.
- Bayesian methods do not escape this either: stopping early on a posterior threshold also inflates the false positive rate.

Q15: ***What is the multiple comparisons problem? How do Bonferroni and BH differ?***

- **When testing many hypotheses at once (many metrics, arms or segments), the chance of at least one false positive grows fast (about 64% for 20 tests). Bonferroni controls the probability of any false positive (FWER) and is strict; BH controls the share of false positives among significant results (FDR), which is looser and has more power.**
- Bonferroni: use threshold $\alpha/m$ for each test; simple but very conservative when $m$ is large. Holm's step-down method is a uniformly better refinement.
- BH: sort p-values ascending, find the largest $k$ whose p-value is at most $\frac{k}{m}q$, and reject the first $k$.
- Use FWER for critical launch decisions; use FDR for exploratory screening of many metrics/segments.
- In practice, pre-register a single primary metric and treat others as secondary; segment findings need replication.

Q16: ***What is SRM? How do you check for it?***

- **SRM (sample ratio mismatch) is when the actual group sizes deviate significantly from the designed split, e.g. a 50/50 design that shows 50.8/49.2. It means the assignment or data pipeline is broken and the results cannot be trusted, so debug first before reading the effect.**
- Check with a chi-square goodness-of-fit test, usually alerting only at a strict threshold (e.g. $p<0.001$).
- Common causes: assignment bugs or hashing problems, a slower treatment page losing tracking events, bot filtering that differs between groups, different trigger conditions, changing the traffic split mid-experiment.

Q17: ***What are network effects / interference? How do you handle them?***

- **A/B testing assumes one user's outcome is not affected by other users' assignments (SUTVA). In social networks and two-sided marketplaces (ride-hailing, food delivery, ad auctions) this breaks: treatment affects control or grabs shared resources, so the effect is diluted or overestimated.**
- Cluster randomization: split by social-graph clusters or by geographic region.
- Switchback experiments: toggle treatment on and off for a whole region by time slice; common in two-sided marketplaces.
- Resource isolation: give each group its own budget or supply pool.
- Cost: fewer randomization units, so higher variance and lower power.

Q18: ***What is CUPED? Why does it reduce variance?***

- **CUPED uses pre-experiment data (usually the same metric measured before the experiment) to explain away the part of the metric's variation that has nothing to do with the treatment, which lowers variance and shortens experiments. Because pre-experiment data are not affected by treatment, both groups have the same expectation and the adjustment introduces no bias.**
- Variance shrinks by a factor of $1-\rho^2$, where $\rho$ is the correlation between the pre- and in-experiment metric; at $\rho=0.7$ the variance roughly halves, almost like doubling the sample.
- It is essentially regression adjustment with a pre-treatment covariate; CUPAC uses an ML model's prediction as the covariate.
- The covariate must be unaffected by treatment; for new users without history, use other pre-experiment features.
- Key formula: $Y_{cuped}=Y-\theta(X-\bar X)$, $\theta=\mathrm{Cov}(X,Y)/\mathrm{Var}(X)$

Q19: ***What are ratio metrics? Why do we need the delta method?***

- **In a ratio metric both numerator and denominator are random, e.g. CTR = total clicks / total impressions. With user-level randomization, one user's many impressions are correlated, so the plain binomial variance underestimates the variance and inflates false positives; the delta method estimates the variance correctly at the user level.**
- Idea of the delta method: take a first-order Taylor expansion of the ratio and approximate its variance from the user-level variances and covariance of numerator and denominator.
- It is also used for confidence intervals on relative lift.
- Alternative: a user-level bootstrap.

Q20: ***What is Simpson's paradox? How does it show up in A/B testing?***

- **Simpson's paradox is when B beats A in every subgroup, yet A beats B once the groups are pooled. It happens when there is a confounder and the two arms have different mixes across subgroups.**
- Classic example: kidney stone treatment A is better for both small and large stones, but it was given mostly to hard large-stone cases, so its pooled cure rate is lower.
- In A/B tests it typically appears when the traffic split changes mid-experiment (with different daily baselines), when segmenting by a variable affected by treatment, or with uneven allocation across platforms.
- Remedies: keep the traffic split fixed, or analyze by stratum and reweight; segment only by pre-experiment attributes.

## Causal Inference and Bayesian Methods

Q21: ***What is confounding? How do propensity score methods work?***

- **A confounder affects both treatment and outcome, so correlation is not causation. Randomized experiments cut confounding by random assignment; with observational data, the propensity score (the probability of receiving treatment given features) can be used to make treated and control groups comparable.**
- Uses: propensity score matching (PSM), inverse probability weighting (IPW, weighting treated units by $1/e(x)$), stratification; doubly robust AIPW combines an outcome model and a propensity model and is consistent if either is correct.
- Assumptions: no unobserved confounding, overlap in features between groups, SUTVA.
- Limits: cannot handle unobserved confounders; scores near 0 or 1 make weights explode and need trimming; always check covariate balance after adjustment.

Q22: ***Explain Difference-in-Differences (DiD)***

- **DiD is for cases where you cannot randomize but have treated and control groups observed before and after treatment: take the treated group's before-after change minus the control group's before-after change. The first difference removes fixed group differences, the second removes the shared time trend, and what remains is the treatment effect.**
- The key assumption is parallel trends: without treatment both groups would have moved the same way; test it with an event study on multiple pre-periods.
- In regression form, the treatment effect is the coefficient on the "treated × post" interaction.
- With staggered treatment timing, classic two-way fixed effects can be biased and newer estimators (e.g. Callaway–Sant'Anna) are needed; if parallel trends fail, consider synthetic control.
- Key formula: $\hat\tau=(\bar Y_{T,post}-\bar Y_{T,pre})-(\bar Y_{C,post}-\bar Y_{C,pre})$

Q23: ***What is uplift modeling? How does it differ from a standard response model?***

- **A standard response model predicts who will buy; an uplift model predicts who will buy because of the intervention, i.e. the individual treatment effect. Marketing should target the latter, not people who would buy anyway.**
- Four user types: persuadables (buy only if treated, the target), sure things (always buy, wasted spend), lost causes (never buy), sleeping dogs (treatment makes them churn, avoid).
- Methods: S-learner (treatment as a feature), T-learner (one model per group, then subtract), X-learner (good when groups are imbalanced), causal forests and others.
- Training data should ideally come from a randomized experiment; true individual uplift is unobservable, so ordinary AUC does not work -- use Qini / uplift curves.

Q24: ***What is the difference between Bayesian and frequentist A/B testing?***

- **Frequentists treat the parameter as fixed and output p-values and confidence intervals; Bayesians treat it as a random variable with a prior and directly output "the probability that B beats A" and the expected loss, which is more intuitive but depends on the prior.**

| | Frequentist | Bayesian |
|---|---|---|
| Output | p-value, confidence interval | posterior, $P(B>A)$, expected loss |
| Interpretation | indirect | intuitive |
| Prior | not needed | needed, matters most with small samples |

- Conversion rates commonly use the Beta-Binomial conjugate pair: the posterior is still a Beta distribution, and sampling gives $P(B>A)$ and the expected loss.
- Misconception: Bayesian tests cannot be peeked at freely either -- "stop once $P(B>A)>95\%$" also inflates the false positive rate.
- Large-traffic platforms mostly use frequentist tests + sequential testing + variance reduction; Bayesian methods suit low traffic, intuitive decision-making or incorporating priors from past experiments.
