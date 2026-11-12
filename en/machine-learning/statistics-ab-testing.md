# Statistics and A/B Testing

## Hypothesis Testing Basics

Q1: ***Describe the general framework of hypothesis testing***

- **Null hypothesis** $H_0$ (usually "no effect / no difference") vs. **alternative hypothesis** $H_1$. The logic is proof by contradiction: assume $H_0$ is true and ask how "unusual" the observed data would be.
- General steps:
    1. State $H_0$ and $H_1$; decide between a one-sided and a two-sided test;
    2. Fix the **significance level** $\alpha$ in advance (commonly 0.05);
    3. Choose a **test statistic** (e.g. $z$, $t$, $\chi^2$) whose distribution under $H_0$ is known;
    4. Compute the statistic and the p-value from the sample (or compare with a critical value);
    5. If $p<\alpha$, **reject** $H_0$; otherwise **fail to reject** $H_0$ (note: this is not "accepting $H_0$").
- Follow-up: $\alpha$, the sample size and the test should all be decided **before** looking at the data, otherwise you invite p-hacking.

Q2: ***What does a p-value actually mean? What are the common misreadings?***

- **Definition**: assuming $H_0$ is true, the probability of obtaining a test statistic **at least as extreme** as the one observed: $p=P(|T|\ge|t_{obs}|\mid H_0)$.
- Common misreadings:
    - "The p-value is the probability that $H_0$ is true" -- the p-value is $P(\text{data}\mid H_0)$, not $P(H_0\mid \text{data})$; the latter requires a prior (Bayesian).
    - "$1-p$ is the probability that $H_1$ is true".
    - "A smaller p-value means a bigger effect" -- the p-value depends on both effect size and sample size; with a huge sample a tiny effect is significant. Always report the **effect size and confidence interval** too.
    - "$p>0.05$ means there is no difference" -- the test may simply be underpowered.
    - "p=0.05 means a 5% false positive rate" -- the share of false positives among significant results (FDR) also depends on the prior fraction of true effects and on power.
- Under $H_0$ the p-value is **uniformly distributed** on $[0,1]$ -- this is the key to understanding peeking and multiple comparisons.

Q3: ***Explain Type I error, Type II error and statistical power***

| | $H_0$ true | $H_1$ true |
|---|---|---|
| Reject $H_0$ | **Type I error** (false positive), probability $\alpha$ | Correct, probability $1-\beta$ (**power**) |
| Do not reject $H_0$ | Correct, probability $1-\alpha$ | **Type II error** (false negative), probability $\beta$ |

- **Power** $=1-\beta$: the probability of detecting an effect when it truly exists; A/B tests usually require 80%.
- Factors that increase power: larger effect size, larger sample size, smaller variance, larger $\alpha$.
- At a fixed sample size, $\alpha$ and $\beta$ trade off; only more samples or lower variance can reduce both at once.

Q4: ***How should a confidence interval be interpreted? How does it relate to hypothesis testing?***

- 95% confidence interval: if we **repeated the sampling** many times and built an interval the same way each time, about 95% of those intervals would contain the true parameter.
- Misreading: "There is a 95% probability that the true parameter lies in this interval" -- in the frequentist view the parameter is fixed, so a specific interval either contains it or not; that statement describes a Bayesian **credible interval**.
- Interval for a difference in means: $(\bar x_B-\bar x_A)\pm z_{1-\alpha/2}\sqrt{\frac{s_A^2}{n_A}+\frac{s_B^2}{n_B}}$.
- **Duality**: a two-sided test rejects $H_0:\delta=0$ at level $\alpha$ ⇔ the $1-\alpha$ confidence interval excludes 0. The interval carries more information than a single p-value (effect size and uncertainty).

Q5: ***What is the Central Limit Theorem (CLT)? Why does it matter for A/B testing?***

- **CLT**: for i.i.d. $X_1,\dots,X_n$ with mean $\mu$ and variance $\sigma^2<\infty$, $\frac{\bar X-\mu}{\sigma/\sqrt n}\xrightarrow{d}\mathcal N(0,1)$.
- Why it matters: most business metrics (revenue, time spent) are **not normal themselves**, but their **sample means** are approximately normal, so we can use z-/t-tests on mean differences and build confidence intervals.
- Caveats:
    - Heavy-tailed distributions (e.g. revenue, a few "whale" users) converge slowly; they need larger samples, **winsorizing/capping**, or a log transform;
    - Samples must be independent -- if the randomization unit is the user but the analysis unit is the session/page view, observations are not independent; use the delta method or cluster-robust standard errors;
    - The CLT fails when the variance is infinite (e.g. the Cauchy distribution).

## Common Tests

Q6: ***What is the difference between a z-test and a t-test? What is Welch's t-test?***

- **z-test**: population variance known, or sample size large (sample variance used as a proxy); the statistic is standard normal. Proportion metrics (conversion rate) commonly use the two-proportion z-test:
  $$z=\frac{\hat p_B-\hat p_A}{\sqrt{\hat p(1-\hat p)\left(\frac1{n_A}+\frac1{n_B}\right)}}$$
  where $\hat p$ is the pooled proportion.
- **t-test**: population variance unknown and estimated from the sample; the statistic follows a t-distribution (depends on degrees of freedom, heavier tails than the normal). More accurate for small samples; as $n$ grows, t → normal.
- **Student's t-test** assumes the two groups have **equal variances**; **Welch's t-test** does not:
  $$t=\frac{\bar x_B-\bar x_A}{\sqrt{\frac{s_A^2}{n_A}+\frac{s_B^2}{n_B}}},\quad \nu\approx\frac{\left(\frac{s_A^2}{n_A}+\frac{s_B^2}{n_B}\right)^2}{\frac{(s_A^2/n_A)^2}{n_A-1}+\frac{(s_B^2/n_B)^2}{n_B-1}}$$
  (Welch–Satterthwaite degrees of freedom). In practice **use Welch by default**: it loses almost no power when variances are equal and is more robust when they are not.
- **Paired t-test**: two measurements on the same subject (before/after); run a one-sample t-test on the differences.

Q7: ***When is the chi-square test used?***

- **Test of independence**: whether two categorical variables are independent (e.g. a contingency table of experiment group × converted or not).
- **Goodness-of-fit test**: whether observed counts match an expected distribution -- the **SRM check** in A/B testing is exactly this.
- Statistic: $\chi^2=\sum\frac{(O_i-E_i)^2}{E_i}$, with $(r-1)(c-1)$ degrees of freedom (independence) or $k-1$ (goodness of fit).
- Condition: expected counts generally at least 5; otherwise use **Fisher's exact test**.
- For a 2×2 table the chi-square test is equivalent to the two-sided two-proportion z-test ($\chi^2=z^2$).

Q8: ***When would you use the Mann-Whitney U test?***

- A non-parametric test comparing two independent samples that **does not assume normality**; it is rank-based: pool both groups, rank the values, and compare rank sums.
- Its null hypothesis is $P(X>Y)=P(Y>X)$ (stochastic dominance); it is **not strictly a test of equal medians** -- it can only be read as a median difference when the two distributions have the same shape and differ only by a shift.
- Use cases: small samples, strong skew, outliers, ordinal data.
- Drawback: it does not directly tell you "how much the mean increased", while the business usually cares about means (e.g. revenue per user); with large samples the t-test is already robust thanks to the CLT. Alternatives include the bootstrap or a t-test after a log transform.

Q9: ***What is the bootstrap? When is it used?***

- Idea: approximate the population distribution with the empirical distribution; resample the data **with replacement** to the same size $B$ times (e.g. 1000–10000), compute the statistic each time, and use the distribution of these statistics to estimate the standard error and confidence interval.
- **Percentile interval**: take the 2.5% and 97.5% quantiles of the bootstrap statistics; BCa intervals are more accurate.
- Use cases: statistics without a simple analytic standard error (median, quantiles, ratio metrics, AUC, etc.) or with a complicated distribution.
- Caveats:
    - Resample by the **randomization unit** (e.g. by user, not by event), otherwise the variance is underestimated;
    - Works poorly for extreme quantiles, very small samples and heavy tails;
    - Computationally expensive; large-scale experimentation platforms usually use the delta method instead.

```python
import numpy as np
def bootstrap_ci(a, b, stat=np.mean, B=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    diffs = [stat(rng.choice(b, len(b))) - stat(rng.choice(a, len(a))) for _ in range(B)]
    return np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
```

## A/B Test Design

Q10: ***How do you choose metrics for an A/B test? What are north-star and guardrail metrics?***

- **North-star metric / OEC (Overall Evaluation Criterion)**: the core metric reflecting long-term product value, e.g. active-user retention, meaningful time spent per user; it should be **sensitive to business value, movable within the experiment window, and hard to game**.
- **Primary (decision) metric**: the metric this experiment directly aims to improve, usually a measurable proxy for the north star, e.g. click-through rate or conversion rate.
- **Guardrail metrics**: metrics that must not get worse, of two kinds:
    - Business guardrails: revenue, retention, complaint rate, uninstall rate;
    - Trustworthiness guardrails: **SRM**, page load latency, crash rate -- used to verify that the experiment itself is reliable.
- Properties of a good metric: **sensitivity** (can detect change), **directionality** (aligned with long-term value), low variance; historical experiments or A/A tests can be used to assess a metric's variance and sensitivity.
- Follow-up: when short-term metrics conflict with long-term value (e.g. more ads raise revenue but hurt retention), you need long-term holdouts or calibrated proxy metrics.

Q11: ***How do you choose the randomization unit?***

- Common units: **user** (user id), device/cookie, session, page view, region/time slice (clusters).
- Principles:
    - **Consistent user experience**: user-visible changes (UI, recommendation policy) should be split by user so the same user does not flip between versions;
    - **The randomization unit should equal or be coarser than the analysis unit**: if you split by user but analyze per page view (e.g. CTR = clicks/PV), observations are not independent and the naive variance formula underestimates the variance; use the **delta method** or a user-level bootstrap;
    - **Interference / network effects**: when users influence each other, use coarser units (region, cluster, time slice);
    - Finer units give more samples and higher power, but may cause inconsistent experiences and leakage.
- Logged-out users can only be split by cookie/device, which suffers from one person using multiple devices.

Q12: ***How do you calculate the sample size for an A/B test? What is MDE?***

- **MDE (Minimum Detectable Effect)**: the smallest effect the experiment can reliably detect at a given $\alpha$ and power; it should be set by **business significance** (how big a lift is worth shipping), not adjusted after the fact.
- Two-sample comparison of means, sample size per group (two-sided):
  $$n=\frac{(z_{1-\alpha/2}+z_{1-\beta})^2\cdot 2\sigma^2}{\delta^2}$$
- With $\alpha=0.05$ and 80% power, $(1.96+0.84)^2\approx 7.85$, giving the rule of thumb $n\approx\frac{16\sigma^2}{\delta^2}$ (per group).
- Proportion metrics: $\sigma^2=p(1-p)$. E.g. with a baseline conversion rate $p=10\%$ and a target absolute lift $\delta=0.5\%$: $n\approx 16\times0.09/0.005^2=57600$ per group.
- Pattern: sample size is proportional to $\sigma^2$ and inversely proportional to $\delta^2$ -- halving the MDE requires **4×** the samples. This is why variance reduction (CUPED) directly shortens experiments.

Q13: ***How long should an experiment run? What are novelty and primacy effects?***

- Duration = required sample size / daily traffic entering the experiment, but it must also:
    - Cover at least **full weekly cycles** (usually 1–2 weeks or more) to remove day-of-week effects;
    - Avoid holidays, big promotions and other unusual periods, or clearly state how far the results generalize;
    - Be long enough to see **novelty/primacy effects** wear off.
- **Novelty effect**: users engage more with a new feature in the short term simply because it is new; the effect decays over time, leading to overestimation.
- **Primacy effect / change aversion**: existing users are used to the old version and perform worse at first, recovering only after adapting, leading to underestimation.
- How to detect them: plot the **treatment effect over time**; analyze new vs. existing users separately (new users are not affected by either effect); analyze by days since first exposure (cohort); set up a long-term holdout if needed.

Q14: ***What is the "peeking" problem? How does sequential testing solve it?***

- **Peeking**: repeatedly checking the p-value during the experiment and stopping as soon as $p<0.05$. Since the p-value wanders randomly over time under $H_0$, multiple looks are a form of multiple comparisons, and **the actual Type I error rate is far above 5%** (the more looks, the higher).
- Solutions:
    - **Fixed sample size**: compute the sample size in advance and only look at the end;
    - **Group sequential testing**: pre-specify a number of interim analyses and allocate α with an **α-spending function** (O'Brien-Fleming, Pocock) -- O'Brien-Fleming boundaries are very strict early and close to the usual threshold at the end;
    - **Always-valid inference**: e.g. mSPRT (mixture sequential probability ratio test) and confidence sequences, which allow looking at any time while controlling the error rate, at the cost of slightly lower power at any fixed time point;
    - Bayesian methods are also affected by optional stopping in terms of their frequentist properties, so they do not fully avoid the problem.

Q15: ***What is the multiple comparisons problem? How do Bonferroni and BH differ?***

- When testing $m$ hypotheses at once (multiple metrics, multiple treatment arms, multiple segments), the probability of at least one false positive, $1-(1-\alpha)^m$, grows quickly (about 64% for $m=20$).
- **FWER (family-wise error rate) control -- Bonferroni**: use threshold $\alpha/m$ for each test. Simple but very conservative; power drops sharply for large $m$. Improvement: the Holm step-down procedure (uniformly more powerful).
- **FDR (false discovery rate) control -- Benjamini-Hochberg**: FDR = expected proportion of false positives among rejected hypotheses. Steps:
    1. Sort p-values in ascending order $p_{(1)}\le\dots\le p_{(m)}$;
    2. Find the largest $k$ such that $p_{(k)}\le\frac{k}{m}q$;
    3. Reject the first $k$ hypotheses.
- Choice: use FWER when a critical decision must strictly avoid any false positive; FDR is more appropriate for exploratory analysis (screening many metrics/segments).
- Practice: pre-specify a **single primary metric**; treat others as secondary/exploratory; segment-level findings need replication.

Q16: ***What is SRM? How do you check for it?***

- **SRM (Sample Ratio Mismatch)**: the actual group proportions deviate significantly from the designed ratio (e.g. designed 50/50, observed 50.8/49.2).
- Check: chi-square goodness-of-fit test, $\chi^2=\sum\frac{(O_i-E_i)^2}{E_i}$; a very strict threshold (e.g. $p<0.001$) is commonly used for alerts.
- SRM means **something is wrong with randomization or the data pipeline**; the results are untrustworthy and must be debugged before interpreting the effect. Common causes:
    - Assignment bugs, hash collisions, the treatment page loading slower so that logging events are lost;
    - Bot filtering applied unevenly across groups;
    - Different triggering conditions in treatment (only some users counted in a triggered analysis);
    - Changing the traffic allocation mid-experiment.

Q17: ***What are network effects / interference? How do you handle them?***

- A/B testing relies on **SUTVA** (Stable Unit Treatment Value Assumption): one unit's outcome is not affected by other units' assignments. It is violated in:
    - **Social networks**: a treatment user's new feature affects the behavior of their friends (in control) → the effect is diluted or amplified;
    - **Two-sided marketplaces** (ride-hailing, food delivery, ad auctions): treatment and control share supply/budget; treatment "steals" resources, so the effect is **overestimated**;
    - Shared inventory, shared model training data (feedback loops).
- Solutions:
    - **Cluster randomization**: randomize by social-graph clusters (graph cluster randomization) or by geographic region;
    - **Switchback experiments**: alternate the treatment on/off for an entire region by time slice; common in two-sided marketplaces;
    - **Budget/resource splitting**: give treatment and control separate budgets or supply pools;
    - Cost: fewer randomization units, higher variance, lower power.

Q18: ***What is CUPED? Why does it reduce variance?***

- **CUPED** (Controlled-experiment Using Pre-Experiment Data, Microsoft 2013): adjust the metric $Y$ using a pre-experiment covariate $X$ (usually the same metric measured before the experiment):
  $$Y_{cuped}=Y-\theta(X-\bar X),\quad \theta=\frac{\mathrm{Cov}(X,Y)}{\mathrm{Var}(X)}$$
- Variance: $\mathrm{Var}(Y_{cuped})=\mathrm{Var}(Y)(1-\rho^2)$, where $\rho$ is the correlation between $X$ and $Y$. With $\rho=0.7$, variance drops by about 51%, roughly equivalent to doubling the sample size.
- **Unbiasedness**: $X$ is pre-experiment data unaffected by the treatment, so both groups have the same expected $X$, and the adjustment does not change the expected treatment effect.
- It is essentially equivalent to adding a pre-treatment covariate in a regression (ANCOVA / regression adjustment); extension: CUPAC uses an ML model's prediction as the covariate.
- Note: the covariate must be **unaffected by the treatment**; for new users with no history, use other pre-experiment features or a missingness indicator.

Q19: ***What are ratio metrics? Why do we need the delta method?***

- **Ratio metric**: both numerator and denominator are random variables, e.g. CTR = total clicks / total impressions; when splitting by user, the analysis unit (impression) is finer than the randomization unit (user).
- Multiple impressions from the same user are correlated, so the naive binomial variance $p(1-p)/N$ **underestimates the variance**, inflating false positives.
- **Delta method**: a first-order Taylor expansion of $R=\bar Y/\bar X$ (where $\bar Y,\bar X$ are user-level means):
  $$\mathrm{Var}\left(\frac{\bar Y}{\bar X}\right)\approx\frac{1}{\mu_X^2}\mathrm{Var}(\bar Y)+\frac{\mu_Y^2}{\mu_X^4}\mathrm{Var}(\bar X)-2\frac{\mu_Y}{\mu_X^3}\mathrm{Cov}(\bar Y,\bar X)$$
  where variances and covariance are estimated at the user level (e.g. $\mathrm{Var}(\bar Y)=s_Y^2/n$).
- The delta method is also used for confidence intervals of the **relative lift** $\frac{\bar Y_B}{\bar Y_A}-1$. Alternative: a user-level bootstrap.

Q20: ***What is Simpson's paradox? How does it show up in A/B testing?***

- **Simpson's paradox**: B beats A in every subgroup, yet A beats B when the subgroups are combined (or vice versa). The cause is a **confounding variable** together with **different subgroup compositions** across groups.
- Classic example: kidney stone treatment -- treatment A has a higher success rate for both small and large stones, but A was used more often for the harder large stones, so A's pooled success rate is lower.
- Scenarios in A/B testing:
    - **Changing the traffic allocation** mid-experiment (e.g. 1% on day 1, 50% on day 2) while baseline conversion differs by day;
    - Segmenting by triggering conditions or post-treatment variables (segment membership affected by the treatment);
    - Unbalanced traffic allocation across platforms/countries.
- Remedies: keep the allocation fixed, or do a **stratified analysis** by day/stratum and reweight; segment only on **pre-experiment** attributes.

## Causal Inference and Bayesian Methods

Q21: ***What is confounding? How do propensity score methods work?***

- **Confounder**: a variable that affects both the treatment $T$ and the outcome $Y$, so that "correlation" is not "causation". Randomized experiments break confounding through random assignment; with observational data we must adjust for it.
- Key assumptions: **ignorability** (given $X$, $T$ is independent of the potential outcomes, i.e. no unobserved confounding), **overlap/positivity** ($0<e(x)<1$), and SUTVA.
- **Propensity score** $e(x)=P(T=1\mid X=x)$ (usually estimated with logistic regression/GBDT). Rosenbaum & Rubin showed that conditioning on $e(x)$ suffices to balance $X$.
- Usage:
    - **Matching (PSM)**: match each treated unit with control units of similar propensity score;
    - **Inverse probability weighting (IPW)**: weight treated units by $1/e(x)$ and control units by $1/(1-e(x))$, $\widehat{ATE}=\frac1n\sum\left[\frac{T_iY_i}{e(x_i)}-\frac{(1-T_i)Y_i}{1-e(x_i)}\right]$;
    - Stratification, or use as a regression covariate; **doubly robust (AIPW)** combines an outcome model and a propensity model and is consistent if either is correct.
- Limitations: **cannot handle unobserved confounding**; weights explode when propensity scores approach 0/1, requiring trimming; check covariate balance after matching (standardized mean differences).

Q22: ***Explain Difference-in-Differences (DiD)***

- Setting: randomization is impossible, but there is data for treatment and control groups **before and after** the treatment (e.g. a new policy launched in one city).
- Estimator:
  $$\hat\tau_{DiD}=(\bar Y_{T,post}-\bar Y_{T,pre})-(\bar Y_{C,post}-\bar Y_{C,pre})$$
  The first difference removes fixed differences of the treated group; the second removes the common time trend.
- Regression form: $Y_{it}=\beta_0+\beta_1\text{Treat}_i+\beta_2\text{Post}_t+\tau(\text{Treat}_i\times\text{Post}_t)+\epsilon_{it}$.
- Core assumption: **parallel trends** -- absent treatment, both groups would have followed the same trend. Check pre-trends with multiple pre-treatment periods (event study).
- Extensions/caveats: with staggered treatment timing (staggered adoption), traditional two-way fixed effects can be biased and newer estimators are needed (e.g. Callaway–Sant'Anna); when parallel trends fails, consider the **synthetic control method**.

Q23: ***What is uplift modeling? How does it differ from a standard response model?***

- A **standard response model** predicts $P(Y=1\mid X)$ -- who will buy; an **uplift model** predicts the **individual treatment effect (CATE)**: $\tau(x)=E[Y\mid T=1,X=x]-E[Y\mid T=0,X=x]$ -- who will buy **because of** the intervention.
- Four user quadrants: persuadables (buy only if treated; the target), sure things (always buy; treatment wasted), lost causes (never buy), sleeping dogs (treatment makes them churn; avoid).
- Methods (meta-learners):
    - **S-learner**: one model with $T$ as a feature; predict $f(x,1)-f(x,0)$;
    - **T-learner**: train separate models on treatment and control, take the difference;
    - **X-learner**: suited to imbalanced treatment/control sample sizes;
    - Also causal forests, trees that directly optimize uplift (split criteria based on treatment-effect differences), and the DR-learner.
- Training data should ideally come from a **randomized experiment**; evaluation cannot use ordinary AUC (individual true uplift is unobservable); use **Qini curves / uplift curves**.

Q24: ***What is the difference between Bayesian and frequentist A/B testing?***

| | Frequentist | Bayesian |
|---|---|---|
| Parameter | Fixed, unknown | Random variable with a prior |
| Output | p-value, confidence interval | Posterior, $P(B>A\mid\text{data})$, expected loss |
| Interpretation | Indirect (how unusual the data are under $H_0$) | Intuitive ("95% probability that B beats A") |
| Prior | Not needed | Needed; affects results (especially with small samples) |

- Classic approach for conversion rates: the **Beta-Binomial conjugate** model. With prior $\text{Beta}(\alpha,\beta)$ and $s$ successes in $n$ trials, the posterior is $\text{Beta}(\alpha+s,\beta+n-s)$; estimate $P(p_B>p_A)$ and the **expected loss** $E[\max(p_A-p_B,0)]$ via Monte Carlo sampling.
- Misconception: "Bayesian methods let you peek freely" -- the posterior is still a valid Bayesian update under optional stopping, but its **frequentist error rate** is not controlled; with a rule like "stop when $P(B>A)>95\%$", the false positive rate inflates as well.
- Practice: high-traffic platforms mostly use frequentist tests + sequential testing + variance reduction; Bayesian methods shine with low traffic, when intuitive decisions (expected loss) are needed, and when incorporating priors from past experiments.

```python
import numpy as np
rng = np.random.default_rng(0)
a = rng.beta(1 + conv_a, 1 + n_a - conv_a, 100_000)
b = rng.beta(1 + conv_b, 1 + n_b - conv_b, 100_000)
prob_b_better = (b > a).mean()
expected_loss_b = np.maximum(a - b, 0).mean()
```
