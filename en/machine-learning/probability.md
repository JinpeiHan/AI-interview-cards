# Probability Theory

Q1: **Describe what a probability distribution is**

- A probability distribution is a statistical function that describes all the possible values, and their likelihoods, that a random variable can take within a given range.
- There are two main types of probability distributions:
    - Discrete probability distributions: used to describe random variables with discrete outcomes, e.g. the number of heads in five consecutive coin tosses
    - Continuous probability distributions: used to describe random variables with continuous outcomes, e.g. house prices

Q2: **What is the difference between the Bernoulli distribution and the binomial distribution?**

- The Bernoulli distribution is a discrete probability distribution of a random variable that takes a binary output: 1 with probability p, and 0 with probability (1-p).
- The binomial distribution is the discrete probability distribution of the number of successes in n independent success/failure trials, where each trial has success probability p; when n=1, the binomial distribution is the Bernoulli distribution

Q3: **What is the difference between the likelihood and the posterior in Bayesian theory?**

- Informally, the posterior probability is the conditional probability of inferring the "cause" after observing the "effect", while the likelihood is the probability of the "effect" being produced by the "cause".

Q4: **What is a Poisson process?**

- A Poisson process is a model for a series of discrete events in which the average time between events is known (fixed), but the exact timing of events is random. The arrival of an event is independent of the previous events (the waiting time between events is memoryless).
- A Poisson process satisfies the following criteria:
1. Events are independent of each other. The occurrence of one event does not affect the probability of another event occurring.
2. The average rate (events per time period) is constant.
3. Two events cannot occur at the same time.

Q5: **What is the difference between disjoint (mutually exclusive) events and independent events?**

- Mutually exclusive events cannot occur at the same time, whereas two independent events are events where the occurrence of one has no effect on the probability of the other occurring
- The number of trials differs. Mutually exclusive events are different events occurring within a single trial, while independent events are different events occurring in two or more different trials
- If A and B are mutually exclusive events, the probability addition rule P(A+B)=P(A)+P(B) holds; if A and B are independent events, the probability multiplication rule P(AB)=p(A)P(B) holds

Q6: **When is an event independent of itself?**

- The only events independent of themselves are those that occur with probability 0 or probability 1. Being independent of itself means $P(A)=(P(A))^2$, in which case $P(A)$ is 0 or 1;

Q7: **What is the difference between the binomial distribution and the geometric distribution?**

- In the binomial distribution, the number of trials is fixed and the number of "successes" is counted; in the geometric and negative binomial distributions, the number of "successes" is fixed and the number of trials needed to obtain the required number of "successes" is counted

Q8: ****Common characteristics of random variables****

- Expectation: represents the average level of the random variable X
- Variance/standard deviation: used to characterize the magnitude of fluctuation of the random variable X; the larger the variance, the greater the uncertainty of the outcome
- Quantile: for a random variable X, if P ( X ⩽ t ) = α, then t is called the α-quantile of X. Quantiles are often used to monitor anomalous data;
- Covariance/correlation coefficient: measures the relationship between two variables

Q9: ****How would you explain the normal distribution to someone who has never studied statistics****

- The normal distribution is a bit like the 80/20 rule. Most people are in the middle, and the closer to the middle, the more people there are; a small number of people are at each end, in roughly equal numbers. The transition from the majority to the minority is smooth, not an abrupt cliff-like change.

Reference: [https://blog.csdn.net/](https://blog.csdn.net/qq_34069667/article/details/107941951)

Q10: **What is parameter estimation?**

- Parameter estimation is the method of estimating the unknown parameters contained in a population distribution based on a sample drawn from the population. It is a basic form of statistical inference and an important branch of mathematical statistics, divided into two parts: point estimation and interval estimation.

Q11: **Common methods of parameter estimation**

- Common point estimation methods are the method of moments and maximum likelihood estimation, as well as least-squares estimation;
    - The method of moments is based on the law of large numbers and uses sample moments to estimate population moments;
    - Maximum likelihood estimation uses the known information of sample outcomes to infer, in reverse, the model parameter values most likely (with maximum probability) to have produced these sample outcomes;
- Interval estimation means, on the basis of point estimation, giving an interval range for the estimate of a population parameter; the interval is usually obtained as the sample statistic plus or minus the estimation error

Q12: **Advantages and disadvantages of maximum likelihood estimation**

- Advantages:
    - Highly interpretable, with a theoretical foundation;
    - Insensitive to standardized specifications of parameters and models;
    - Easy to implement
    - Lower variance than other methods (i.e. the estimation method least affected by sampling error) and unbiased as the sample size increases
- Disadvantages:
    - Requires strong assumptions about the structure of the data (e.g. assuming a Gaussian distribution)
    - The accuracy of the estimate depends heavily on whether the assumed form of the probability distribution matches the underlying true data distribution

Reference: [https://www.exploredatabase.com/2020/05/advantages-and-disadvantages-of-maximum-likelihood-methods.html](https://www.exploredatabase.com/2020/05/advantages-and-disadvantages-of-maximum-likelihood-methods.html)

Q13: **General steps of maximum likelihood estimation**

1. Write down the likelihood function according to the distribution assumed for the data;
2. Take the logarithm of the likelihood function;
3. Compute the derivative of the likelihood function;
4. Solve the likelihood equation.

Q14: ****How do you transform a skewed distribution into an approximately normal distribution?****

- Many models make numerous assumptions about the data, and the normal distribution is a common one; if the predictor variables and the target variable are normally distributed
, the model can make more reliable predictions; when the original data does not follow a normal distribution, some transformation needs to be applied to make it tend towards a normal distribution. Common data transformations include:
    - Log transformation
    - Square root transformation
    - The Box-Cox method, which performs a family of power transformations, including log and square root, to remove skewness from the data

Reference: [https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45](https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45)

Q15: **What is the difference between the cumulative distribution function (CDF) and the probability density function (PDF)?**

- The probability density function describes the probability distribution of a continuous random variable. The cumulative distribution function is a way of handling the probability distribution of both continuous and discrete random variables.

|  | PDF | CDF |
| --- | --- | --- |
| Definition | The PDF is the probability that the random variable (X) takes a value exactly equal to the random variable (Y). | The CDF is the probability that the random variable (X) takes a value less than or equal to the random variable (Y). |
| Variables | Applies only to continuous random variables. | Applies to both continuous and discrete random variables. |
| Values | The value lies between 0 and 1. | The value lies between 0 and 1. |

Q16: ****What is the difference between Bayesian estimation and maximum likelihood estimation?****

- **Maximum likelihood is a point estimate, while Bayesian inference estimates a distribution;** suppose we need to solve for a parameter θ: maximum likelihood finds the most likely value of θ, while Bayesian inference solves for the distribution of θ. In terms of formulas, Bayesian inference also introduces a prior and solves for the posterior distribution from the prior and the likelihood, while maximum likelihood uses the likelihood function directly and solves by maximizing it.

Q17: ****How do you check whether two events are independent?****

- By definition, if the occurrence of event A has no effect on the probability of event B occurring, and vice versa, then the two events are independent
- Mathematically, if for two events A, B we have P(AB)=P(A)·P(B), then events A and B are independent

Q18: **Name some methods for estimating the parameters of a probability distribution**

- Maximum likelihood (ML)
- Ordinary least squares (OLS)
- Method of moments

Reference: [https://www.itl.nist.gov/](https://www.itl.nist.gov/)

Q19: **What are the similarities and differences between frequentist inference and Bayesian inference?**

- There are two main approaches to statistical inference, frequentist and Bayesian, which interpret uncertainty differently; the frequentist approach deals with long-run probabilities (i.e. how probable a dataset is given the null hypothesis), while the Bayesian approach deals with the probability of a hypothesis given a particular dataset.
- Bayesian analysis incorporates prior information into the analysis, while frequentist analysis is purely data-driven
- The Bayesian approach can compute the probability that a particular hypothesis is true, while the frequentist approach computes the probability of obtaining another dataset at least as extreme as the collected one (giving a *P*-value).
- Compared with the frequentist approach, the Bayesian approach gives a more intuitive interpretation of results; the frequentist approach is often misunderstood.
- The frequentist approach assigns probabilities to data rather than to hypotheses, while the Bayesian approach assigns probabilities to hypotheses. In addition, Bayesian models incorporate prior knowledge into the analysis and update hypothesis probabilities as more data becomes available.

Reference: [Understanding the Differences Between Bayesian and Frequentist Statistics - International Journal of Radiation Oncology, Biology, Physics (redjournal.org)](https://www.redjournal.org/article/S0360-3016(21)03256-9/fulltext)
