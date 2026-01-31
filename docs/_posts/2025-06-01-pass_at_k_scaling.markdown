---
layout: post
title:  "Scaling Laws for pass@k and temperature"
date:   2025-06-01
categories: scaling
usemathjax: true
---

* TOC
{:toc}

# Takeaways

<aside>

- Pass@k is a complicated metric to reason about because it has a nonlinear dependency on k and temperature (T), which (I think) cannot be factored into independent dependencies on k and T.
- The optimal temperature $$T^*$$ (which maximizes E(pass@k)) depends on k. I present results from a toy model to argue that $T^*(k)$ follows a power law over several decades of k.
    - Crucially, this behavior occurs because eval sets contain tasks of varying difficulties. We do not see it if we evaluate only a single task. (A difficult task just means one with a low pass@1 rate.)
    - As $k \rightarrow \infin$ there is an asymptotic upper bound on $T^*$ that depends only on the single most difficult task in the eval set. (Theoretically, if we had an infinite eval set with no single ???hardest task??? then the $T^*(k)$ power law scaling would continue as  $k \rightarrow \infin$ because there would always be harder tasks to pass.)
- However, **the $T^*(k)$  power law also depends on the details of the eval set, specifically the tail of the distribution of task difficulty**. The two plots below show $T^*(k)$ curves for increasingly fat-tailed families of eval distributions (Gaussian < exponential < power law). For more fat-tailed distributions, one should raise T significantly more as k increases. Note that this is true even when comparing distributions with the same mean and variance - it is only due to the shape of the tail.
- As a next step, I will confirm whether the scaling behavior of LLMs on real coding benchmarks matches the predictions of this toy model.
</aside>

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%201.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%202.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%203.png)

# Overview / Motivation

$pass@k$ is a widely used metric for AI benchmarks where i) model output is open-ended  with many possible formulations of a ???correct solution??? and ii) it is feasible to automatically verify solutions at scale. Code is one prominent domain where this is the case, and formal proofs are another (using Lean for verification). This class of benchmarks is distinct from benchmarks where ???valid??? answers must come from a limited set (e.g. multiple choice or math benchmarks where the answer must be an integer between 0 and 999) or short-form knowledge benchmarks, where there is unlikely to be much variation in correct responses (e.g. ???What is the capital of France????)

For these simpler benchmarks, $accuracy$ (which I???ll define here as pass@1 for T=0) and $perplexity$ (of a correct reference solution) may be sufficient metrics to estimate a model???s capabilities. However they have shortcomings on open-ended benchmarks:

1. Accuracy is a binary variable for each task. For challenging benchmarks (or weak models) the average value may be very low (even zero), meaning that it is less sensitive to model improvements and has wide confidence intervals. By contrast, E(pass@k) is continuous-valued  for each task, increases with k, and can be estimated to arbitrarily high precision by producing N > k generations.
2. Perplexity is calculated on one (or at best, a few) reference solutions. Any coding task has a large number of correct solutions, due to trivial style variations (such as variable names), and possibly more significant algorithmic variations as well. During training, a model may learn to solve a task in a particular way which is not well reflected in the reference solution (imagine a model that always uses snake case while the provided solutions are in camel case).

$pass@k$ would seem to be the most useful metric to estimate performance (and track improvement) on open-ended, verifiable tasks. However, it brings some distinct challenges:

1. ***Estimation***: both accuracy (for T=0) and perplexity are deterministic and can be exactly calculated for a given model and task. Strictly speaking, pass@k is a value ***in expectation*** for a random sample of k generations from a model. This requires more care in how to best estimate it.
2. ***Sensitivity to temperature***: using T=0 for k>1 is pointless since all generations will be identical. Any T>0 value can be used for pass@k and it is not intuitively clear which T is ???best??? or if comparing two models at the same T is truly ???apples-to-apples???. (Is it possible that model A beats model B at T=0.5 but model B wins at T=1? NOTE: I think this is provably possible, and might even be fairly common))
3. ***Sensitivity to k***: Likewise, pass@k can be thought of as curve over $1 \leq k < \infin$. It is not immediately obvious whether a finite set of k???s provides the ???full picture??? when comparing two models.

I developed a toy model to  dig deeper into the behavior of pass@k as a metric and hopefully provide guidance on how to use it to extract the most actionable signal for model development. The toy model is motivated by the following points that I want to better understand:

1. Pass@k clearly depends on the strength of the model and the overall difficulty of the benchmark (the things we care most about), but it also depends on the following parameters:
    1. k
    2. temperature (T)
    3. the variance of difficulty of tasks in the benchmark set
2. For a given k, there must be an optimal T that gives the highest pass@k. I would guess that T=0 is usually optimal for k=1, and assume that the higher k is, the higher the optimal T will be. (This is basically an exploration vs exploitation trade-off)
3. For a single task (at a given T) there is a simple relation between pass@k and pass@1 (or p):
    1. $pass@k = 1 - (1-p)^k$
4. However, we usually report the average pass@k over a benchmark (a set of tasks) and there is no obvious relationship between pass@k and pass@1 for the full set
5. It is also not obvious how the pass@k curve will vary with T and what causes differences between different benchmarks. Do these things matter for choosing a useful metric to optimize?

# Toy Model

For any task prompt, assume there is some non-zero probability (at $T>0$) that a given LLM will generate a correct solution. The probability of generating a specific token $x_i$ scales with $T$  as:

$$
p(x_i) \sim \exp(z_i / T)
$$

(of course $T$ also influences the normalization constant). Note the similarity to a zero-mean normal distribution:

$$
p(x) \sim \exp(-x^2 / 2 \sigma^2)
$$

Consider a discrete normal distribution (i.e. defined only on the integers $\mathbb{Z}$). This is equivalent to a softmax function over $\mathbb{Z}$ where $z_i = -x_i^2 / 2$ and $T = \sigma^2$.

Since the possible generations of an LLM form a countable set, we can view this distribution as a simplified representation of an LLM (where each integer represents one generation). In this vein, we can also create a simple representation of a ???task??? - we???ll say that a given task has a ???correct answer??? which is some integer $c$. Then, for that task, we can generate/sample a random integer $g$ from the model/distribution and we say it passes the task if $g=c$. We can think of think of tasks with large $|c|$ as ???difficult??? and tasks with small $|c|$ as ???easy???.

This may seem like a strange analogy, since neither the model nor the task have any semantic meaning, but that should not matter. The key mathematical point is that every task has a correct answer and the model can generate it with some probability that depends on $T$ as in the softmax function. It might also seem strange that each task does not condition on a task-specific prompt, i.e. we always generate from the same normal distribution. Again, I think that is all we need to capture the basic notion of ???task difficulty??? - we don???t need to flesh out the specific details of each task, what matters is that each task has a pass@1 rate of  $p_{gen}(c|T)$.

The final element of this toy model is that we want to create a benchmark or evaluation set of tasks $\{c_i\}$. As mentioned above, the scaling behavior of pass@k has a complex dependence on this ???task difficulty distribution???. It???s not representative to just consider a scenario where $c$ is the same for all tasks. I???ll create an evaluation set by sampling $\{c_i\}$ from some distribution which I???ll call $p_{task}(c)$ - not to be confused with $p_{gen}$.  

 $p_{task}(c)$ can be a discrete normal distribution, but we can also sample tasks from: 

- an exponential distribution:  $p_{task}(c = j) \sim \exp(-|j|/\sqrt{T})$
- a power law distribution: $p_{task}(c = j) \sim |j|^{-\gamma}$
    - (for the power law, $T$ is a more complicated polynomial function of $\gamma$)
- (note that  $p_{task}$ refers to the probability of selecting a task to be in the eval set, while $p_{gen}$ refers to the probability of our model passing a task it is given.)

# Results

As I mentioned, a key issue for pass@k is the dependence on T. For pass@1 (that is, $p_{gen}(c|T)$) on a single task we can think of this as follows:

1. For $T=0$, we often have $p_{gen}(c)=0$ (if c is not the maximum-likelihood generation)
2. If we raise T, then $p_{gen}(c|T)$ will at first increase. Loosely speaking, the distribution is spreading out and probability mass ???flows towards??? c from the higher-likelihood values closer to the mode) 
3. Eventually, $p_{gen}(c|T)$ will reach some maximum value and decrease if we continue raising T. In this regime, we???re spreading out the distribution so much that more probability mass ???flows away??? from c towards the tails than ???flows towards??? it from the mode.
4. To give a concrete illustration: for a zero-mean normal distribution, $p(x)$ attains its largest value when $\sigma = |x|$.

## *How does the optimal T for an eval set scale with k?*

For a single task, the optimal T is the same for all k (as the above points show), but for a set of tasks the answer is not immediately obvious. I created an eval set by sampling tasks from a discrete normal distribution. I then calculated  $T^*$, the temperature which maximizes pass@k for each k (using Brent???s method).

<aside>

- One important observation is that as $k \rightarrow \infin$, pass@k is dominated by the contribution of the most difficult task (i.e. the largest $|c|$). Asymptotically, the optimal T is the value that maximizes pass@1 for this task.
- At first I had planned to use this as a metric to compare different eval sets. However it is very noisy since it depends on the single most extreme task so I abandoned this idea.
- I noticed that for intermediate values of k, optimal temperature follows a smooth power law over several decades so I focused on this instead.
</aside>

### Normally Distributed Task Difficulty

On the left is the full plot for  $T^*$  vs k. (task_spread is the standard deviation of the task distribution). On the right I show only those points with $0.03 < pass@k < 0.98$, to make the scaling regime clearer. $T^*$ also follows a scaling law with respect to task_spread, as shown in the bottom plot.

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%204.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%205.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%206.png)

I fit a regression to the data in this scaling regime and got the following relation (*s = task_spread*):

$$
T^* ??? 0.264*k^{0.728}*s^{1.283} \text{ ; R??=0.9996}
$$

### Exponentially Distributed Task Difficulty

Next I did the same analysis where the tasks are drawn from an exponential distribution instead of a normal distribution. It turns out that the scaling exponents are different: 

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%207.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%208.png)

$$
T^* ??? 0.117*k^{1.054}*s^{0.957} \text{ ; R??= 0.9989}
$$

To expand on this point - the optimal temperature for a given eval set does not only depend on the mean and variance of the task difficulty distribution, it also depends on the kurtosis (and/or skewness?). (An exponential distribution has greater kurtosis than a normal distribution w/ equal variance - in other words, its variance is more due to fat tails than to intermediate values)

### Power Law Distributed Task Difficulty

Following this finding, I look at tasks drawn from a (truncated) power law distribution, which should have even greater kurtosis. In this case the scaling behavior wrt k does not appear at low k and low pass@k rates. It only seems to  kick in above ~k>20. (Note: task_stddev_trunc is the equivalent of task_spread here. I calculated it from the actual truncated dist I used, since truncation has a much stronger effect on power law than on gaussian/exponential dists)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%209.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%2010.png)

### Comparing all three task distributions

When we compare the $T^*(k)$ for each family the differences are striking. Reproducing the plots from the introduction below, we can tell a clear story. For instance, there are seemingly similar gaussian and power law task sets that have a similar pass@k for $k<10$. However, as we increase k, performance on the gaussian set saturates quickly (say by $k \sim O(100))$, requiring only a modest increase in T to pass the hardest tasks. But for the power law set, it is necessary to increase T by several orders of magnitude to maximize pass@k (to ???chase after??? the hardest tasks) and even then, pass@k rises slowly with k.

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%201.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%202.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%203.png)

## Deciding how large to make k

When planning experiments it would be useful to be able to estimate the minimum value of k neccessary to reach a desired pass@k score. For instance, if we are using pass@k as a metric to compare models, we will typically get more statistical precision (in other words, information) when values we are comparing are centered around 0.5, rather than clustered closer to 0 or 1.
The plots below show that in many cases there is a linear relationship $\log(k) \sim \text{logit}(pass@k)$, thus it seems like we should be able to extrapolate from results at small k.

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%2011.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%2012.png)

![image.png](Labnote%20-%20Pass@k%20and%20temperature/image%2013.png)

*Top left: Gaussian task distribution*

*Top right: Exponential task distribution*

*Bottom left: Power law task distribution*

