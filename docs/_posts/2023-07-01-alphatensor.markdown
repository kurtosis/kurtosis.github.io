---
layout: post
title:  "DeepMind's AlphaTensor"
date:   2023-07-01 14:23:37 -0400
categories: alphatensor
usemathjax: true
---

DeepMind's AlphaTensor system ([blog][alphatensor-blog], [paper][alphatensor-nature]), introduced in October, 2022 uses deep reinforcement learning to discover efficient algorithms for matrix multiplication. It has, perhaps understandably, not received the same level of attention as recent advances in generative AI. However, there are a few aspects of this work which I think make it a particularly interesting application of deep learning:
* The complexity of the problem comes from its underlying mathematical structure. The challenge is not extracting information from large empirical data sets (e.g. text, image, omics) but solving an NP-hard problem. Also, unlike two-player games, such as chess and go, it cannot rely on self-play to provide continuous feedback and bootstrap improvements.
* AlphaTensor uses several techniques to tackle the problem, such as a transformer network to select actions from a high dimensional, discrete space and Monte Carlo tree search (MCTS) to solve the reinforcement learning problem.

In this post I'll break down the matrix multiplication problem and walk through [my implementation][my-repo] of AlphaTensor. (All figures below are from the AlphaTensor paper.)


# Addition and Multiplication

The importance of the order of operations stems from the simple fact that multiplication is a more expensive operation than addition. To build some intuition, consider which of the following arithmetic calculations takes more effort for you to do:\
$$15047 + 30821$$\
or \
$$15047 \times 30821$$

Clearly multiplication is more work (you might even be able to do the addition in your head, but for the multiplication this would require a strong memory!).

Consider two integers of N digits each. The cost of adding these numbers is $$O(N)$$ - we first add the digits in the ones place, then those in the tens place, etc. The cost of multiplying the numbers is $$O(N^2)$$ - we must multiply the "ones" digit of the first number by every digit in the second number, then do the same for the "tens" digit, etc. More generally, if we denote the magnitude of an integer as $$M$$ then the number of digits scales as $$log(M)$$. Thus, addition is $$O(log(M))$$ and multiplication is $$O(log(M)^2)$$. The key point here is the quadratic relation between multiplication and addition - this holds whether the numbers are of different magnitudes, are expressed in bits rather than base 10, or are floating point rather than integers.


Why does this observation matter? As it turns out, many calculations have multiple paths (i.e. sequences of scalar arithmetic operations) which produce the correct result. The number of paths increases with the size of the calculation and in general, paths with fewer multiplications are more efficient. As a simple example (courtesy of [Assembly AI][assembly-ai]) consider how we might compute the difference of two squares, $$a^2 - b^2$$. Both of these algorithms perform three operations and return the correct result:

$$c_1 = a \cdot a$$\
$$c_2 = b \cdot b$$\
return $$c_1 - c2$$

$$d_1 = a + b$$\
$$d_2 = a - b$$\
return $$d_1 \cdot d_2$$


However the second one is more efficient because it requires one multiplication rather than two. (You can try timing this on your laptop to confirm!)

# Matrix Multiplication

Consider matrix multiplication $$C = AB$$ where $$A$$ and $$B$$ are $$2 \times 2$$ matrices.

<!-- ![](/assets/images/two_by_two.png){: width="300"} -->

$$\begin{pmatrix}c_1 & c_2\\\ c_3 & c_4\end{pmatrix} = \begin{pmatrix}a_1 & a_2\\\ a_3 & a_4\end{pmatrix} \cdot \begin{pmatrix}b_1 & b_2\\\ b_3 & b_4\end{pmatrix}$$

The standard way to compute this is:

$$c_1 = a_1b_1 + a_2b_3 \text{, etc.}$$

Each element of the product requires two multiplications, resulting in eight multiplications overall. In 1969, [Strassen][strassen] showed that this can be computed using a two-level method which requires only seven multiplications.

![](/assets/images/strassen_algo.png){: width="200"}


In general, the standard way of multipling matrices of sizes $$(n \times m)$$ and $$(m \times p)$$, requires $$(n \times p) \times m$$ scalar multiplications. It turns out that for most cases which have been examined, it is possible to perform the operation with substantially fewer scalar multiplications. For example, the AlphaTensor paper reported that the case $$(n=4, m=5, p=5)$$ can be computed with 76 multiplications rather than 100.

![](/assets/images/best_ranks.png){: width="400"}

# Framing the tensor product as a set of discrete moves

The next step is to frame this search for efficient solutions with few multiplications (known as low-rank decompositions) as a problem that is amenable to machine learning. Looking at Strassen's algorithm, this may seem challening at first glance, as there are an enormous number of discrete permutations to choose from and it is not obvious what sort of gradient we might perform gradient descent over.

To start, consider describing the matrix multiplication $$C=AB$$ by a three-dimensional tensor $$T$$ where the element $$T_{i,j,k}$$ denotes the contribution of $$a_ib_j$$ to $$c_k$$. (Note that the dimensions of $$T$$ are $$(n \times m) \times (m \times p) \times (n \times p)$$.)

![](/assets/images/tensor_3d.png){: width="300"}

For example, in the $$(2,2,2)$$ case, $$c_1 = a_1b_1 + a_2b_3$$ is denoted by:\
$$T_{1, 1, 1} = 1$$\
$$T_{2, 3, 1} = 1$$\
$$T_{i, j, 1} = 0 \space \forall \space \text{other} \space (i, j)$$\
and similarly for $$c_2$$, $$c_3$$, and $$c_4$$.


Consider a process in which we first set a tensor $$S$$ equal to the zero tensor and then sequentially modify it by choosing three vectors $${\bf u}$$, $${\bf v}$$, and $${\bf w}$$, each of length 4, and performing the following update:\
$$S_{i, j, k} \leftarrow S_{i, j, k} + u_iv_jw_k$$


For example, applying the following vectors to $$S=0$$:\
$${\bf u =  v = w} = (1, 0, 0, 1)$$\
results in:\
$$S_{1, 1, 1} = 1$$\
$$S_{1, 1, 4} = 1$$\
$$S_{1, 4, 1} = 1$$\
$$S_{1, 4, 4} = 1$$\
$$S_{4, 1, 1} = 1$$\
$$S_{4, 1, 4} = 1$$\
$$S_{4, 4, 1} = 1$$\
$$S_{4, 4, 4} = 1$$

Finding a more efficient matrix multiplication algorithm is equivalent to reducing the number of such updates needed to go from $$S=0$$ to $$S=T$$.
Each update involves a series of arithmetic operations with one scalar multiplication:
$$u$$ specifies a set of elements in $$A$$ to be linearly combined.
$$v$$ specifies a set of elements in $$B$$ to be linearly combined.
The product of these two combinations contributes to elements of $$C$$ as specified by $$w$$.

We can stack these vectors into three matrices $$U$$, $$V$$, and $$W$$ Following this method, Strassen's algorithm (shown above) can be represented as:\
![](/assets/images/uvw.png){: width="300"}



# Describing this as a game

It should now be clearer how this can be cast as a reinforcement learning problem. It will be more convenient to set the initial state as $$S=T$$ and subtract, rather than add, the $${\bf u}$$, $${\bf v}$$, and $${\bf w}$$ contributions at each step. Our goal is to find the smallest number steps necessary to arrive at $$S=0$$.

At first glance it might seem that there is little coherent structure to efficient algorithms and given the discrete nature of the steps it will be hard to do better than trial-and-error.
In Strassen's algorithm for instance $$a_1b_4$$ contributes to three intermediate variables ($$m_1$$, $$m_3$$, and $$m_5$$) despite not contributing to any of the elements of $$C$$.


1. Build a policy model to choose an action $$\{ {\bf u,  v,  w}\}$$ given a state $$S$$.
2. Define a sufficiently dense reward function to provide feedback to the policy model.
3. Develop a RL/exploration algorithm to efficiently search for low-rank decompositions.
4. Supplement the RL problem with a supervised learning problem on known decompositions.


## Reward function

Since the goal is to minimize the number of steps taken to reach $$S=0$$, AlphaTensor provides a reward of $$-1$$ for each step taken. In practice, games are terminated after a finite number ($$R_{limit}$$) of steps. If we still have a non-zero tensor $$S$$ at this point, an additional reward of $$-\gamma(S)$$ is given, equal to "an upper bound on the rank of the terminal tensor." In simpler terms, $$\gamma(S)$$ is the number of non-zero entries remaining in $$S$$ since we know that each of these could be eliminated by a single update. Note that this terminal reward plays an important role in creating a dense reward function. Without it, the agend would only recieve useful feedback when it reaches the zero tensor within $$R_{limit}$$ steps - a sparse reward on its own. ([Implementation of terminal reward][code-terminal-reward])

## Supervised learning

While it is NP-hard to decompose a given tensor $$T$$ into factors, it is straightfoward to do the inverse: to construct a tensor $$D$$ from a given set of factors $$\{({\bf u}^(r), {\bf v}^(r), {\bf w}^(r))\}^R_{r=1}$$. This suggests a way to create synthetic demonstrations for supervised training - a set of factors is sampled from some distribution, and the related tensor $$D$$ is given as an initial condition to the actor network, which is then trained to output the correct factors. AlphaTensor generates a large dataset of such demonstrations and uses a mixed training strategy, alternating between training on supervised loss on the demonstrations and reinforcement learning loss on the target tensor $$T$$. This was found to substantially outperform either strategy separately. ([Implementation of synthetic demonstrations][code-synth-demos])


## Network Architecture and Training

The AlphaTensor network consists of three components:
1. A [torso][code-torso], which takes information about the current state and produces an embedding.
2. A [policy head][code-policy], which takes the embedding produced by the torso and generates a distribution over candidate actions.
3. A [value head][code-value], which takes the embedding produced by the torso and generates a distribution over candidate actions.

In the rest of this secion I'll give a brief overview of the architecture with links to my implementation. The network is quite complex (particularly the torso) and I won't attempt to cover all the details - to fully understand it I recommend both the paper and the pseudocode provided in the [Supplementary Information][alphatensor-supplementary].

![](/assets/images/network_architecture.png){: width="600"}


### Training Process

The network is trained on a dataset which initially consists of synthetic demonstrations. Training is done by teacher-forcing - loss is computed for each action in the ground-truth training sequence given the previous ground-truth actions. Note that loss os computed both for the policy head (based on the probability assigned to the next ground-truth action) and for the value head (comparing the output value distribution to the ground-truth rank of the current tensor state).

Periodically (after a given number of epochs), a MCTS is performed, starting from $T$. This is the step in which we actually use the model to explore and look for a solution to the problem we are interested in. Note that MCTS uses both the policy and value heads in deciding which directions to explore. All of the played games are added to a buffer, and the game with the best reward is added to a separate buffer. Both of these buffers are merged with the training dataset and eventually training is performed on fixed proportions of synthetic demonstrations, played games, and "best" played games.


### Torso

The torso converts the current state of $$S$$ (to be more precise, its past $$n$$ states), as well as any scalar inputs (such as the time index of the current action), to an embedding that feeds into the policy and value heads. It projects the $$4 \times 4 \times 4$$ tensor onto three $$4 \times 4$$ grids, one along each of its three directions. Following this, [attention-based blocks][code-attention] are used to propagate information between the three grids. A [block][code-torso-attention] has three stages - in each stage one of the three pairs of grids is concatenated and [axial attention][axial-attn] is applied. The output of the final block is flattened to an embedding vector which is the output of the torso.

![](/assets/images/torso_architecture.png){: width="800"}

### Policy Head

The policy head is responsible for converting the torso's output into a distribution over the action space of the factors $$\{ {\bf u,  v,  w}\}$$ which we can run backpropagation on (during the training step) and sample from (during the action step used in MCTS). However, this action space is far too large for us to represent the distribution explicitly. Consider multiplication of $$5 \times 5$$ matrices. In this case each factor is of length $$25$$. In the AlphaTensor paper, entries in the factors are restricted to the five values $$(-2, -1, 0, 1, 2)$$. Thus the cardinality of the action space is $$5^{3 \cdot 25} \approx 10^{52}$$.

The solution is to use a transformer architecture to represent an autoregressive policy. In other words, an action is produced sequentially, with each token in the factors drawn from a distribution that is conditioned on the previous tokens (via self-attention), as well as on the embedding produced by the torso (via cross-attention). Naively, we might treat each of the $$75$$ entries in the three factors as a token. However, now we have moved from an enormous action space to the opposite extreme, a transformer with a vocabulary size of only $$5$$! Recall that transformers learn embeddings for each "word" in the vocabulary- the benefit of this is most evident for larger vocabularies. Note that for any sequential data, we can use various representations that trade off between vocabulary size and sequence length. In this example, we can split the factors into chunks of 5 entries and represent each chunk as a token. With this approach, the vocabulary size (i.e. the number of distinct values a chunk can take on) increases to $$5^5 = 3125$$ and the sequence length decreases to $$15$$. The vocabulary size is still small enough to learn token embeddings over, but we have also reduced the context length that the transformer must learn to attend to.


![](/assets/images/policy_head.png){: width="600"}

### Value Head

The value head is a multilayer perceptron whose output is an estimate of the distribution of returns from the current state. This is expressed as a series of evenly spaced quantile values. The value head is trained against ground truth values using  [quantile regression][code-quantile] ([reference][distributional-rl]).

![](/assets/images/value_head.png){: width="600"}


### Training AlphaTensor

Now that we have all the main parts, let's put them together!


...

## Exploration via Monte Carlo Tree Search
used in [AlphaZero][alphazero] and extended [here][muzero]
...




[alphatensor-blog]: https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor
[alphatensor-nature]: https://www.nature.com/articles/s41586-022-05172-4
[alphatensor-supplementary]: https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-022-05172-4/MediaObjects/41586_2022_5172_MOESM1_ESM.pdf
[strassen]: https://eudml.org/doc/131927
[alphazero]: https://www.science.org/doi/10.1126/science.aar6404
[muzero]: https://arxiv.org/abs/2104.06303
[axial-attn]: https://arxiv.org/abs/1912.12180
[distributional-rl]: https://arxiv.org/abs/1710.10044


[assembly-ai]: https://www.assemblyai.com/blog/deepminds-alphatensor-explained/

[my-repo]: https://github.com/kurtosis/mat_mul
[code-synth-demos]: https://github.com/kurtosis/mat_mul/blob/7fa10f5fd351bff72712b122888ee220354f5e45/datasets.py#L20
[code-terminal-reward]: https://github.com/kurtosis/mat_mul/blob/7fa10f5fd351bff72712b122888ee220354f5e45/act.py#L59
[code-torso]: https://github.com/kurtosis/mat_mul/blob/7fa10f5fd351bff72712b122888ee220354f5e45/model.py#L99
[code-torso-attention]: https://github.com/kurtosis/mat_mul/blob/7fa10f5fd351bff72712b122888ee220354f5e45/model.py#L71
[code-value]: https://github.com/kurtosis/mat_mul/blob/7fa10f5fd351bff72712b122888ee220354f5e45/model.py#L211
[code-policy]: https://github.com/kurtosis/mat_mul/blob/7fa10f5fd351bff72712b122888ee220354f5e45/model.py#L283
[code-attention]: https://github.com/kurtosis/mat_mul/blob/7fa10f5fd351bff72712b122888ee220354f5e45/model.py#L71
[code-quantile]: https://github.com/kurtosis/mat_mul/blob/7fa10f5fd351bff72712b122888ee220354f5e45/model.py#L300

