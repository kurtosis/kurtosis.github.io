---
layout: post
title:  "DeepMind's AlphaTensor"
date:   2023-07-01 14:23:37 -0400
categories: alphatensor
usemathjax: true
---
DeepMind announced [AlphaTensor][alphatensor-blog] in October 2022 ([Nature publication][alphatensor-nature]). I have implemented the algorithm [here][my-repo]


# Addition and Multiplication

The importance of the order of operations stems from the simple fact that multiplication is a more expensive operation than addition. To build some intuition, consider which of the following arithmetic calculations would take more effort to solve:\
$$15047 + 30821$$\
or \
$$15047 \times 30821$$

Clearly the multiplication would be harder (you might even be able to do the addition in your head, but for the multiplication this would require a strong memory!).

Consider two integers of N digits each. The cost of adding these numbers is $$O(N)$$ - we first add the digits in the ones place, then those in the tens place, etc. The cost of multiplying the numbers is $$O(N^2)$$ - we must multiply the "ones" digit of the first number by every digit in the second number, then do the same for the "tens" digit, etc. More generally, if we denote the magnitude of an integer as $$M$$ then the number of digits scales as $$log(M)$$. Thus, addition is $$O(log(M))$$ and multiplication is $$O(log(M)^2)$$. The key point here is the quadratic relation between multiplication and addition - this holds whether the numbers are of different magnitudes, are expressed in bits rather than base 10, or are floating point rather than integers.

Another point to consider is the number of operations. Consider calculating the following:\
$$(15047 + 30821) \times (39012 + 82615)$$\
Most of us would instinctively perform two additions followed by a single multiplication, rather than the more arduous task of four multiplications followed by three additions (or one addition, followed by two multiplications, followed by another addition). The key point here is that for complex arithmetic calculations there are multiple paths to arrive at the answer, some of which are more efficient than others. While each operation has a cost, generally reducing the number of multiplications is most important to reducing the overall cost.

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


Consider a process in which we first set a tensor $$S$$ equal to the zero tensor and then sequentially modify it by choosing three vectors $$\bf{u}$$, $$\bf{v}$$, and $$\bf{w}$$, each of length 4, and performing the following update:\
$$S_{i, j, k} \leftarrow S_{i, j, k} + u_iv_jw_k$$


For example, applying the following vectors to $$S=0$$:\
$$u = (1, 0, 0, 1)$$\
$$v = (1, 0, 0, 1)$$\
$$w = (1, 0, 0, 1)$$\
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

It should now be clearer how this can be cast as a reinforcement learning problem. It will be more convenient to set the initial state as $$S=T$$ and subtract, rather than add, the $${u,v,w}$$ contributions at each step. Our goal is to find the smallest number steps necessary to arrive at $$S=0$$.

At first glance it might seem that there is little coherent structure to efficient algorithms and given the discrete nature of the steps it will be hard to do better than trial-and-error.
In Strassen's algorithm for instance $$a_1b_4$$ contributes to three intermediate variables ($$m_1$$, $$m_3$$, and $$m_5$$) despite not contributing to any of the elements of $C$.



1. Build a policy model to choose an action $$\{u, v, w\}$$ given a state $$S$$.
2. Define a sufficiently dense reward function to provide feedback to the policy model.
3. Develop a RL/exploration algorithm to efficiently search for low-rank decompositions.
4. Supplement the RL problem with a supervised learning problem on known decompositions.


## Reward function

Since the goal is to minimize the number of steps taken to reach $$S=0$$, AlphaTensor provides a reward of $$-1$$ for each step taken. In practice, games are terminated after a finite number ($$R_{limit}$$) of steps. If we still have a non-zero tensor $$S$$ at this point, an additional reward of $$-\gamma(S)$$ is given, equal to "an upper bound on the rank of the terminal tensor." In simpler terms, $$\gamma(S)$$ is the number of non-zero entries remaining in $$S$$ since we know that each of these could be eliminated by a single update. Note that this terminal reward plays an important role in creating a dense reward function. Without it, the agend would only recieve useful feedback when it reaches the zero tensor within $$R_{limit}$$ steps - a sparse reward on its own.

## Supervised learning

While it is NP-hard to decompose a given tensor $$T$$ into factors, it is straightfoward to do the inverse: to construct a tensor $$D$$ from a given set of factors $$\{(\bf{u}^{(r)}, \bf{v}^{(r)}, \bf{w}^{(r)})\}^{R}_{r=1}$$. This suggests a way to create synthetic demonstrations for supervised training - a set of factors is sampled from some distribution, and the related tensor $$D$$ is given as an initial condition to the actor network, which is then trained to output the correct factors. AlphaTensor generates a large dataset of such demonstrations and uses a mixed training strategy, alternating between training on supervised loss on the demonstrations and reinforcement learning loss on the target tensor $$T$$. This was found to substantially outperform either strategy separately.


## The Policy and Value Networks

...

## Exploration via Monte Carlo Tree Search
...


[alphatensor-blog]: https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor
[alphatensor-nature]: https://www.nature.com/articles/s41586-022-05172-4
[my-repo]: [https://github.com/kurtosis/mat_mul]
[strassen]: [https://eudml.org/doc/131927]
