---
layout: post
title:  "DeepMind's AlphaTensor"
date:   2023-07-01 14:23:37 -0400
categories: alphatensor
---
DeepMind announced [AlphaTensor][alphatensor-blog] in October 2022 ([Nature publication][alphatensor-nature]). I have implemented the algorithm [here][my-repo]


# Addition and Multiplication

The importance of the order of operations stems from the simple fact that multiplication is a more expensive operation than addition. To build some intuition, consider which of the following arithmetic calculations would take more effort to solve:

(15047 + 30821)

(15047 * 30821)

Clearly the multiplication would be harder (you might even be able to do the addition in your head, but for the multiplication this would require a strong memory!).

Consider two integers of N digits each. The cost of adding these numbers is O(N) - we first add the digits in the ones place, then those in the tens place, etc. The cost of multiplying the numbers is O(N^2) - we must multiply the "ones" digit of the first number by every digit in the second number, then do the same for the "tens" digit, etc. More generally, if we denote the magnitude of an integer as M then the number of digits scales as log(M). Thus we can say that addition is O(log(M)) and multiplication is O(log(M)^2). The key point here is the quadratic relation between multiplication and addition - this holds whether the numbers are of different magnitudes, are expressed in bits rather than base 10, or are floating point rather than integers.

Another point to consider is the number of operations. Consider calculating the following:
(15047 + 30821) * (39012 + 82615)
Most of us would instinctively perform two additions followed by a single multiplication, rather than the more arduous task of four multiplications followed by three additions (or one addition, followed by two multiplications, followed by another addition). The key point here is that for complex arithmetic calculations there are multiple paths to arrive at the answer, some of which are more efficient than others. While each operation has a cost, generally reducing the number of multiplications is most important to reducing the overall cost.

# Matrix Multiplication

Consider matrix multiplication for two 2x2 matrices.

![](/assets/images/two_by_two.png){: width="300"}

The standard way of computing this is:

c1 = a1 * b1 + a2 * b3, etc.

Each element of the product requires two multiplications, resulting in eight multiplications overall. In 1969, [Strassen][strassen] showed that this can be computed using a two-level method which requires only seven multiplications.

![](/assets/images/strassen_algo.png){: width="200"}


In general, the standard way of multipling matrices of sizes (n x m) and (m x p), requires (n x p) x m scalar multiplications. It turns out that for most cases which have been examined, it is possible to perform the operation with substantially fewer scalar multiplications. For example, the AlphaTensor paper reported that the case (n=4, m=5, p=5) can be computed with 76 multiplications rather than 100.

![](/assets/images/best_ranks.png){: width="400"}

# Framing the tensor product as a set of discrete moves

The next step is to frame this search for "efficient" solutions with few multiplications (known as low-rank decompositions) as a problem that is amenable to machine learning. Looking at Strassen's algorithm, this may seem challening at first glance, as there are an enormous number of discrete permutations to choose from and it is not obvious what sort of gradient we might perform gradient descent over.

To start, consider describing matrix multiplication by a three-dimensional tensor, T, in which the first index refers to an element of $A$, the second index refers to an element of $B$ and the third index refers to an element of $C$.

![](/assets/images/tensor_3d.png){: width="300"}

For the (2,2,2) multiplication we know that:
c1 = a1 * b1 + a2 * b3
and this is represented by the following entries in T:

T(1, 1, 1) = 1

T(2, 3, 1) = 1

T(i, j, 1) = 0 for all other i, j

we can fill out the remaining entries in T given the definitions of c2, c3, and c4.

Let

T_(i, j, k) += u_i * v_j * w_k


Let's start with T being a zero matrix and consider the following vectors:
u = (1, 0, 0, 1)
v = (1, 0, 0, 1)
w = (1, 0, 0, 1)

Applying these to T results in:

T_(1, 1, 1) = 1

T_(1, 1, 4) = 1

T_(1, 4, 1) = 1

T_(1, 4, 4) = 1

T_(4, 1, 1) = 1

T_(4, 1, 4) = 1

T_(4, 4, 1) = 1

T_(4, 4, 4) = 1


The basic step is as follows:

u - add elements in A

v - add elements in B

multiply these two sums

w - add the product to elements in C

Following this method, Strassen's algorithm can be represented by the following U, V, and W matrices:
![](/assets/images/uvw.png){: width="300"}

Each of the seven columns defines a single multiplication which contributes to $C$.


# Describing this as a game

We can now begin to see how this might be casted into a sequential move game which is amenable to reinforcement learning. Starting with T as a zero tensor, we choose a "move" defined by the vectors {u, v, w} and update T accordingly. We continue to select moves until T is equal to the target tensor which defines our (n, m, p) matrix multiplication. At this point we have "won" the game and receive a "reward" that is equal to -1 * the number of moves we have taken (meaning that fewer moves is better.)

It turns out to be more convenient to set the initial value of T to the target tensor, and subtract rather than add the {u,v,w} contributions, so that our goal for T to equal the zero tensor. In this way, the goal state is the same for any matrix multiplication we wish to solve and only the initial state differs.




Learn a strategy to make a move for each input
The goal is to get a matrix to zero - we can provide some training examples
We also can let it explore and give feedback, score

reward - not just all or nothing, approximate reward, upper bound


[alphatensor-blog]: https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor
[alphatensor-nature]: https://www.nature.com/articles/s41586-022-05172-4
[my-repo]: [https://github.com/kurtosis/mat_mul]
[strassen]: [https://eudml.org/doc/131927]
