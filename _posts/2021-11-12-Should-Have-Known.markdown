---
layout: post
title:  "Should have known better"
date:   2021-11-12 13:55:00 +0200
---

<head>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
	<script id="MathJax-script" async
					src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">

	</script>
</head>


<style>

.center {
	display: block;
	margin-left: auto;
	margin-right: auto;
}
</style>

* `Some notes from Mathematical Statistics with Applications (Ramachandran and Tsokos, 2009)`

# Basic Concepts from Probability Theory

A (random) experiment satisfies three conditions:
1. the set of all possible outcomes is known in advance in each trial
2. in any particular trial, it is not known which particular outcome will happen
3. the experiment can be repeated under identical conditions

* `Informal definition of probability` - The probability of an event is a measure (number ) of the chance with which we can expect an event to occur.

* `Classical definition of probability` - If there are $$n$$ equally likely possibilities, of which one must occur, and $$m$$ of these are regarded as favorable to an event, or a "success", then the **probability** of the even or a "success" is given by $$\frac{m}{n}$$.

The classical probability concept is not applicable in situations where the various possibilities cannot be regarded as equally likely. In such cases, one could rely on the **frequency interpretation** of probability. The frequentistic view is a natural extension of the classical view of probability

* `Frequency definition of probability` - The probability of an outcome (event) is the proportion of times the outcome (event) would occur in a long run of repeated experiments (R. Von Mises, 1936)

The probability of head of biased coin could be defined as $$P(H) == \lim_{n \rightarrow \infty} (n(H)/n)$$. Although the frequency interpretation is often useful, it is not complete. Because of the condition of repetition under identical circumstances.

* `Axiomatic definition of probability` - Let $$S$$ be a sample space of an experiment. Probability $$P(\cdot)$$ is a real-valued function that assigns to each event $$A$$ in the sample space $$S$$ a number $$P(A)$$, called the probability of A, with the following conditions satisfied:
1. It is **non-negative**, $$P(A) > 0$$
2. Is is unity for a certain event. That is, $$P(S)=1$$
3. It is additive over the union of an infinite number of pairwise disjoint events: $$P(\cup_{i=1}^{\infty}A_i) = \sum_{i=1}^{\infty}P(A_i)$$. I.e., The probability of the union of two events can be obtained by adding the individual probabilities and subtracting the probability of their intersection: $$P(A \cup B) = P(A) +  P(B) - P(A \cap B)$$.

# Counting Techniques and calculation of probabilities

* `Multiplication Principle`: If the experiments $$A_1, A_2, \dots, A_m$$ contain, respectively, $$n_1, n_2, \dots, n_m$$ outcomes, such that for each possible outcome of $$A_1$$ there are $$n_2$$ possible outcome for $$A_2$$, and so on, then there are a total of $$n_1, n_2, \dots, n_m$$ possible outcomes for the composite experiment $$A_1, A_2, \dots, A_m$$.  Stated simply, it is the idea that if there are $$a$$ ways of doing something and $$b$$ ways of doing another thing, then there are $$a \times b$$ ways of performing both actions.

* `Sampling with Replacement and Order Matters`: when a random sample of size $$k$$ is taken with replacement from a total of $$n$$ objects and the objects being ordered, then there are $$n^k$$ possible ways of selecting $$k$$-tuples.

* `Sampling without Replacement and Order Matters`: if $$r$$ objects are chosen from a set of $$n$$ distinct objects without replacement, any particular ordered arrangement of these objects is called a **permutation**. The permutation of $$n$$ objects taken $$m$$ at a time is:

$$
_{n}P_m = \frac{n!}{(n-m)!}
$$

* `Sampling without Replacement and Order  does Not Matters`: When the order does not matter, the arrangement is called a **combination**. The number of combinations of $$n$$ objects taken $$m$$ at a time is:

$$
{n\choose m} = \frac{n!}{m!(n-m)!}
$$

* `Sampling with Replacement and Order does Not Matters`: in obtaining an unordered sample of size $$k$$, with replacement, from a total of $$n$$ objects, $$k-1$$ replacement will be made before sampling ceases. Thus $$n$$ is increased by $$k-1$$ so that sampling in this manner may be thought of as drawing an unordered sample of size $$k$$ from a population of size $$n+k-1$$. Hence, the number of possible samples can be obtained by using the formula:

$$
{n + k -1\choose k} = \frac{(n+k-1)!}{k!(n-1)!}
$$