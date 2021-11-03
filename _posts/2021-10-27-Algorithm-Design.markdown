---
layout: post
title:  "Algorithm Design"
date:   2021-10-27 10:10:00 +0200
categories: Algorithms
description: Some notes for Algorithm Design 2020-2021
---

<head>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">

  </script>
</head>

`Written Exam:`
1. **choose one of the topics discussed in class**
2. write a very **short to-do list** and ask the instructor for approval
3. if the instructor suggests some mods, modify the to-do list according to the instructor's comments and repeat step 2
4. if the chosen topic and the to-do list are approved, **expand the to-do list** into a more detailed to-do list and **repeat step 3**
5. make a **written report in English** and submit it to the instructor (**recall to add at least 20% new content, when compared to what seen in class**) (use Jupyter Lab to prepare your report (so as to mix Markdown, LaTeX and Python code).
6. meet the instructor to **read together** the report and get some comments on it
7. make the necessary mods.


### Topics

1. Randomization and indicator variables. Headphone problem.
2. Digital fingerprints. **Karp-Rabin algorithm**. Montecarlo and Las vegas algorithms.
3. **Randomized quicksort**. Universal hash family. 
4. Perfect hashing and randomized hash tables.  
5. Markov's inequality. K-wise limited independence. Cuckoo hashing.  
6. Cuckoo hashing (analysis). Randomized load balancing. Concentration bounds: Chebychev's inequality.  
7. Randomized load balancing (cont'd). Concentration bounds: Chernoff's bounds. 
8. **Bloom filters.**
9. **Randomized approximate dictionary**. Lower bound. Succinct rank data structure.  
10. Randomized approximate dictionary. Upper bound.
11. Application of fingerprints: rsync algorithm. 
12. **Approximate counting: Morris' counter**.  
13. Fully polynomial-time approximation scheme (FPTAS). Morris' counter (cont'd). 
14. Application of data stream statistics (part I). 
15. Boosting with Chernoff's bounds. Morris' counter (cont'd).  
16. Sketching algorithms and network analysis. FM-sketches. 
17. Application of data stream statistics (part II).  
18. **Count-Min sketches**. 
19. Extensions of Count-Min sketches. Interval queries. 
20. **Min-hash and network analysis**.  
21. Introduction to algorithmic game theory (part I)  
22. Systematic view: min-k sketches, bottom-k sketches, threshold-t sketches. 
23. Distance distribution in network. Azuma-Hoeffding inequality. Milgram's experiment. 
24. Introduction to algorithmic game theory (part II) 
25. **Graph diameter. Fine-grained complexity. SETH conditional lower bound.**
26. **Randomized approximation for the graph diameter.**  
27. Introduction to algorithmic game theory (part III)  
28. Randomized algorithm for min-cut in graphs. 
29. NP-hardness. Knapsack exact solutions (baseline, dynamic programming version 1 and 2).  
30. Stable matching and game theory.  
31. Approximation algorithms. Knapsack 2-approximations and FPTAS.  
32. Negative results for TSP approximation. Metric TSP 2-approximation. Max-cut 2-approximations: local search, greedy, random coin tossing. Recap. 
33. K-center graph clustering, and video summarization. 
34. Graph based community detection.  
35. Fixed-parameter tractable (FPT) algorithms. Kernelization. Branching and bounded recursion tree. Case study: min-vertex cover in graphs.  
36. Randomized FPT algorithms: color coding and randomized separation. Case study: longest path in graphs and subgraph isomorphism. 



### Grossi interesting papers


1. [K-plex cover pooling for graph neural networks][K-plex cover pooling for graph neural networks (2021)]
2. [String Sanitization: A Combinatorial Approach][String Sanitization: A Combinatorial Approach]
3. [Combinatorial Algorithms for String Sanitization][Combinatorial Algorithms for String Sanitization]
4. [On-Line Pattern Matching on Similar Texts][On-Line Pattern Matching on Similar Texts]


<figure>
<img src="/images/Comparison_computational_complexity.svg" alt="Comparison of computational complexity" class="center">
</figure>



## Randomized Algorithms (Chapter 12, by Motwani and Raghavan)

A **randomized algorithm** is one that makes random choices during its execution. The design and analysis of a randomized algorithm depends on establishing that is likely to behave *well* on every input. The likelihood in such a statement depends only on the probabilistic choices made by the algorithm during its execution and **not** on any assumption about the input.

It is important to distinguish a randomized algorithm from the average-case analysis of algorithms, where one analyzes an algorithm assuming that its input is drawn from a fixed probability distribution. With a randomized algorithm, in contrast, no assumption is made about the input.

Two benefits of randomized algorithms: **(1) simplicity**, and **(2) efficiency**.

### Sorting and Selection by Random Sampling

The main idea is behind this algorithm is the use of **random sampling**: a random chosen member of $$S$$ is unlikely to be one its largest or smallest elements; rather it is likely to be *near the middle*. Extending this intuition suggests that a random sample of elements from $$S$$ is likely to be spread *roughly uniformly* in $$S$$.

`Algorithm RQS (Random Quick Sort):`

```
Input: A set of number S;
Output: The elements of S sorted in increasing order.
  
  1. Choose an element y uniformly at random from S. Every element 
     in S has equal probability of being chosen.
  2. By comparing each element of S with y, determine the set S_1 of elements
     smaller than y and set S_2 of elements larger than y.
  3. Recursively sort S_1 and S_2. Output the sorted version of S_1, followed
     by y, and the the sorted version of S_2.
```
`My implementation:`
```python
import numpy as np

def RQS(S):
    """
    S: set on number S (i.e., no repeated numbers)
    Randomized Quick Sort
    """

    # If S has zero elements, return empty array (we have reached maximum depth)
    if len(S) == 0:
        return np.array([], dtype=int)
    
    # 1. Choose an element y uniformly at random from S: every element in
    # S has equal probability of being chosen.
    y = np.random.choice(S)


    # 2. By comparing each element of S with y, determine the set S1 of
    # elements smaller than y and the set S2 of elements larger
    # than y.
    s1_mask = S < y
    s2_mask = S > y
    s1 = S[s1_mask]
    s2 = S[s2_mask]

    # 3. Recursively sort S1 and S2 . Output the sorted version of S1 ,
    # followed by y, and then the sorted version of S2.
    return np.concatenate((RQS(s1), [y], RQS(s2)))

if __name__ == '__main__':
    rng = np.random.default_rng()
    S = rng.choice(1000, size=100, replace=False)
    print(S)
    print(RQS(S))
```

What can we prove about the running time of RQS?

Comparison are performed in step (2.) where we compare a randomly selected element to the remaining elements. For $$1 \leq i \leq n$$, let $$S_{(i)}$$ denote the element of rank $$i$$ (the $$i$$-th smallest element) in the set $$S$$. Define the random variable $$X_{ij}$$ to assume the value of $$1$$ if $$S_{(i)}$$ and $$S_{(j)}$$ are compared in an execution, $$0$$ otherwise.

The total number of comparison is:

$$
\mathbb{E} \left[ \sum_{i=1}^n \sum_{j>i}X_{ij} \right] = \sum_{i=1}^n \sum_{j>i} \mathbb{E} \left[ X_{ij} \right]
$$

$$
\mathbb{E} \left[ X_{ij} \right] = p_{ij} \times 1 + (1 - p_{ij}) \times 0 = p_{ij}
$$

to compute $$p_{ij}$$ we view the execution of RQS as a binary tree $$T$$, each node of which is labeled with a distinct element of $$S$$. Let $$\pi$$ be the permutation obtained by visiting the nodes of $$T$$ in increasing order of the level numbers, and in a left-to-right order within each call (the $$i$$-th level of the tree is the set of all nodes at distance $$i$$ from the root). Two observations:
1. There is a comparison between elements $$S_{(i)}$$ and $$S_{(j)}$$ if and only if $$S_{(i)}$$ or $$S_{(j)}$$ occurs earlier in the permutation $$\pi$$ than any elements $$S_{(l)}$$ such that $$ i < l < j$$.
2. Any of the elements $$S_{(i)}, S_{(i+1)}, \dots, S_{(j)}$$ is equally likely to be the first of these elements to be chosen as a partitioning element and hence, to appear first in $$\pi$$. Thus, the probability that this first element is either $$S_{(i)}$$ or $$S_{(j)}$$ is exactly $$2 / (j-i+1)$$.

It follows that $$p_{ij} = 2 / (j-i+1)$$. The expected number of comparison is:

$$
\sum_{i=1}^n \sum_{j>i} p_{ij} = \sum_{i=1}^n \sum_{j>i} \frac{2}{(j-i+1)}
$$ 

$$
\leq \sum_{i=1}^{n-1} \sum_{k=1}^{n-i} \frac{2}{(k+1)} \leq 2 \sum_{i=1}^{n} \sum_{k=1}^{n} \frac{1}{k}
$$

it follows that the expected number of comparisons is bounded above by $$2nH_n$$ where $$H_n$$ is the $$n$$-th harmonic number defined as $$H_n = \sum_{k=1}^{n} \frac{1}{k}$$.

`The expected number of comparisons in an execution of RQS is at most 2nH_n`

Now $$H_n = \ln n + \Theta(1)$$, so that the expected running time of RQS is $$\mathcal{O}(n\log{n})$$.

[K-plex cover pooling for graph neural networks (2021)]: https://link.springer.com/article/10.1007/s10618-021-00779-z
[String Sanitization: A Combinatorial Approach]: https://kclpure.kcl.ac.uk/portal/files/112391596/Strin_sanitization_ECMLPKDD19.pdf
[Combinatorial Algorithms for String Sanitization]: https://dl.acm.org/doi/pdf/10.1145/3418683
[On-Line Pattern Matching on Similar Texts]: https://kclpure.kcl.ac.uk/portal/files/76760410/On_Line_Pattern_Matching_GROSSI_Published_2017_GOLD_VoR_CC_BY.pdf
