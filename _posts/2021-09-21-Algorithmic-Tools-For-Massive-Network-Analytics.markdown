---
layout: post
title:  "Algorithmic Tools for Massive Network Analytics"
date:   2021-09-21 12:05:00 +0200
categories: Algorithms, Social-Network-Analysis
description: Network analysis aims at finding interesting properties hidden in the linked structure. This analysis is challenging from a computational point of view due to the sheer size of the networks and the combinatorial nature of the corresponding graph problems.
---

<head>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">

  </script>
</head>



# Index
1. [Course home-page][course-site]
2. [Computing Diameter](#computing-the-diameter-of-real-world-graphs)
3. [Graphs and Compression](#graphs-and-compression)

## Graph Theory Basic Theory:

#### Edge Direction
1. Undirected
2. Directed

#### Density
The **density** of a graph with *n* nodes is the number of edges in the graph divided by the number of edges in the clique of *n* nodes:

$$ d = \frac{\text{# edges}}{\text{# max edges}} $$

where $$\text{# max edges}$$ is $$n(n-1)$$ if the graph is directed, $$\frac{n(n-1)}{2}$$ otherwise.

#### Node labels
1. Labeled
2. Unlabeled

#### Edge Weights
1. Unweighted
2. Weighted

#### Degree
The degree of a node is the number of edges adjacent to it, that is, the number of its **neighbors**.
In the case of **directed graphs**, we distinguish between:
1. the *in-degree* of a node;
2. the *out-degree* of a node.

#### Power law

TODO

### Simple Graphs
If a graph has no loop and has no multi-edge (i.e., several edges between the same two nodes), it is said to be **simple**

#### Paths
If a node *x* is linked to a node *y* and *y* is linked to another node *z*, then we say that there is a path from *x* to *z*. The number of edges in a path is called the **length** of the path.

#### Connectivity
1. If, **for any pair of nodes**, there exists a path between them, the the graph is said to be **connected**;
2. For directed graph, a graph is **strongly connected** if, for any pair of nodes *x* and *y*, there exist a path from *x* to *y* and vice-versa.
3. If a directed graph is not strongly connected but removing the directions of the edges the resulting graphs is connected, then we say that the graph is **weakly connected**.

#### The Giant Component
A giant component is a (strongly) connected component of a given (directed) graph that contains a constant fraction of the entire graph's nodes.


## Computing the Diameter of Real-World Graphs
Given a graph $$ G = (V,E) $$ undirected connected.
The distance $$ d(u,v) $$ is the number (sum of the weights) of edges along shortest path from $$ u $$ to $$ v $$.

Thus, the **diameter** is defined as:

$$ D = \max_{u,v \in V} d(u,v) $$

which is to say the longest possible shortest path between any two nodes in the graph.

The **eccentricity** of a node $$ u $$, $$ \text{ecc} = \max_{v \in V}d(u,v) $$: in how many hops $$ u $$ can reach any node?

$$ D = \max_{u \in V} \text{ecc}(u) $$

### Strong Exponential Time Hypothesis (SETH)
`Unless the SET Hypothesis is false, deciding whether a graph has diameter 2 or 3 requires` $$ \Omega{n^2} $$`. Informally, SETH says that SAT cannot be solved in sub-exponential time.
By this reduction, unless SETH fails, `$$ \Omega{n^2} $$` time is required to get a `$$ (3/2 - \epsilon)$$`-approximation algorithm for computing the diameter even in the case of sparse graphs.`

Cool algorithms to compute Graphs' Diameter:
1. $$ n $$-BFS
2. ...
3. 2-sweep
4. iFUB (Iterative Fringe Upper Bound)

## Graphs and Compression


#### Sizes
Usually graphs are very sparse: a sparse graph is one with $$ O(n) $$ arcs, instead of $$ O(n^2))).

#### Graph
A graph $$ G = (V_G,E_G) $$ is defined by:
1. A set $$ V = V_G $$ of nodes;
2. A set $$ E = E_G \subseteq V \times V $$ of arcs (ordered pairs of nodes)

The transpose $$G^T = (V, E^T)$$ of $$ G = (V,E) $$ is defined by:

$$
E^T = \{(y,x)|(x,y)\in E\}
$$

A graph is symmetric iff $$ G = G^T$$.

We similarly define $$G^S=(V,E \cup E^T)$$ (the symmetric closure of $$G$$).
1. Undirected graphs can be safely defined with symmetric graphs.
2. In an undirected graph, nodes are often called **vertices** and pairs of opposite arcs are called **edges**.

#### Multigraphs and Hypergraphs
1. More than one arc between two nodes (i.e., that $$E$$ is a multiset of pairs, instead of a set): **multigraph**.
2. If $$E$$ is not a set of pairs, but a set of $$r$$-tuples: **hypergraph**.


#### Labels
A graph can be labelled on its nodes and/or on its arcs. Node-labeling functions map nodes (or arcs) to a set of suitable labels. A special case of labelling is the assignment of real values, that is often called a *weighting function* (hence we call a graph a node-weighted or arc-weighted)

#### Paths
A path in $$ G $$ is a sequence $$ \pi = X_0, x_1, \dots, x_k \in V $$ such that $$ (x_i,x_{i+1}) \in E $$ for all $$ i = 0, \dots, k-1 $$. Furthermore, we say that:
1. $$ \pi $$ starts at node $$ x_0 $$ (a.k.a. the source of $$ \pi $$);
2. $$ \pi $$ ends at node $$ x_k $$ (a.k.a. the target of $$ \pi $$);
3. has length $$ \mid\pi\mid = k $$;
4. is *simple* if $$x_0,\dots,x_{k-1} $$ are all distinct;
5. it is a *cycle* iff $$ k > 0 $$ and the source and the target coincide;
6. if there is a path from $$ x $$ to $$ y $$, we say that $$ y $$ is reachable from $$ x $$;
6. if there is a cycle, $$ G $$ is called *cyclic*.

#### Strongly Connected and Weakly Connected Components
TODO

#### Neighborhoods and Degrees
Given $$G = (V,E) $$ and $$ x \in V $$, we define:
1. $$ N_G^-(x) = \{y\mid(y,x) \in E\} $$ the in-neighborhood of $$ x $$, predecessor of $$ x $$;
2. $$ N_G^+(x) = \{y\mid(y,x) \in E\} $$ the out-neighborhood of $$ x $$, successor of $$ x $$;
3. $$ d_G^-(x) = \mid N_G^-(x)\mid $$ in-degree of $$x$$;
4. $$ d_G^+(x) = \mid N_G^+(x)\mid $$ out-degree of $$x$$;

#### Local Clustering Coefficient
Given $$G = (V,E) $$ and $$ x \in V $$, we define:
* in-directed clustering coefficient of $$x$$: 

$$ c_G^- = \frac{\mid E_G \cap (N_G^-(x) \times N_G^-(x))\mid}{d_G^-(x)^2}$$ 

or if loop are not allowed: 

$$ c_G^- = \frac{\mid E_G \cap (N_G^-(x) \times N_G^-(x))\mid}{d_G^-(x) \dot (/d_G^-(x)-1)}$$

* for undirected graph: 

$$ c_G^- = \frac{2\mid E_G \cap (N_G(x) \times N_G(x))\mid}{d_G(x)\dot (d_g(x)-1)}$$


#### Graph Morphism


## Pattern Mining
1. TopKWY


[course-site]: https://sites.google.com/view/algtools