---
layout: post
title:  "Generative Discriminative"
date:   2021-11-08 17:45:00 +0200
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

* `Notes from "On Discriminative vs. Generative classifiers: A comparison of logistic regression and naive Bayes (Ng et al., (2002))" and http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf`

* `Generative classifiers` learn a model of the **joint probability**, $$p(x,y)$$, of the inputs $$x$$ and the label $$y$$, and make their predictions by using **Bayes rules** to calculate $$p(y \vert x)$$, and then picking the most likely label $$y$$.

* `Discriminative classifier` models the posterior probability $$p(y \vert x)$$ directly, or learn a direct map from inputs $$x$$ to the class labels.

* Vapnik: `"one should solve the problem directly and never solve a more general problem as an intermediate step".`

* The VC dimension is roughly linear - or at most some low-order polynomial - in the number of parameters, and it is known that sample complexity in the discriminative setting is liner in the VC dimension.

