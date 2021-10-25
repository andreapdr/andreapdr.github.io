---
layout: post
title:  "Pay your Attention"
date:   2021-09-17 16:27:00 +0200
categories: Multi-Modal, Aggregation-strategies
description: Attention is a scarce resource. On music or video streaming services, we either pay attention to their ads or pay money to hide them. For growth in the world of online video games, we either pay attention to participate in battles, which attract new gamers, or pay money to instantly become powerful. Nothing comes for free.
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

# **Index**
1. [Attention Scoring Functions](#attention-scoring-functions)
2. [Scaled Dot-Product Attention](#scaled-dot-product-attention)
3. [Additive Attention](#bahdanaus-attention-additive-attention)
4. [General Attention](#general-attention)
5. [Soft and Hard Attention](#soft-and-hard-attention)
6. [Self-Attention](#self-attention)
7. [Multi-Head Attention](#multi-head-attention)
8. [Cross-Modal Transformers](#cross-modal-transformers)
9. [AI Coffee Break][AI Coffee Break]
10. [LAION Dataset][LAION Dataset]


`"Attention is a scarce resource. On music or video streaming services, we either pay attention to their ads or pay money to hide them. For growth in the world of online video games, we either pay attention to participate in battles, which attract new gamers, or pay money to instantly become powerful. Nothing comes for free"`

| Biological | Driven by |Artificial |
| ---------- | ---------- |
| nonvolitional cue | saliency and conspicuity of objects in environment | keys|
| volitional cue | active research | queries |
| sensory inputs | perceived from the environment | values |

In a simple scenario, where only nonvolitional cues are available, to bias selection over sensory inputs, we can simply use a parametrized fully-connected layer or even non-parametrized max or average pooling. Therefore, what sets attention mechanism apart from those fully-connected layers or pooling layers is the inclusion of **volitional cues**.

Attention pooling can be treated as weighted average of inputs, where weights are uniform. In practice, attention pooling aggregates values using weighted averages, where weights are computed between the given query and different keys.

## **Nonparametric Attention Models**
#### Attention Pooling: Nadaraya-Watson Kernel Regression

The interaction between queries (volitional cues) and keys (nonvoltional cues) result in **attention pooling**. The attention pooling selectively aggregates values (sensory inputs) to produce the output.

The Nadaraya-Watson Kernel Regression model (1964) is a simple yet complete example for demonstrating machine learning with attention mechanism.

**Average Pooling**: simples estimator for a regression problem: using average pooling to average over all training inputs:

$$f(x)=\frac{1}{n} \sum_{i=1}^{n}y_i$$

<img src="/images/average_estimator.png" alt="Average Estimator" class="center">


#### Nonparametric Attention Pooling
Nadaraya-Watson kernel estimator: weight the outputs $$y_i$$ according to their input locations:

$$ f(x) = \sum_{i=i}^{n} \frac{K(x-x_i)}{\sum_{j=1}^{n}K(x-x_j)}y_i $$

where $$K$$ is a kernel. From the perspective of attention pooling, we can rewrite it as:

$$ f(x) = \sum_{i=i}^{n} \alpha(x,x_i)y_i $$

where $$x$$ is the query and $$(x_i,y_i)$$ is the key-value pair. The attention weight $$\alpha(x,x_i)$$ is assigned to the corresponding value $$y_i$$ based on the interaction between the query $$x$$ and the key $$x_i$$ modeled by $$\alpha(x,x_i)$$.
Now, consider a **Gaussian Kernel**:

$$ K(u) = \frac{1}{\sqrt{2\pi}}\exp(-\frac{u^2}{2}) $$

Plugging the Gaussian Kernel into the previous equation, we have:


$$ f(x) = \sum_{i=i}^{n} \alpha(x,x_i)y_i $$ 

$$ f(x) = \sum_{i=i}^{n} \frac{\exp(-\frac{1}{2}(x-x_i)^2)}{\sum_{j=1}^{n}\exp(-\frac{1}{2}(x-x_j)^2)}y_i $$ 

$$ f(x) = \sum_{i=i}^{n} \text{softmax}(-\frac{1}{2}(x,x_i))y_i $$ 

Here, a key $$x_i$$ that is closer to the given query $$x$$ will get **more attention weight** assigned to the key's corresponding value's $$y_i$$. Nadaraya-Watson kernel estimator is a **non-parametric** model (i.e., it depends on the number of training samples -- enjoys **consistency** benefit: given enough data the model converges to the optimal solution).

<img src="/images/nadaraya_estimator.png" alt="Nadaraya-Watson Estimator" class="center">

By looking at the attention weight heat-map, we can see that `the closer the query-key pair is, the higher the attention weight is in the attention score` (testing inputs are queries while training inputs are keys, and both inputs are sorted).

<img src="/images/w_heatmap.png" alt="Weight Heatmap" class="center">

To sum it up: Nadaraya-Watson proposed an estimator that uses a weighted average where weights corresponds to the **relevance** of the training instance to the query: $$\hat{y}=\sum_{i=1}^{n}\alpha(x,x_i)y_i$$. The weighting function $$\alpha(x,x_i)$$ encodes the relevance of the instance $$x_i$$ to predict for $$x$$. A common choice for for the weighting function is a normalized Gaussian kernel, though other similarity measures can also be used with normalization.

`Fast forward 50 years, attention mechanism in DNN can be viewed as a generalization that also allows learning the weighting function.`

## **Parametric Attention Models**

### Attention Scoring Functions
In Nadaraya-Watson, we used a Gaussian Kernel to model interactions between queries and keys. Treating the exponent of the Gaussian kernel as an *attention scoring function* (or *scoring function/aligning function* for short), the results of this function were essentially fed into a softmax operation. As a result, we obtained a probability distribution (attention weights) over values that are paired with keys. `In the end, the output of the attention pooling is simply a weighted sum of the values based on these attention weights. Furthermore, since attention weights are a probability distribution, the weighted sum is essentially a weighted average`.


Mathematically, suppose that we have a query $$\mathbf{q} \in \mathbb{R}^q$$ and $$m$$ key-value pairs $$(\mathbf{k_1},\mathbf{v_1}, \dots, \mathbf{k_m},\mathbf{v_m})$$, where any $$\mathbf{k_i}\in \mathbb{R}^k$$ and any $$\mathbf{v_i} \in \mathbb{R}^v$$. The attention pooling $$\mathcal{f}$$ is instantiated as a weighted sum of the values:

$$
\mathcal{f}(\mathbf{q}, (\mathbf{k_1},\mathbf{v_1})),\dots,(\mathbf{k_m},\mathbf{v_m}) = \sum_{i=1}^{m}\alpha(\mathbf{q},\mathbf{k_i})\mathbf{v_i} \in \mathbb{R}^v
$$

where the attention weights (scalar) for the query $$\mathbf{q}$$ and key $$k_i$$ is computed by the softmax operation of an attention scoring function $$\alpha$$ that maps two vectors to a scalar:

$$
\alpha(\mathbf{q},\mathbf{k_i}) = \text{softmax}(\alpha(\mathbf{q},\mathbf{k_i})) = \frac{\exp{(\alpha(\mathbf{q},\mathbf{k_i}))}}{\sum_{j=1}^{m}\exp{(\alpha(\mathbf{q},\mathbf{k_j}))}} \in \mathbb{R}
$$

`Different choices of the attention scoring function` $$\alpha$$ `lead to different behaviors of attention pooling`.

### Scaled Dot-Product Attention
* `Reference:` [Vaswani et al., (2017)][Vaswani et al., (2017)]

A more **computationally efficient design** for the scoring function can be simply dot product. However, such an operation requires that both the query and the key have the same vector length $$d$$. To ensure that the variance of the dot product still remains one regardless of the vector length, the scaled dot-product scoring function

$$
\alpha(\mathbf{q},\mathbf{k}) = \frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d}}
$$

divides the dot product by $$\sqrt{d}$$. In practice, we often think in minibatches for efficiency, such as computing the attention for $$n$$ queries and $$m$$ key-value pairs, where queries and keys are of length $$d$$ and values of length $$v$$. The scaled dot-product attention of queries $$\mathbf{Q} \in \mathbb{R}^{n\times d}$$, keys $$\mathbf{K}\in \mathbb{R}^{m\times d}$$, and values $$\mathbf{V}\in \mathbb{R}^{m\times v}$$ is:

$$
\text{softmax}(\frac{\mathbf{QK}^T}{\sqrt{d}})\mathbf{V} \in \mathbb{R}^{n\times v}
$$


### Bahdanau's Attention (Additive Attention)
* `Reference:` [Bahdanau et al., (2015)][Bahdanau et al., (2015)]

The work of [Bahdanau et al., (2015)][Bahdanau et al., (2015)] popularized the use of Neural (parametrized) attention mechanism (AM). In their work, AM was used in a seq2seq task, in order to let the decoder better **attend** the the hidden state of the encoder.

$$
c_{t'} = \sum_{t=1}^{T}\alpha(\mathbf{s_{t'-1}}, \mathbf{h_t})\mathbf{h_t}
$$

where the decoder hidden state $$ \mathbf{s_{t^{'}-1}} $$ at time step $$ t^{'}-1 $$ is the **query**, and the encoder hidden states $$ \mathbf{h_t} $$ are both the **keys** and the **values**, and the attention weight $$\alpha$$ is computed using an additive attention scoring function (a.k.a., the **alignment model**) defined as:


$$
\alpha(\mathbf{s_{t^{'}-1}}, \mathbf{h_t}) = v_a^T\text{tanh}(W_{a}s_{t'-1} + U_{a}h_t)
$$

[Note that][stack-bahdanau] the sum of two projection sis equivalent to a projection of the vector concatenation]:

$$
W_{a}s_{t'-1} + U_{a}h_t = (W_{a} \oplus U_{a}) \cdot (s_{t'-1} \oplus h_t)
$$

<figure>
<img src="/images/bahdanau_attention.png" alt="Additive Attention" class="center"  width="50%">
<figcaption>Caption: TODO</figcaption>
</figure>

Implementation details: the **decoder** is initialized with:
1. the **encoder** final-layer hidden states at all the time steps (as keys and values of the attention);
2. the **encoder** all-layer hidden state at the final time step (to initialize the hidden state of the decoder);
3. the **encoder** valid length (to exclude the padding tokens in attention pooling).
At each decoding time step, the decoder final-layer hidden state at the previous time step is used as the query of the attention. As a result, both the attention output and the input embedding are concatenated as the input of the RNN decoder.

`In the RNN encoder-decoder, Bahdanau attention treats the decoder hidden state at the previous time step as the query, and the encoder hidden states at all the time steps as both the keys and values.`

* `Implementation idea from PyTorch tutorial:` [Here][Bahdanau implementation]
* `Colab:` [Colab][Bahdanau Colab]


### General Attention
* `Reference:` [Luong et al., (2015)][Luong et al., (2015)]

Authors examine two simple and effective classes of attentional mechanism: a **global** approach which always attends to all source words, and a **local** one that only looks at a subset of source words at a time.

1. Global: it resembles the approach of Bahdanau et al., (2015) (i.e., additive attention), but it is simpler architecturally;
2. Local: It can be viewed as a blend of the **hard** and **soft** attention models proposed in [[Xu et al., (2015)][Xu et al., (2015)].

These approaches differ in how the attention score is computed, nevertheless they leverage it in the same fashion. Specifically, given the hidden state $$\mathbf{h_t}$$ and the context vector $$\mathbf{c_i}$$, the attentional hidden state $$\mathbf{\tilde{h}_t}$$ is computed via a simple **concatenation layer** (and hyperobolic tangent activation function).

$$
\mathbf{\tilde{h}_t} = \text{tanh}(\mathbf{W_c}\lbrack\mathbf{c_t};\mathbf{h_t}\rbrack)
$$

**Global Attention**: the idea is to consider all the hidden states of the encoder when deriving the context vector $$\mathbf{c_t}$$. A variable-length alignment vector $$\mathbf{\alpha_t}$$, whose size equals the number of time steps on the source side, is derived by comparing the current target hidden state $$\mathbf{h_t}$$ with each source hidden state $$\mathbf{\bar{h}_s}$$:

$$
\displaylines{\mathbf{\alpha_i}(s) = \text{align}(\mathbf{h_t}, \mathbf{\bar{h}_s}) \\ = \frac{\exp{(\text{score}(\mathbf{h_t}, \mathbf{\bar{h}_s}))}}{\sum_{s^{'}}\exp{(\text{score}(\mathbf{h_t}, \mathbf{\bar{h}_s}))}}}
$$

where *score* is a **content-based** function such as:

$$
\text{score}(\mathbf{h_t}, \mathbf{\bar{h}_s}) = 
  \begin{cases}
  \mathbf{h}^T_t \mathbf{\bar{h}_s}\\
  \mathbf{h}^T_t \mathbf{W_a}\mathbf{\bar{h}_s}\\
  \mathbf{v}^T_a \text{tanh}(\mathbf{W_a}\lbrack \mathbf{h_t}, \mathbf{\bar{h}_s} \rbrack)
  \end{cases}
$$

The drawback is that it has to attend to all the words o the source side for each target word, which is expensive and can potentially render it impractical to deal with long sequences.

**Local Attention**: it chooses to focus only on a small subset of the source positions per target word. This models takes inspiration from the trade-off between the **soft** and **hard** attentional models proposed by [Xu et al., (2015)][Xu et al., (2015)] to tackle the image caption generation task. Soft attention weights are placed *softly* over all patches in the source image (i.e., global attention), whereas hard attention selects one patch of the image to attend a time (i.e., more likely local attention). Nevertheless, differently from the Xu et al., local attention focuses on a small window of context and it is more easily differentiable.

In concrete, the model first generates an aligned position $$p_t$$ for each target word at time $$t$$. The context vector $$c_t$$ is then derived as a weighted average over the set of source hidden state within the window $$\lbrack p_t - D, p_t + D\rbrack$$ (where $$D$$ is determined empirically). Unlike the global approach, the local alignment vector $$a_t$$ is now fixed-dimensional, i.e., $$\mathbb{R}^{2D+1}$$.
There are two approaches to compute the aligned position $$p_t$$:

 **Monotonic alignment (local-m)**: where $$p_t = t$$, assuming that source and target sequences are roughly monotonically aligned;

 **Predictive alignment (local-p)**: the model predicts an aligned position as follows:

$$
p_t = S \times \text{sigmoid}(\mathbf{v_p^T}\text{tanh}(\mathbf{W_p h_t}))
$$

where $$W_p$$ and $$v_p$$ are model parameters which will be learned to predict positions. $$S$$ is the source sentence length. As a result of sigmoid, $$p_t \in \lbrack 0, S \rbrack $$. To favor alignment points near $$p_t$$, we place a Gaussian distribution centered around $$p_t$$:

$$
\alpha_t(s) = \text{align}(\mathbf{h_t}, \mathbf{\bar{h}_s})\exp{(-\frac{(s-p_t)^2}{2\sigma^2})}
$$

with $$\sigma = \frac{D}{2}$$.

<figure>
<img src="/images/luong.png" alt="General Attention">
<figcaption>Caption:</figcaption>
</figure>


### Soft and Hard Attention
* `Reference:` [Xu et al., (2015)][Xu et al., (2015)]

In the context of image caption generation, authors propose an attention-based (encoder-decoder) model that automatically learns to describe the content of images. They introduce two attention mechanism: (1) **soft** deterministic attention, trainable by standard back-propagation methods, and (2) **hard** stochastic (i.e., sampled-based) attention, trainable by maximizing an approximate variational lower bound (via RL techniques). The model does not explicitly use object detection (classical approach) but instead learns latent alignments from scratch. This allows the model to go beyond "objectness" (?) and learn to attend to abstract concepts.

Images are first encoded by a CNN in order to extract a set of feature vectors $$a$$ (annotation vectors). The encoder (or extractor) produces $$L$$ vectors, each of which is a $$D$$-dimensional representation corresponding to a part of the image.

$$
a = \{a_1, \dots, a_L\}, a_i \in \mathbb{R}^D
$$

The caption is then generated by an LSTM decoder that generates a word at each time step $$t$$ conditioned on a context vector $$\mathbf{\hat{z}_t}$$, the previous hidden state $$\mathbf{h_t}$$ and the previously generated word $$y_{t-1}$$.

The context vector $$\mathbf{\hat{z}_t}$$ is a dynamic representation of the relevant part of the image input at time $$t$$. Such vector is computed by an attention mechanism ($$\phi$$) taking into account the annotation vectors (image features) $$a_i, i=1,\dots,L$$.
For each location $$i$$, the mechanism generates a positive weight $$\alpha_i$$ which can be interpreted either as (1) the probability that location $$i$$ is the right place to focus for producing the next word (in the context of **hard** attention), or (2) as the relative importance to give to location $$i$$ in blending the annotation vectors together (in the context of **soft** attention).


`This paper first proposed the distinction between “soft” vs “hard” attention, based on whether the attention has access to the entire image (or generally, the whole input) or only a patch.`

**Stochastic Hard Attention**: **TODO**


**Deterministic Soft Attention**: the context vector $$\mathbf{\hat{z}_t}$$ is the expected vector over all of the annotations:

$$
\mathbb{E}_{p(s_t|a)} \lbrack \mathbf{\hat{z}_t} \rbrack = \sum_{i=1}^{L}\alpha_{t,i}\mathbf{a_i}
$$

This corresponds to feeding in a soft $$\alpha$$-weighted context into the system. **TODO**

The context vector is computed from stochastically sampled hidden states in the input sequence. This is accomplished using a multi-Bernoulli distribution parametrized by the attention weights ([Chaudhari et al., (2019)][Chaudhari et al., (2019)]).

**TLDR**: `in `**soft**` attention,` $$\alpha$$` are real-values, generally between 0 and 1. Attention scores are then used to compute a weighted average of the encoder hidden states. In `**hard**` attention, `$$\alpha$$` are sampled from a multi-Bernoulli. They take values `$$[0,1]$$`, thus discarding whole regions of the input space.`


### Self-Attention
* `Reference:` [Cheng et al., (2016)][Cheng et al., (2016)] - not sure about who first proposed self-attention **TODO**
 
In self attention mechanism, all tokens plays the both the role of key and query at the same time. Specifically, each query attends to all the key-value pairs and generates one attention output. Since the queries, the keys, and the values come from the same place, this is called **self-attention** (or intra-attention).

### Multi-Head Attention
Given the same set of queries, keys, and values we may want our model to combine knowledge from different behaviors of the same attention mechanism (i.e., we would like to have different attention heads to model different kinds of relation, such as short-range or long-range ones). 
To this end, instead of performing single attention pooling, queries, keys, and values can be transformed into $$h$$ independently learned linear projections. Then these $$h$$ projected queries, keys, and values are fed into attention pooling in parallel. In the end, the $$h$$ attention pooling outputs are concatenated and transformed with another learned linear projection to produce the final output. This design is called **multi-head** attention.

$$\mathbf{h_i} = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v}$$


[The Annotated Transformer]: https://nlp.seas.harvard.edu/2018/04/03/attention.html
[stack-bahdanau]: https://stats.stackexchange.com/questions/524039/why-is-bahdanaus-attention-sometimes-called-concat-attention
[Luong et al., (2015)]: https://arxiv.org/abs/1508.04025
[Xu et al., (2015)]: https://arxiv.org/abs/1502.03044
[Bahdanau et al., (2015)]: https://arxiv.org/abs/1409.0473
[Bahdanau implementation]: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
[Bahdanau Colab]: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/a60617788061539b5449701ae76aee56/seq2seq_translation_tutorial.ipynb
[Cheng et al., (2016)]: https://arxiv.org/abs/1601.06733
[Vaswani et al., (2017)]: https://arxiv.org/abs/1706.03762
[Chaudhari et al., (2019)]: https://arxiv.org/abs/1904.02874
[Tsai et al., (2019)]: https://arxiv.org/abs/1906.00295
[Su et al., (2019)]: https://arxiv.org/abs/1908.08530
[Rahman et al., (2019)]: https://arxiv.org/abs/1908.05787
[Wei et al., (2020)]: https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.html
[Ye et al., (2019)]: https://arxiv.org/abs/1904.04745

[Wang et al., (2018)]: https://arxiv.org/abs/1804.05448
[Song et al., (2021)]: https://arxiv.org/abs/2107.04548
[Taylor et al., (2019)]: https://arxiv.org/abs/1909.06442
[Mohla et al., (2020)]: https://ieeexplore.ieee.org/document/9150738
[Jaegle et al., (2021)]: https://arxiv.org/abs/2103.03206
[Zadeh et al., (2020)]: https://www.sciencedirect.com/science/article/pii/S1566253520303006
[Rahate et al., (2021)]: https://arxiv.org/abs/2107.13782
[Graves et al., (2016)]: https://www.cs.toronto.edu/~graves/icml_2006.pdf
[Messina et al., (2020)]: https://arxiv.org/abs/2008.05231
[Wang et al., (2019)]: https://arxiv.org/abs/1902.04094
[Lu et al., (2019)]: https://arxiv.org/abs/1908.02265

[MulT GitHub]: https://github.com/yaohungt/Multimodal-Transformer
[covarep]: https://ieeexplore.ieee.org/abstract/document/6853739
[facet]: https://imotions.com/biosensor/fea-facial-expression-analysis/
[glove]: https://aclanthology.org/D14-1162.pdf
[AI Coffee Break]: https://www.youtube.com/c/AICoffeeBreak/videos
[LAION Dataset]: https://laion.ai/laion-400-open-dataset/