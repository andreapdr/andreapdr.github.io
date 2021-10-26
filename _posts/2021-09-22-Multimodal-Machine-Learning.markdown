---
layout: post
title:  "Multimodal Machine Learning"
date:   2021-09-22 08:45:00 +0200
categories: Representation-Learning
description: Can't learn language from the radio.
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


### **TODO**: Look for Conferences, workshops...
### **TODO**: Order by publication date
### **TODO**: Data amount?

# **Index**:
1. [Tensor Fusion Network for Multimodal Sentiment Analysis (2017)](#tensor-fusion-network-for-multimodal-sentiment-analysis-2017)
2. [Memory Fusion Network for Multi-view Sequential Learning (2018)](#memory-fusion-network-for-multi-view-sequential-learning-2018)
3. [Multi-Attention Recurrent Network for Human Communication Comprehension (2018)](#multi-attention-recurrent-network-for-human-communication-comprehension-2018)
4. [Multimodal Language Analysis with Recurrent Multistage Fusion (2018)](#multimodal-language-analysis-with-recurrent-multistage-fusion-2018)
5. [MAG-BERT: Integrating Multimodal Information in Large Pretrained Transformers (2019)](#mag-bert-integrating-multimodal-information-in-large-pretrained-transformers-2019)
6. [Multimodal Transformer for Unaligned Multimodal Language Sequences (2020)](#multimodal-transformer-for-unaligned-multimodal-language-sequences-2020)
7. [VL-BERT: Pre-training of Generic Visual-Linguistic Representations (2019)](#vl-bert-pre-training-of-generic-visual-linguistic-representations-2019)
8. [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representation fro Vision-and-Language Tasks (2019)](#vilbert-pretraining-task-agnostic-visiolinguistic-representation-fro-vision-and-language-tasks-2019)
9. [Multi-Modality Cross-Attention Network for Image and Sentence Matching (2020)](#multi-modality-cross-attention-network-for-image-and-sentence-matching-2020)
10. [Cross-Modal Self-Attention for Referring Image Segmentation (2019)](#cross-modal-self-attention-for-referring-image-segmentation-2019)
11. [Learning Transferable Visual Models From Natural Language Supervision (2021)](#learning-transferable-visual-models-from-natural-language-supervision-2021)
12. [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks (2020)](#oscar-object-semantics-aligned-pre-training-for-vision-language-tasks-2020)
13. [VinVL: Revisiting Visual Representations in Vision-Language Models (2021)](#vinvl-revisiting-visual-representations-in-vision-language-models-2021)
14. [VirTex: Learning Visual Representations from Textual Annotations (2020)](#virtex-learning-visual-representations-from-textual-annotations-2020)
15. [UNITER: UNiversal Image-TExt Representation Learning (2020)](#uniter-universal-image-text-representation-learning-2020)
16. [Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training (2019)](#unicoder-vl-a-universal-encoder-for-vision-and-language-by-cross-modal-pre-training-2019)
17. [VisualBERT: A Simple and Performant Baseline for Vision and Language (2019)](#visualbert-a-simple-and-performant-baseline-for-vision-and-language-2019)
18. [LXMERT: Learning Cross-Modality Encoder Representations from Transformers (2019)](#lxmert-learning-cross-modality-encoder-representations-from-transformers-2019)
19. [Learning Visual Representations with Caption Annotations (2020)](#learning-visual-representations-with-caption-annotations-2020)
20. [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision (2021)](#simvlm-simple-visual-language-model-pretraining-with-weak-supervision-2021)
21. [Villa: Large-Scale Adversarial Training for Vision-and-Language Representation Learning (2020)](#villa-large-scale-adversarial-training-for-vision-and-language-representation-learning-2020)
22. [ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph (2021)](#ernie-vil-knowledge-enhanced-vision-language-representations-through-scene-graph-2021)
23. [UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning (2021)](#unimo-towards-unified-modal-understanding-and-generation-via-cross-modal-contrastive-learning-2021)
24. [VIVO: Visual Vocabulary Pre-Training for Novel Object Captioning (2021)](#vivo-visual-vocabulary-pre-training-for-novel-object-captioning-2021)
25. [VL-T5: Unifying Vision-and-Language Tasks via Text Generation (2021)](#vl-t5-unifying-vision-and-language-tasks-via-text-generation-2021)
26. [Seeing Out of tHe bOx: End-to-End Pre-training for Vision-Language Representation Learning (2021)](#seeing-out-of-the-box-end-to-end-pre-training-for-vision-language-representation-learning-2021)
27. [E2E-VLP: End-to-End Vision-Language Pre-training Enhanced by Visual Learning (2021)](#e2e-vlp-end-to-end-vision-language-pre-training-enhanced-by-visual-learning-2021)
28. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021)](#an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-2021)
29. [CoAtNet: Marrying Convolution and Attention for All Data Sizes (2021)](#coatnet-marrying-convolution-and-attention-for-all-data-sizes-2021)
30. [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering (2018)](#bottom-up-and-top-down-attention-for-image-captioning-and-visual-question-answering-2018)
31. [Watch, Listen, and Describe: Globally and Locally Aligned Cross-Modal Attentions for Video Captioning (2018)](#watch-listen-and-describe-globally-and-locally-aligned-cross-modal-attentions-for-video-captioning-2018)

### Tensor Fusion Network for Multimodal Sentiment Analysis (2017)

*  `Reference:` [Zadeh et al., (2017)][Zadeh et al., (2017)]

<figure>
<img src="/images/tfn.png" alt="TFN" class="center">
</figure>

The central challenge in multimodal sentiment analysis is to model the **inter-modality** and the **intra-modality** dynamics, as well. 

Previous works in multimodal sentiment analysis does not account for both intra-modality and inter-modality dynamics, instead they either perform early fusion or late fusion. Early fusion (usually concatenation) does not allow the intra-modality dynamics to be efficiently modeled. This is due to the fact that inter-modality dynamics can be more complex at input level and can dominate the learning process or result in overfitting. Late fusion, instead, consists in training unimodal classifiers independently and performing decision voting. THis prevents the model from learning inter-modality dynamics.

There are three major components in TFN:
1. Modality Embedding Subnetworks (i.e., unimodal embedders);
2. Tensor Fusion Layer;
3. Sentiment Inference Subnetwork (i.e., final classifier).

The core is the Tensor Fusion Layer which takes care of explicitly model the unimodal, bimodal and trimodal interactions using a 3-fold Cartesian product from modality embeddings.

It is defined as the vector field using three-fold Cartesian product:

$$
\{(z^l, z^v, z^a) | z^l \in \begin{bmatrix}\mathbf{z}^l\\1\end{bmatrix}, z^v \in \begin{bmatrix}\mathbf{z}^v\\1\end{bmatrix}, z^a \in \begin{bmatrix}\mathbf{z}^a\\1\end{bmatrix}\}
$$

The extra constat dimension with value 1 generates the unimodal and bimodal dynamics. This defintion is mathematically equivalent to a differentiable outer product between $$\lbrack z^l 1 \rbrack^T, \lbrack z^v 1\rbrack^T, \lbrack z^a 1 \rbrack^T$$:

$$
z^m = \begin{bmatrix}\mathbf{z}^l\\1\end{bmatrix} \otimes \begin{bmatrix}\mathbf{z}^v\\1\end{bmatrix} \otimes \begin{bmatrix}\mathbf{z}^a\\1\end{bmatrix}
$$

where $$\otimes$$ indicates the outer product between vectors.



**NB:** after the TFN, the resulting tensor is **flattened** before being fed to the Sentiment Inference Subnetwork.


### Memory Fusion Network for Multi-view Sequential Learning (2018)

*  `Reference:` [Zadeh, Liang, et al., (2018)][Zadeh, Liang, et al., (2018)]
*  `Memory is used, together with attention, when the attention mechanism is not able to look at all the hidden states (b/c they are discarded or whatsoever) - usually deployed with RNN, since Transformer architectures ditch the recurrent behavior and process the whole input in a single time-step`
* `Difference wrt MARN: MFN has three separate LSTMs to model each modality separately and a multi-view gated memory to synchronize among them.`

<figure>
<img src="/images/mfn.png" alt="MFN" class="center">
</figure>


The Memory Fusion Network (MFN) is a recurrent model for multi-view (i.e., multi-modal) sequential learning (i.e., anything that depends on time, sequential data such as text, speech, video etc.) that consists of three main components:
1. **System of LSTMs**: multiple LSTMs, one for each of the views. Each learner encodes the view-specific dynamics and interactions;
2. **Delta-Memory Attention Network**: attention mechanism designed to discover both cross-view and temporal interactions across different dimensions of memories;
3. **Multi-view Gated Memory**: a unifying memory that stores the cross-view interactions over time.


**Delta-Memory Attention Network**: goal of the DMAN is to outline the cross-view interactions at time-step $$t$$ between different view memories in the system of LSTMs. 
The input to the DMAN is the **concatenation** of memories at time $$t-1$$ and $$t$$, denoted as $$c^{\lbrack t-1, t\rbrack}$$. These memories are passed to a NN $$\mathcal{D}_a : \mathbb{R}^{2d_{c}} \rightarrow \mathbb{R}^{2d_{c}}, d_c = \sum_{n} d_{c_n}$$.

$$
\alpha^{\lbrack t-1, t\rbrack} = \mathcal{D}_{\alpha}(c^{\lbrack t-1, t\rbrack})
$$

$$\alpha^{\lbrack t-1, t\rbrack}$$  are softmax activated scores. The attended memory $$\hat{c}$$ output by the DMAN module is defined as:

$$
\hat{c}^{\lbrack t-1, t\rbrack} = c^{\lbrack t-1, t\rbrack} \cdot \alpha^{\lbrack t-1, t\rbrack}
$$


**Multi-view Gated Memory**: 


Multi-view Gated Memory $$u$$ is the neural component that stores a history of cross-view interactions over time. The current weighted memory $$\hat{c}^{\lbrack t-1, t\rbrack}$$ is first used as input to a neural network $$\mathcal{D}_u : \mathbb{R}^{2 \times d_c} \rightarrow \mathbb{R}^{d_\text{mem}}$$ to generate a cross-view update proposal $$\hat{u}^t$$.

The Multi-view Gated Memory is controlled by two parameters $$\gamma_1, \gamma_2$$. At each time step $$t$$, $$\theta_1$$ assigns how much of the current state of the Multi-view Gated Memory to remember to update based on the update proposal $$\hat{u}$$. Parameters $$\gamma_1, \gamma_2$$ are each controlled by a neural network $$\mathcal{D}_{\gamma_1}, \mathcal{D}_{\gamma_2} : \mathbb{R}^{2 \times d_c} \rightarrow \mathbb{R}^{d_\text{mem}}$$.

The outputs of the MFN are the final state of the Multi-view Gated Memory $$u^T$$ and the outputs of each of the $$n$$ LSTMs concatenated ($$\bigoplus$$) together:

$$
h^T = \bigoplus_{n\in N} h^{T}_n
$$


### Multi-Attention Recurrent Network for Human Communication Comprehension (2018)

*  `Reference:` [Zadeh et al., (2018)][Zadeh et al., (2018)]
* `Difference wrt MFN: MARN models view-specific interactions using hybrid LSTM memories and cross-modal interactions using a Multi-Attention Block (MAB).`

MARN has two key components: Long-Short Term Hybrid Memory and Multi-Attention Block. The first one, LSTHM, is an extension of the Long Short-Term Memory (LSTM) by reformulating the memory component to carry to carry hybrid information (`AN: Essentially, the LSTHM takes as input the cross-view vector (dynamics) as well as the canonical input at the given time-step and the previous hidden state` ). 

LSTHM is intrinsically designed for multimodal setups and each modality is assigned to a unique LSTMH. LSTHM has a hybrid memory that stores view-specific dynamics of its assigned modality and cross-view dynamics related to its assigned modality. The component that discovers cross-view dynamics across different modalities is called the Multi-Attention Block (MAB). THe MAB first uses information from hidden states of all LSTHMs at a timestep to regress coefficients to outline the multiple existing cross-view dynamics (`AN: or simply, scoring`) among them. It then weights the output dimension based on these coefficients and learns a neural cross-view dynamics code for LSTHMs to update their hybrid memories.


<figure>
<img src="/images/marn.png" alt="MARN" class="center">
</figure>


**Long Short-Term Hybrid Memory (LSTHM)**:  The most important component of the LSTM is a memory which stores a representation of its inputs through time. The LSTHM model is designed with a memory mechanism for each modality which in addition to storing view-specific (`AN: unimodal interactions`) dynamics, it is also able to store the cross-view dynamics that are important for that modality. This allows the memory to function in a hybrid manner.

For each modality $$m \in M$$, the input to the $$m$$-th LSTHM is of the form $$\mathbf{X}^{m} = \{ x_1^m, x_2^m, \dots, x_T^m \}$$. For each modality, its assigned LSTHM outputs the hidden representation $$h^m$$. The different modalities representations are first concatenated and then fed as input to the Multi-Attention Block (MAB).

The neural cross-view dynamics code $$z_{t-1}$$, is the output of the Multi-Attention Block at the previous time-step $$t$$. Such a representation is passed to each of the individual LSTHMs and is the hybrid factor, allowing each individual LSTHM to carry cross-view dynamics that finds related to its modality.

$$
h_t \leftarrow \text{LSTM}(\cup_{m \in M}\{x_t^m\}, h_{t-1}, z_{t-1} )
$$

$$
z_t \leftarrow \text{MAB}(h_t)
$$

**Multi-Attention Block**: The MAB is a network that capture multiple different, possibly asynchronous, cross-view dynamics and encode all of them in a neural cross-view dynamics code $$z_t$$ (`AN: also known as multi-modal context vector`). In the most important step of the MAB, different dimension of the LSTHM outputs $$h_t^m$$ are assigned attention coefficients according to whether or not they form cross-view dynamics. The coefficient assignment is performed multiple $$K$$ times due to the existence of possibly multiple cross-view dynamics across the outputs of the LSTHM (`AN: it is multi-headed`).

To obtain the $$K$$ attention coefficients, $$K$$ softmax distribution are  assigned to the concatenated LSTHM memories using a deep neural network $$\mathcal{A} : \mathbb{R}^{d_{mem}} \rightarrow \mathbb{R}^{K \times d_{mem}} $$. At each time-step the output is a set of $$K$$ attentions: $$\{ a_t^k: k \leq K, a_t^k \in \mathbb{R}^{d_{mem}} \}$$ with $$a_r = \bigoplus_{k=1}^{K} a_t^k, a_t \in \mathbb{R}^{K \times d_{mem}}$$.

LSTHM output representation $$h_t$$ is then broadcasted (from $$ \mathbb{R}^{d_{mem}}$$ to $$ \mathbb{R}^{K \times d_{mem}}$$) and element-wise multiplied the $$a_t$$ to produce attended outputs $$\tilde{h}_t = \{ \tilde{h}_t^k : k \leq K, \tilde{h}_t^k \in \mathbb{R}^{d_{mem}}, \tilde{h}_t^k \in \mathbb{R}^{K \times d_{mem}} \}$$. Thus, the first dimension contains the first-cross view interactions highlighted by $$a_t^1$$, and so on and so forth.

Then, $$\tilde{h}_{t}$$ is split  into $$m$$ different parts - one for each modality - and undergoes dimensionality reduction using $$C_{m} : \mathbb{R}^{K \times d_{mem}^m} \rightarrow \mathbb{R}^{d_{local}^m}, \forall m \in M$$, with $$d_{local}^m$$ as the target low dimension of each modality split in $$\tilde{h}_{t}$$.
The set of networks $$\{ C_m : m \in M \}$$ maps the attended outputs of each modality to the same vector space. This dimensionality reduction produces a dense code $$s_{t}^{m}$$ for the $$K$$-times attended dimension of each modality. Finally, the set of all $$M$$ attended modality outputs, $$ s_t^ = \bigoplus_{m \in M}s_t^m $$, are passed into a deep neural network $$\mathcal{G}: \mathbb{R}^{\sum_{m \in M}d_{local}^m} \rightarrow \mathbb{R}^{d_{mem}}$$ to generate the neural cross-view dynamic code $$z_t$$ at time $$t$$.


<figure>
<img src="/images/MARN_alg.png" alt="MARN algorithm" class="center" width="70%">
</figure>


### Multimodal Language Analysis with Recurrent Multistage Fusion (2018)

*  `Reference:` [Liang et al., (2018)][Liang et al., (2018)]

Recurrent Multistage Fusion Network (RMFN) decomposes the fusion problem into multiple stages, each of them focused on a subset of multimodal signals for specialized, effective fusion `(AN: essentially, the multimodal input (i.e, concatenation of the output (hybird) LSTM - early stage approach) is scored via attention mechanism. Such scored representation (a.k.a., HIGHLIGHTED) is passed to an LSTM module (a.k.a., FUSE) which additionally takes as input the previous FUSE module output. This process (HIGHLIGHT and FUSE) is repeated K times (hence, multistage). Finally, each FUSEd output is concatenated and fed to a NN (a.k.a., SUMMARIZE) that outputs a multimodal context vector that is fed to the LSTHM. The overall process is repeated for the whole sequence)`.


<figure>
<img src="/images/RMFN_general.png" alt="RMFN general overview" class="center">
</figure>

Each sequence $$\mathbf{X}^m = \{x_1^m, x_2^m \dots, x_T^m\}$$ is modeled with an intra-modal recurrent neural networks (unimodal ''hybrid'' LST(H)M). At each time step $$t$$, each LSTHM will output a unimodal (`AN: how can it be UNIMODAL if in the other paper is presented as cross-modal?`) representation $$h_{t}^m$$. The Multistage Fusion Process uses a recursive approach to fuse all unimodal representation into a cross modal representation $$z_t$$ which is fed back into the each intra-modal LSTHM.

<figure>
<img src="/images/RMFN_fusion.png" alt="RMFN multi-stage fusion process" class="center">
</figure>


Three multistage fusion modules: HIGHLIGHT, FUSE and SUMMARIZE. Multistage fusion begins with the concatenation of intra-modal network outputs (LSTHM) $$h_t = \bigoplus_{m \in M}h_t^m$$.

`HIGHLIGHT`: at each stage $$k$$, a subset of the multimodal signal represented in $$h_t$$ is automatically highlighted for fusion. This modules functions as decoder LSTM. Attention weights are inferred based on the previously assigned weights.

$$
a_t^{\lbrack k \rbrack} = f_H(h_t; a_t^{\lbrack 1:k-1\rbrack}, \Theta)
$$

Highlighting is performed by element-wise multiplication of the attention weights and the multimodal signal $$h_t$$.

$$
\tilde{h}_t^{\lbrack k\rbrack} = a^{\lbrack k\rbrack} \odot h_t
$$

`FUSE`: the highlighted multimodal signal are simultaneously fused in a **local fusion** and then **integrated with fusion representations from previous stages**. This module contains a single LSTM that, over $$K$$ time steps, takes as input the output of the `HIGHLIGHT` module, as hidden state, the previous hidden state, and the cell state flows continuously through each step $$k$$.	

$$
s_t^{\lbrack k\rbrack} = f_F(\tilde{h}_t^{\lbrack k\rbrack}; s_t^{\lbrack 1:k-1\rbrack}, \Theta)
$$

This is achieved by means of another FUSE LSTM, where the input gate enables a local fusion and forget and output gates enable integration with previous fusion results.

`SUMMARIZE`: it generates a cross-modal representation using all final fusion representations $$s_t^{\lbrack 1:k-1\rbrack}$$. Formally, this operation is defined as:

$$
z_t = \mathcal{S}(s_t^{\lbrack 1:k-1\rbrack}; \Theta)
$$

where $$z_t$$ is the final output of the multistage fusion process and represents all cross-modal interactions discovered at time $$t$$. The summarized cross-modal interaction is then fed into the intra-modal recurrent network.


### MAG-BERT: Integrating Multimodal Information in Large Pretrained Transformers (2019)

*  `Reference:` [Rahman et al., (2019)][Rahman et al., (2019)]

Multimodal Adaptation Gate (MAG) is a module designed to be attached to pre-trained models such as BERT and XLNet. MAG accommodate for multimodal non-verbal finetuning. It does so by generating a shift (by means of a displacement vector) to internal distribution of the pre-trained model. Such a shift is conditioned on visual and acoustic modalities.

In absence of multimodal accompaniments, each word falls withing some part of a semantic space, depending only on the meaning of the word in a linguistic structure (i.e., a sentence). Nonverbal behavior can have an impact on the meaning of words, and therefore on the position of words in this semantic space. Together language and nonverbal accompaniments decide on the new position of the word in the semantic space. This is achieved by means of a displacement vector (i.e., a vector with a trajectory and magnitude that shifts the language-only position).

<figure>
<img src="/images/mag.png" alt="MAG" class="center" width="50%">
</figure>


MAG units receives three inputs, one is purely lexical, one is visual and one is acoustic. Let the triplet $$(Z_i, A_i, V_i)$$ denote these inputs for the $$i$$-th word in a sequence. The displacement vector is factorized into bimodal components $$\lbrack Z_i;A_i\rbrack$$ and $$\lbrack Z_i, V_i\rbrack$$ (i.e., the concatenation of lexical with acoustic and visual vectors, respectively).
Two gating vectors $$g_i^v$$ and $$g_i^a$$ are produced as follows:

$$
g_i^v = R(W_{gv} \lbrack Z_i, V_i\rbrack + b_v )
$$

$$
g_i^a = R(W_{ga} \lbrack Z_i, A_i\rbrack + b_a )
$$

where $$W_{gv}, W_{ga}$$ are weight matrices for visual and acoustic, $$b_v, b_a$$ scalar biases and $$R(\cdot)$$ is any non-linear activation function. **These gates highlight the relevant information in visual and acoustic modality conditioned on the lexical vector.**

Then a non-verbal displacement vector $$H$$ is created by fusing together $$A_i$$ and $$V_i$$. multiplied by their respective gating vectors:

$$
H_i = g_i^a \cdot (W_a A_i) + g_i^v \cdot (W_v V_i) + b_H
$$

where $$W_a, W_v$$ are weight matrices for acoustic and visual information and $$b_H$$ a scalar bias vector.
Subsequently, the multimodal vector $$\bar{Z}_i$$ is created by a weighted summation of $$Z_i$$ and its nonverbal displacement vector $$H_i$$.

$$
\bar{Z}_i = Z_i + \alpha H_i
$$

$$
\alpha = \text{min}(\frac{||Z_i||_2}{||H_i||_2}\beta, 1)
$$

where $$\beta$$ is a hyper-parameter, $$\|Z_i\|_{2}, \|H_i\|_{2}$$ denote the $$L_{2}$$ norm of the $$Z_i$$ and $$H_i$$ vectors respectively. Finally, a layer normalization and dropout layer is applied to $$\bar{Z}_i$$.




### Multimodal Transformer for Unaligned Multimodal Language Sequences (2020)
* `Reference:` [Tsai et al., (2019)][Tsai et al., (2019)]
* `Source code:` [MulT GitHub][MulT GitHub]


Multimodal Transformer (MulT) is designed to model unaligned multi-modal language sequences. At the high-level MulT merges multi-modal time-series via a feed-forward fusion process from multiple directional pairwise **crossmodal transformers**.

<figure>
<img src="/images/mult_overall.png" alt="MulT Architecture">
</figure>

Specifically, each **crossmodal transformer** serves to repeatedly **reinforce** a *target* modality **with the low-level features** from another *source* modality by learning the attention across the two modalities' features. Differently from [Vaswani et al., (2017)][Vaswani et al., (2017)], each crossmodal attention block adapts directly from the low-feature sequence and **does not rely on self-attention** (Note: otherwise it would not be cross-modal). Authors argue that performing adaptation from low-level features benefits MulT to preserve the low-level information for each modality.

A MulT architecture hence models all pairs of modalities with such crossmodal transformers, followed by sequence models (e.g., self-attention transformers) that predicts using the fused features.

The core of the architecture is the **crossmodal attention module**.

Let $$\alpha$$ and $$\beta$$ be two different (potentially non-aligned) modalities. Let $$X_{\alpha} \in \mathbb{R}^{T_{\alpha} \times d_{\alpha}}$$ and $$X_{\beta} \in \mathbb{R}^{T_{\beta} \times d_{\beta}}$$ (where $$T_{(\cdot)})$$ and $$d_{(\cdot)}$$ represents sequence length and feature dimension, respectively).

Queries, keys and values are thus $$Q_{\alpha} = X_{\alpha}W_{Q_{\alpha}}$$, $$K_{\beta} = X_{\beta}W_{K_{\beta}}$$, and $$V_{\beta} = X_{\beta}W_{V_{\beta}}$$. The alignment (a.k.a., latent adaptation) from $$\beta$$ to $$\alpha$$ is:

$$
\displaylines{
	Y_{\alpha} = CM_{\beta \rightarrow \alpha}(X_{\alpha}, X_{\beta}) \\
	= \text{softmax}(\frac{Q_\alpha K_{\beta} ^ T}{2}/\sqrt{d_{k}})V_{\beta} \\
	= \text{softmax}(\frac{X_{\alpha}W_{Q_{\alpha}} X_{\beta}^T W^T_{K_{\beta}}}{\sqrt{d_{k}}})X_{\beta}W_{V_{\beta}}

}
$$

Note that $$T_{\alpha}$$ has the same length as $$Q_{\alpha}$$ (i.e., $$T_{\alpha}$$), but it meanwhile represented in the feature space of $$V_{\beta}$$. The softmax computes a score matrix $$\in \mathbb{R}^{T_{\alpha} \times {T_{\beta}}}$$, whose $$(i,j)$$-th entry is measures the attention given to by the $$i$$-th time step of modality $$\alpha$$ to the $$j$$-th time-step of modality $$\beta$$. Hence, the $$i$$-th step of modality $$Y_{\alpha}$$ is a weighted summary of $$V_{\beta}$$, with the weight determined by $$i$$-th rows in $$\text{softmax}(\cdot)$$.

<figure>
<img src="/images/mult_attention.png" alt="MulT Attention">
</figure>

**Temporal Convolutions**: to ensure that each elements of the input sequences has sufficient awareness of its neighborhood elements, the input sequence is passed through a 1D temporal convolution layer:

$$
\hat{X}_{\{L, V, A\}} = \text{Conv1D}(X_{\{L, V, A\}}, k_{\{L, V, A\}}) \in \mathbb{R}^{T_{\{L, V, A\}} \times d}
$$
 where $$k_{\{L, V, A\}}$$ are the sizes of the convolutional kernels for each modality.

Model is trained end-to-end via The **Connectionist Temporal Classification loss** (CTCLoss) [Graves et al., (2016)][Graves et al., (2016)].


<figure>
<img src="/images/mult_block.png" alt="MulT Block" class="center">
</figure>


Multimodal features are extracted as follows:
1. Textual $$\rightarrow$$ [GloVe Word Embeddings][glove]
2. Visual $$\rightarrow$$ [Facet][facet]
3. Acoustic $$\rightarrow$$ [COVAREP][covarep]

`TLDR:` 
1. Multimodal features are encoded via pre-computed representations;
2. Each input sequence is passed through a 1D (temporal) convolutional layer to ensure they have awareness of their neighborhood
3. For each modality (ordered) pair, for each transformer layer, hidden state is processed by the crossmodal attention block where K and V are always computed from the low-level feature sequence ($$Z_{\beta}^{\lbrack 0 \rbrack})$$ in Figure below.
4. This architecture is designed to deal with human language input (it takes input as described in point 1. Thus it does not (natively) work on images-description pairs)

### VL-BERT: Pre-training of Generic Visual-Linguistic Representations (2019)

*  `Reference:` [Su et al., (2019)][Su et al., (2019)]

<figure>
<img src="/images/vl_bert.png" alt="vlbert">
</figure>

VL-BERT is a unified-stream architecture. The model takes both visual and linguistics elements as inputs (RoI for images, subword piece-wise for sentences).
Three types of inputs are involved (1) visual, (2) linguistic, and (3) special elements to disambiguate different inputs formats. For each input element, its embedding is the summation of four elements: token embeddings, visual feature embeddings, segment embeddings and sequence position embeddings.

**Token Embeddings**: the linguistic input is embedded via Word-Piece embeddings with a 30.000 vocabulary. **For the visual elements, a special \[IMG\] token is assigned to each one of them**.

**Visual Feature Embeddings**: For the visual element corresponding to a RoI, the representation is extracted via a **Fast** R-CNN detector (specifically, the hidden representation preceding the output layer, 2048 dimensional vector). The non-visual elements are, instead, assigned to vectors extracted from the whole input image (i.e., **Faster** R-CNN on a RoI covering the whole input image -the pixels in the masked ROI are set to zeros before applying Fast R-CNN). **How many ROIs can be fed as input?**

**Segment Embeddings**: Three types, A, B, and C. Simply designed to separate different input elements. A and B for textual input (Question/Answer, or Answer/Reason etc.), C for the image.

**Sequence position Embeddings**: **TODO**

Model is trained on two tasks:
1. Masked Language Modeling with Visual Clues: the task drives the network to not only model the dependencies in sentence words, but also to align the visual and linguistic content. For example in "*kitten drinking from \[MASK\]*", without the input image, the masked word could be any container.
2. Masked RoI Classification with Linguistic Clues: to avoid any visual clue leakage from the visual feature embedding of other elements, the pixels in the masked ROI are set to zeros before applying Fast R-CNN. The category label for the masked RoI is predicted by pre-trained Faster R-CNN. For a sample drawn from the BooksCorpurs & English Wikipedia datasets, the input format *degenerates* to only text. In such a scenario, the visual feature embedding is a learnable emebedding shared for all words.


### ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representation fro Vision-and-Language Tasks (2019)

*  `Reference:` [Lu et al., (2019)][Lu et al., (2019)]
*  `AN: the model allows for any variable number of ''vanilla'' unimodal transformers as well as any number of co-TRM (cross-attention) transformer blocks.`


`To learn a joint visual-linguistic representation, we look to the recent success in self-supervised learning which have captures rich semantic and structural information from large,
unlabelled data sources by training models to perform so-called proxy tasks. These proxy tasks leverage structure within the data to generate supervised task automatically (e.g., colorizing images or reconstructing words in text).`

<figure>
<img src="/images/vilbert_overall.png" alt="vilbert overall" class="center">
</figure>


ViLBERT consists of two parallel BERT-style models operating over image regions and text segments. Each stream (i.e., modality) is modeled via series of transformer blocks and novel co-attentional transformer layers (Co-TRM) that enable information exchange between modalities. Notice that exchange between the two streams is restricted to be between specific layers and that the text stream has significantly more processing before interacting with visual features.

**Co-Attentional Transformer Layers (Co-TRM)**: The module computes query, key, and value matrices as in standard transformer block. However, the keys and the values from each modality are passed as input to the other modality multi-headed attention block. Consequently, the attention block produces attention-pooled features for each modality conditioned on the other (a.k.a., cross-attention).

<figure>
<img src="/images/attention_vilbert.png" alt="vilbert" class="center">
</figure>


**Image representations**: Images lack a natural ordering, thus authors encode spatial location instead, constructing a 5-d vector from region position (normalized top-left and bottom-right coordinates), and the fraction of the image area covered. This is then projected to match the dimension of the visual features and they are summed (positional encoding).

**Training task and objectives**:
1. *Masked multi-modal alignment*: predict if caption describes the image.
2. *Multi-modal prediction*: for images, masked textual inputs is handled as in BERT. For **images**: the model is set to predict a distribution over the semantic classes for the corresponding are image. To supervise this, authors take the output distribution for the region from the same pre-trained detection model used in feature extraction. **The model is trained to minimize the KL divergence between the two distribution**.



### Multi-Modality Cross-Attention Network for Image and Sentence Matching (2020)

* `Reference:` [Wei et al., (2020)][Wei et al., (2020)]
* `TASK SPECIFIC: Image-Text Retrieval`
<figure>
<img src="/images/mmca.png" alt="MMCA architecture" class="center">
</figure>

Multi-Modality Cross-Attention Network (MMCA) mainly consists of two modules: (1) the self-attention module and the (2) cross-attention module.

**Cross-Attention Module**: it takes as input the stacked features of image regions and sentence words $$Y = \begin{bmatrix}R\\E\end{bmatrix} = \{r_1, \dots, r_k; e_1, \dots, e_n\}, Y \in \mathbb{R}^{(k+n) \times d_x}$$.
The Query, Key and Value are computed as follows:

$$
K_Y = YW^K = \begin{bmatrix} RW^K\\EW^K\end{bmatrix} = \begin{bmatrix}  K_R\\K_E\end{bmatrix}
$$

$$
Q_Y = YW^Q = \begin{bmatrix} RW^Q\\EW^Q\end{bmatrix} = \begin{bmatrix}  Q_R\\Q_E\end{bmatrix}
$$

$$
V_Y = YW^V = \begin{bmatrix} RW^V\\EW^V\end{bmatrix} = \begin{bmatrix}  V_R\\V_E\end{bmatrix}
$$


Scaled dot-product attention is deployed as:

$$
Q_Y K_Y^Y V_Y = \begin{bmatrix} Q_R\\Q_E\end{bmatrix} \cdot \lbrack K_R^T K_E^T\rbrack \cdot \begin{bmatrix}  V_R\\V_E\end{bmatrix}
$$

$$
= \begin{bmatrix} Q_R K_R^T V_R + Q_R K_E^T V_E\\Q_E K_E^T V_E + Q_E K_R^T V_R\end{bmatrix}
$$

which models both the inter-modality (self-attention) as well as the cross-modality (cross-attention) at the same time.


### Cross-Modal Self-Attention for Referring Image Segmentation (2019)

* `Reference:` [Ye et al., (2019)][Ye et al., (2019)]
* `TASK SPECIFIC: Image Segmentation`

<figure>
<img src="/images/cmsa.png" alt="cross-modal self-attention" class="center">
</figure>


Given an image and a referring expression query, a CNN is used to extract visual feature maps at different levels from the image. Each word in the referring expression is represented as a vector of word embedding. Every word embedding is then appended to the visual feature map to produce a multimodal feature map. Thus, there is a multimodal feature map for each word in the referring expression. A Cross-Modal Self-Attention mechanism is deployed to combine the feature maps of different words into a cross-modal representation (self-attentive feature map).

**Multimodal Features**: input consists of an image $$I$$ and a referring expression with $$N$$ words $$w_1, w_2, \dots, w_n$$.
The feature map extracted from a specific CNN layer is represented as $$V \in \mathbb{R}^{H \times W \times C_v}$$.
For the language description, each word $$w_n$$ is encoded as an embedding $$e_n \in \mathbb{R}^{C_l}$$. Different from other methods, authors do not generate an overall representation (i.e., via recurrent NN) but they keep the individual embeddings.
In addition, spatial coordinate feature are defined as an 8-D vector. The first 3-dimensions encode the normalized horizontal position, the subsequent 3, normalized vertical position. The last 2-dimensions encode the normalized width and height of the image.
`Finally, a joint multimodal representation, at each spatial position for each word, is constructed by concatenating the visual features, word vectors, and spatial coordinate features.` The multimodal feature $$f_{pn}$$, corresponding to the location $$p$$ and the $$n$$-th word, is defined as follows:

$$
f_{pn} = \textit{concat}(\frac{v_p}{||v_p||_{2}}, \frac{e_n}{||e_n||_2}, s_p)
$$

The feature vector $$f_{pn}$$ encodes information about the combination of a specific location $$p$$ in the iamge and the $$n$$-th world of the referring expression with a total dimension of $$(C_v + C_l + 8)$$. The overall mutlimodal feature map $$F = \{f_{pn} : \forall{p}, \forall{n}\}$$ to represent the colletion fo features $$f_{pn}$$ for different spatial locations and words. The dimension of $$F$$ is $$N \times H \times W \times (C_v + C_l + 8)$$.

**Cross-Modal Self-Attention**: The multimodal feature $$F$$ is quite large and may contain redundant information. Additionally, its size is depending on the number of words in the referring expression. Thus, it is difficult to directly exploit $$F$$ to produce the segmentation output.

The proposed CMSA module can capture long-range dependencies between words in a referring expression and different spatial location.

`(AN: I did not understand this part`) The proposed module takes $$F$$ as the input and produces a feature maps that summarizes $$F$$ after learning the correlation between the language expression and the visual context. Note that the size of this output does not depend on the number or words present in the textual description.

Given a multimodal feature vector $$f_{pn}$$, the cross-modal self-attention module first produces a set of query, key and value pair by the usual linear transformation `AT EACH SPATIAL LOCATION AND WORD` $$p$$ and $$n$$, respectively. 
Cross-modal self-attentive feature $$\hat{v}_{pn}$$ is computed as:

$$
\hat{v}_{pn} = \sum_{p'}\sum_{n'}a_{p,n,p',n'}v_{p'n'}
$$ 

$$
a_{p,n,p',n'} = \text{Softmax}(q_{p'n'}^T  \cdot k_{pn})
$$ 

where $$a_{p,n,p',n'}$$ is the attention score that `takes into account of the correlation between` $$(p,n)$$ `and any other combinations of spatial location and word` $$(p',n')$$.
Then $$\hat{v}_{pn}$$ is transformed back to the same dimension as $$f_{pn}$$ via a liner layer and is added element-wise with $$f_{pn}$$ to form a residual connection. The final feature representation is average-polled over all words in the referring expression, These operations can be summarized as:

$$
\hat{f}_{p} = \text{avg-pool}_n (W_{\hat{v}}\hat{v}_{pn} + f_{pn}) = \frac{\sum_{n=1}^{N}(W_{\hat{v}}\hat{v}_{pn} + f_{pn})}{N}
$$

where $$W_{\hat{v}}\hat{v}_{pn}$$ is the information coming from the CMSA mechanism, thus from any position-word combination, that is integrated with the original feature representation for the specific position-word pair $$(p,n)$$ which is called $$f_{pn}$$. These values are averaged for each position, across all the words $$N$$.

$$\hat{F} = \{\hat{f}_p\ \forall{p}\}$$ denotes the collection of $$\hat{f}_p$$ at all spatial locations, i.e., $$\hat{F} \in \mathbb{R}^{C_v + C_l + 8}$$.

**Gated Multi-Level Fusion**: for each feature map $$i$$, a memory gate $$m^i$$ and a reset gate $$r^i$$ are generated ($$m^i, r^i \in \mathbb{R}^{H_i \times W_i}$$). Each level $$i$$ has a contextual controller $$G_i$$ which modulates the information flow from other levels to the $$i$$-th level:

$$
G_i = (1-m_i) \odot X^i + \sum_{j: j \neq i}^J \gamma^j m^j \cdot X^j
$$

$$
F_o^i = r^i \odot \textit{tanh}(G^i) + (1-r^i) \odot X^i
$$

Then, feature maps $$F_o^i$$ are aggregated and convoluted over a $$3 \times 3$$ kernel, followed by a sigmoid to produce `the likelihood of each pixel being in the foreground` in the segmentation mask:

$$
P = \sigma (\mathbb{C}_{3 \times 3}(\sum_{i=1}^I F_o^i))
$$


### Learning Transferable Visual Models From Natural Language Supervision (2021)

* `Reference:` [Radford et al., (2021)][Radford et al., (2021)]
* `a.k.a., CLIP`


<figure>
<img src="/images/clip.png" alt="CLIP" class="center">
</figure>

Simple pre-training task of predicting which captions goes with which image is an efficient and scalable way to learn high-quality image representations.

At the core of CLIP is the idea of learning perception from supervision contained in natural language.

To predict the exact caption from an image is a (too) difficult task. Recent work in **contrastive representation learning** for images ([Zhang et al., (2020)][Zhang et al., (2020)]) can learn better representations than their equivalent predictive objective  ([Tian et al., (2019)][Tian et al., (2019)]). Other works has found that although generative models of image can learn high quality image representations, they require over an order of magnitude more compute than models with the same performance ([Chen et al. (2020)][Chen et al. (2020)]).

Thus, training a system to solve the potentially easier proxy task of predicting only which text as a whole is paired with which image and not the exact words of that text. Given a batch of $$N$$ (image, text) pairs, CLIP is trained to predict which of the $$N \times N$$ possible (image, text) pairings across a batch actually occurred. It learns a multi-modal embedding space by jointly training an image encoder and a text encoder to maximize the cosine similarity of the image and text embeddings of the $$N$$ real pairs in the batch while minimizing the cosine similarity of the embeddings of the $$N^2 - N$$ incorrect pairings. Such a symmetric cross entropy loss was adapted for contrastive (image, text) representation learning in the domain of medical image by [Zhang et al., (2020)][Zhang et al., (2020)] (differently from them, in CLIP the image transformation function $$t_v$$ is simply as a random square crop from resized images and it is the only data augmentation used during training).

Implementation details: for the image encoder they use test ResNet-50 and ResNet-D + anti-aliased rect-2 blur pooling. They replace the global average pooling layer with an attention polling mechanism. They do also test ViT (Vision Transformer [Dosovitskiy et al., (2021)][Dosovitskiy et al., (2021)]). As the text encoder, they rely on a Transformer architecture with 12 layer 512-wide with 8 attention heads, lower-cased byte pair encoding, max sequence length set at 76. As final representation,the activations of the highest layer of the transformer at the `[EOS]` token are treated as the feature representation of the text which is layer normalized and then linearly projected into the multi-modal embedding space (`AN: I do not understand whether they are using just the [EOS] token, or all of the previous tokens`).

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N}(\lambda l_i^{(v \rightarrow u)} + (1-\lambda)l_i^{(u \rightarrow v)})
$$

$$
l_i^{(v \rightarrow u)} = -\log \frac{\exp{(\langle \mathbf{v}_i, \mathbf{u}_i \rangle / \tau)}}{\sum_{k=1}^N \exp{(\langle \mathbf{v}_i, \mathbf{u}_k \rangle / \tau)}}
$$

with a batch size $$N$$ of 32.768 (image, text) pairs.

**NB:** both the encoder are trained from scratch since they (openai) have a ton of training data and compute power.


### Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks (2020)

* `Reference:` [Li et al., (2020)][Li et al., (2020)]
* `Learning method which uses object tags detected in images as anchor points to ease the learning of alignments`.
* `AN: what kind of SIGNAL SUPERVISION can we exploit to align different modalities? We could try with span from from the dependency parsing - alberto sintagmatico UD dependencies`

<figure>
<img src="/images/oscar.png" alt="oscar" class="center">
</figure>

Oscar differs from existing VLP (Vision-Language Pretraining) in the way that the input image-text pairs are represented and the pre-training objective as well.

**Input:** each input image-text pair is **Word-Tag-Image triple** $$(\mathbf{w,q,v})$$, where $$\mathbf{w}$$ is the sequence of word embeddings of the text, $$\mathbf{q}$$ is the word embedding sequence of the object tags detected from the image, and $$\mathbf{v}$$ is the set of region vectors of the image.
Oscar introduces $$\mathbf{q}$$ as anchor points to ease the learning of image-text alignment. Important objects in an image are often also *presented* in the image-paired text, using either the same words as object tags or different but semantically similar or related words.
The process can also be interpreted as learning to ground the image objects, which might be ambiguously represented in the vision space, in distinctive entities represented in the language space.

Specifically, $$\mathbf{v}$$ and $$\mathbf{q}$$ are generated by means of a Faster R-CNN. The pre-trained model is used to extra t the visual semantics of each region as $$(v', z)$$ where $$v' \in \mathbb{R}^P$$ is a P-dimensional vector (i.e., P=2048), and region position $$z$$ a R-dimensional vector (i.e., 4 or 6). These two representations are concatenated to form a position-sensitive region feature vector, which is further transformed into $$v$$ using a linear projection to ensure that it has the same vector dimension as that of word embeddings.
The same Faster R-CNN is used to detect a set of high precision object tags. $$\mathbf{q}$$ is the sequence of word embeddings of the object tags detected in a given image.

**Pre-Training Objective**: multi-task pre-training composed by a *Dictionary View* task (Masked Token Loss) and a *Modality View* task (Contrastive Loss).

*Masked Token Loss:*

$$
\mathcal{L}_{MTL} = -\mathbb{E}_{(\mathbf{v,h}) \sim \mathcal{D}} \log{p(h_i|\mathbf{h}_{\setminus i}, \mathbf{v})}
$$

where $$\mathbf{h}$$ is the discrete token sequence $$\lbrack \mathbf{w, q} \rbrack$$. At each iteration each input token in $$h$$ is randomly masked with a probability of 15%.


*Contrastive Loss:*

$$
\mathcal{L}_{C} = -\mathbb{E}_{(\mathbf{h',w}) \sim \mathcal{D}} \log{p(y|f(\mathbf{h', w}))}
$$

where $$\mathbf{h'}$$ is the grouping of $$\lbrack \mathbf{q,v} \rbrack$$ and represent the image modality, whereas $$\mathbf{w}$$ is the language modality. A set of "polluted" image representations is created by replacing $$\mathbf{q}$$ with a probability of 50% with a different tag sequence randomly sampled from the dataset. The special token `[CLS]` (which contains the fused representation of $$(\mathbf{h', w})$$) is passed through a FC layer and a binary classifier to predict whether the pair contains the original image representation ($$y=1$$) or any polluted ones ($$y=0$$).

The full pre-training objective is defined as:

$$
\mathcal{L}_{\text{pretraining}} = \mathcal{L}_{\text{MTL}} + \mathcal{L}_C
$$

* `The sequence length of discrete tokens `$$h$$` and region features `$$v$$` are 35 and 50, respectively.`

### VinVL: Revisiting Visual Representations in Vision-Language Models (2021)

* `Reference:` [Zhang et al., (2021)][Zhang et al., (2021)]
* `a.k.a., OSCAR+`
* `While most VLP research focuses on improving the cross-modal fusion model, this paper focuses on improving the object-centric visual representations (i.e., enhance the object-detector (OD model). As depicted in the image below, the X152-C4 model (right), which is trained on four public dataset merged together, is able to capture much richer semantics, visual concepts and attribute information in comparison to the X152-FPN model (left).`

<figure>
<img src="/images/vinvl.png" alt="vinvl" class="center">
</figure>

VPL typically consists of two stages: (1) an object detection (OD) model is pre-trained to encode an image and the visual objects in the image to feature vectors; (2) a cross-modal fusion model is pre-trained to blend text and visual features. The OD model (1) provides an object-centric representation of images, and has been used in many VL models as a black-box.

In this paper, authors pre-train a OD model (X152-C4) on a four dataset (VG, COCO, Objects365, and OpenImagesV5) merged together. Dataset are balanced across classes and dataset sizes. Target are merged together if the class names or aliases match, otherwise they are added as a new class if no match is found.

Furthermore, they propose a variant of the [OSCAR][Li et al., (2020)] training loss to effectively optimize for VQA and text-image matching tasks. They construct two types of negative (unmatched) triples: the polluted caption $$w'$$ and the polluted answer $$q'$$. To classify whether a caption-tags-image triples contains a polluted caption is a text-image matching task. To classify whether a question-answer-image triplet contains a polluted answer is an answer selection task for VQA. They apply a FC layer on top of the transformer as a 3-way classifier to predict wheter the triples is matched ($$c=0$$), contains a polluted $$q'$$ ($$c=1$$), or contains a polluted $$w'$$ ($$c=2$$).
They modify the second terms of the original training loss as:

$$
\mathcal{L}_{\text{pretraining}} = \mathcal{L}_{\text{MTL}} + \mathcal{L}_{\text{CL3}}
$$

$$
\mathcal{L}_{\text{CL3}} = -\mathbb{E}_{(\mathbf{w,q,v;c}) \sim \mathcal{\tilde{D}}} \log{p(c|f(\mathbf{w,q,v}))}
$$

where the dataset $$(\mathbf{w,q,v;c}) \sim \mathcal{\tilde{D}}$$ contains 50% matched triplets, 25% $$w$$-polluted triples and 25% $$q$$-polluted triples.


### VirTex: Learning Visual Representations from Textual Annotations (2020)

* `Reference:` [Desai et al., (2020)][Desai et al., (2020)]
* `Source Code:` [GitHub VirTex][GitHub Virtex]

<figure>
<img src="/images/virtex.png" alt="virtex" class="center">
</figure>

Contrary to [CLIP][Radford et al., (2021)], they focus on **generative** natural language supervision via image captioning task. Captions carry rich semantic information about images, including the presence of objects, attributes of objects, spatial arrangement and actions performed. They train image captioning models to predict captions from image.
The model has two components: (1) a **visual backbone** which extracts visual features from the image input, and (2) a **textual head** that accepts the features extracted by the visual components and predicts a caption token by token. The textual head performs bi-direction captioning (bicaptioning): it comprises of a forward and a backward model. All model components are jointly trained to maximize the log-likelihood of the correct caption.
After training they discard the textual head and transfer the visual backbone to the downstream tasks.

**Language Modeling**: They also perform experiment with MLM. However, they observe that MLM training converges more slowly than directional models. Furthermore, they note point out that MLMs have poor sample efficiency since they only predict a subset of tokens for each caption, whereas the generative directional predicts all of them.

**Visual Backbone**: is a convolutional NN, specifically ResNet-50.

**Textual Head**: two identical Transformer, in forward and backward setting. During training the model receives two inputs: image features from the visual backbone, and a caption describing the image. It is trained to predict token-by-token starting with $$c_0$$. The prediction process is causal since it depends on the previous outputs. First, they convert the tokens of the caption $$C$$ to vectors via learned token and positional embeddings, followed by element-wise sum, layer norm and dropout.

Each layer performs masked multi-head self-attention over token vectors, multi-head (cross) attention between token vectors (query) and image vectors (key-values), and applies a two-layer FC to each vector. These operations are each followed by dropout, wrapped in residual connection, and followed by layer norm. After the last transformer block, they apply a linear layer to each vector to predict un-normalized log-probability over the token vocabulary.

Forward and backward model consists of independent transformer layers. However, they share the same token embedding matrix which is also reused at the output layers of each model.

**Tokenization**: Sentence-Piece using BPE algorithm. Lowecase and strip accents. Vocabulary of 10k tokens. They restrict subword merges between letters and punctuation to prevent redundant tokens such as `dog?` and `dog!`.


### UNITER: UNiversal Image-TExt Representation Learning (2020)

* `Reference:` [Chen et al., (2020)][Chen et al., (2020)]
* `WRA (Word-Region Alignment) via Optimal Transport (OT). OT-based learning aims to optimize for distribution matching via minimizing the cost of transporting one distribution to another. @Fabrizio - any trick for distribution shift that could be applied here?` 

<figure>
<img src="/images/uniter.png" alt="uniter" class="center">
</figure>

UNITER is pre-trained through **four different tasks**:
1. **Masked Language Modeling (MLM)**: mask word token with probability 15%. Note that each time the only mask one modality while keeping the other modality intact (i.e., the MLM is conditioned on the full observation rather than applying joint random masking to both modalities).

	$$
	\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{(\mathbf{w,v}) \sim \mathcal{D}} \log{P_{\theta}(\mathbf{w_m}|\mathbf{w_{\setminus m}, v})}
	$$

2. **Image-Text Matching (IMR)**: Inputs are (sentence, image) pairs. A negative pair is created by replacing the image or the text in paired sample with a randomly-selected one from other samples. The model has to predict whether the pair is legit or polluted. 

	$$
	\mathcal{L}_{\text{ITM}} = -\mathbb{E}_{(\mathbf{w,v}) \sim \mathcal{D}}\lbrack y \log{s_{\theta}(\mathbf{w,v}) + (1-y)\log{(1-s_{\theta}(\mathbf{w,v}))}} \rbrack
	$$

3. **Masked Region Modeling (MRM)**: they **propose three variants** which all share the same objective base:
	
	$$
	\mathcal{L}_{\text{MRM}} = -\mathbb{E}_{(\mathbf{w,v}) \sim \mathcal{D}} f_{\theta}(\mathbf{v_m}|\mathbf{v_{\setminus m}, w})
	$$
	

	1. **Masked Region Feature Regression** (MRFR)
	2. **Masked Region Classification** (MRC) with hard labels from the object detector
	3. **Masked Region Classification with KL-Divergence** (MRC-kl), where they use soft-labels as supervision signal, the raw output from the detector. Then they try to minimize the KL divergence between the model and the detector signal.
4. **Word-Region Alignment**: via Optimal Transport (OT) they try to learn a transport plan $$\mathbf{T} \in \mathbb{R}^{T \times K}$$ to align $$w$$ and $$v$$.

	$$
	\mathcal{L}_{\text{WRA}} = \mathcal{D}_{\text{ot}}(\mathbf{\mu, \nu}) = \min_{\mathbf{T} \prod \in (a,b)}\sum_{i=1}^T \sum_{j=1}^K \mathbf{T}_{ij} \cdot c(\mathbf{w_i, v_j})
	$$

* **TODO**

### Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training (2019)

* `Reference:` [Li et al., (2019)][Li et al., (2019)]
* `AN: starting idea/architecture of OSCAR`

<figure>
<img src="/images/unicoder_vl.png" alt="UNICODER-VL" class="center">
</figure>

Unicoder-VL takes the visual regions of the image and textual tokens of the sentence as the input and then encode the input to the linguistic embedding and image embedding.

**Linguistic Embedding**: It follows the pre-processing of BERT. Each sentences is tokenized via WordPiece algorithm. They add the usual tokens `[CLS], [SEP]` and add a special `[IMG]` token to denote the visual input.

**Image Embedding**: a Faster R-CNN is used to extract the visual features (pooled ROI features) for each image region. THe encode the location features in a 5-D vector. Image features and position are fused together by means of a FC layer. They keep the predicted label of each detected object. These predictions will be successively used in the object label prediction task.

**Pre-Training Tasks**:
1. **Masked Language Modeling (MLM)**

	$$
	\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{(w,v) \sim \mathcal{D}} \log{P_{\theta}(w_m|w_{\setminus m}, v)}
	$$

2. **Masked Object Classification (MOC)**: Sample image regions and mask the visual features with probability of 15%. Replace the object feature vector with a zero-initialized vector $$v_m$$ 90% of the time, and keep the object feature unchanged in the remaining 10%. As target: take the object detected by the Fast R-CNN with the highest confidence score $$c(v_m^{(i)}).

	$$
	\mathcal{L}_{\text{MOC}} = -\mathbb{E}_{(w,v) \sim \mathcal{D}} \sum_{i=1}^{M}\text{CE}(c(v_m^{(i)}), g_{\theta}(v_m^{(i)}))
	$$

3. **Visual-linguistic Matching (VLM)**: Instance-level alignment (rather than token/region-level) between the whole image and the sentence via VLM. Final hidden state of `[CLS]` to predict whether the linguistic sentence is semantically matched with the visual content. During training, both negative and positive image-sentence pairs are sampled.

	$$
	\mathcal{L}_{\text{VLM}} = -\mathbb{E}_{(w,v) \sim \mathcal{D}} \lbrack y \log{s_{\theta}(w,v) + (1-y)\log{(1-s_{\theta}(w,v))\rbrack}}
	$$

Overall, the **final training loss** is the sum of the three previous losses:

$$
\mathcal{L} = (\mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{MOC}}) \cdot I[y=1] + \mathcal{L}_{\text{VLM}}
$$

where $$I[y = 1]$$ is and indicator function for the label 1 being correct for the image-caption pair.

### VisualBERT: A Simple and Performant Baseline for Vision and Language (2019)

* `Reference:` [Harold Li et al., (2019)][Harold Li et al., (2019)]
* `Concurrent works: VideoBERT, ViLBERT.`

<figure>
<img src="/images/visual_bert.png" alt="VisualBERT" class="center">
</figure>

VisualBERT consists of a stack of transformer layers that implicitly align elements of an input text and regions in an associated input image with self-attention. The model is pre-trained on two visually-grounded language model objectives: **MLM** (where image tokens are never masked!) and **Sentence-Image Prediction**.

The core idea of is to reuse the self-attention mechanism within the Transformer to implicitly align elements of the input text and regions in the input image. In addition to all the components of BERT, a set of visual embeddings, $$F$$, is introduced to model the image. Each $$f \in F$$ corresponds to a **bounding image, derived from an object detector**.

Each embedding in $$FF$$ is computed by summing three embeddings:
1. $$f_o$$, a visual feature representation of the bounding region of $$f$$, computed by a CNN;
2. $$f_s$$, a segment embedding indicating it is an image embedding as opposed to a text embedding; 
3. $$f_p$$, a position embedding.

VisualBERT is pre-trained on the COCO dataset with the following objectives:
1. Masked Language Modeling: Vectors corresponding to the image are not masked.
2. Sentence-Image Prediction: the model is provided with a text segment containing two captions. One of the caption is describing the image, while the other has a 50% chance to be another corresponding caption (`AN: in COCO dataset, each image has 5 correct but different captions`), and a 50% change to be a randomly drawn caption. The model is trained to distinguish these two situations.


### LXMERT: Learning Cross-Modality Encoder Representations from Transformers (2019)

* `Reference:` [Tan et al., (2019)][Tan et al., (2019)]
* `Concurrent works: VideoBERT, ViLBERT.`

<figure>
<img src="/images/lxmert.png" alt="LXMERT" class="center">
</figure>

LXMERT consists of three Transformer encoders:
1. Object relationship encoder;
2. Language encoder;
3. Cross-modality encoder.

In order to better learn cross-modal alignments between vision and language, the model is pre-trained with five different objectives:
1. Masked Cross-Modality Language Model;
2. Masked Object Prediction via ROI-feature regression;
3. Masked Object Prediction via detected-label classification;
4. Cross-Modality Matching;
5. Image Question Answering.


**Model Architecture**: the model takes two inputs, an image and its related sentence. Each image is represented as sequence of objects, and each sentence is represented as a sequence of words. 
1. Word-Level Sentence Embeddings: WordPiece tokenizer and positional encoding.
2. Object-Level Image Embeddings: they follow [Bottom-Up and Top-Down Attention (Anderson et al., (2018))][Anderson et al., (2018)]: instead of using the feature map of a CNN, they take the features detected by an object detector backbone. They learn a position-aware embedding via 2 fully connected layers.

The three encoders revolve around self (unimodal) and cross-attention (crossmodal):
1. Single Modality Encoders: the **Language encoder** and the **Object-relationship encoder**. Each layer contains a self-attention module and a 2 feed-forward layer. There are residual connection and layer normalization in each layer.
2. Cross-Modality Encoders: each Cross-Modality Encoder layer consists of two self-attention sub-layers, one bi-directional cross-attention sub-layer and two feed forward sub-layers. The bi-direction cross-attention contains two unidirectional cross-attention modules: one from language to vision, and one from vision to language. The query and the context vectors are the outputs of the $$(k-1)$$-th layer. The cross-attention sub-layer is followed by the self-attention modules and the outputs are produced by FF networks. Residual connection and layer normalization are applied at each layer level.


### SimVLM: Simple Visual Language Model Pretraining with Weak Supervision (2021)

* `Reference:` [Wang et al., (2021)][Wang et al., (2021)]
* `AN: SimVLM is trained on 1.000.000.000+ data! (MSCOCO: 330.000, Conceptual Caps: 3.300.000, CLIP Dataset: 400.000.000)` [Jia et al., (2021)][Jia et al., (2021)]
* `BLogpost`: [Blogpost](https://ai.googleblog.com/2021/10/simvlm-simple-visual-language-model-pre.html)

<figure>
<img src="/images/simvlm.png" alt="SimVLM" class="center">
</figure>

It follows a *minimalist approach* that takes raw images as inputs and make use of only the language modeling loss, without resorting to auxiliary models (like Fast R-CNN for image detection).

They propose a new objective function: **Prefix Language Modeling (PrefixLM)**. It enables bi-directional attention on the prefix sequence (e.g., $$x_{< T p}$$), and only conducts auto-regressive factorization on the remaining tokens ($$x_{\geq T_p}$$). During pre-training, a prefix sequence of tokens of length $$T_p$$ is truncated from input sequence and the training objective becomes:

$$
\mathcal{L}_{\text{PrefixLM}} = -\mathbb{E}_{x \sim \mathcal{D}} \lbrack \log{P_{\theta}(x_{\geq T_p}|x_{< T_p})}\rbrack = -\mathbb{E}_{x \sim \mathcal{D}} \lbrack \sum_{t=T_p}^{T} P_{\theta}(x_t|x_{\lbrack T_p, t \rbrack}, x_{< T_p}) \rbrack
$$

Images can be considered as prefix for their textual descriptions as they often appear before text in a web document. Therefore, for a given image-text pair, we prepend image feature sequence of length $$T_i$$ to the text sequence, and enforce the model to sample a prefix of length $$T_p \leq T_i$$ to calculate LM loss on text data only. PrefixLM model under the sequence-to-sequence framework not only enjoys the bidirectional contextualized representation an in MLM, but also can perform text generation similar to LM.

**Architecture**: PrefixLM enabled bidirectional attention within the prefix sequence, and thus it is applicable for both decoder-only and encoder-decoder sequence-to-sequence language models.


### Villa: Large-Scale Adversarial Training for Vision-and-Language Representation Learning (2020) 

* `Reference:` [Gan et al.,(2020)][Gan et al.,(2020)]

<figure>
<img src="/images/villa.png" alt="villa" class="center">
</figure>

Vision-and-Language Large-scale Adversarial Training (VILLA) advocate the use of adversarial training in VLP. VILLA consists of two training stages: (1) task-agnostic adversarial pre-training(APT), and (2) task-specific adversarial fine-tuning (AFT). To bring in more flexibility in generating adversarial examples, adversarial training is performed on the embedding level for multi-modalities, instead of operating on image pixel and sub-work token as in conventional practice. 

For text modality, adversarial perturbation are added to word embeddings. For image modality, perturbations are directly added to extracted image-region features (ROI). To power efficient large-scale training, "free" adversarial training strategy is deployed. It obtains the gradients of parameters with almost no extra cost when computing the gradients of inputs (`AN: HOW?`). Perturbations are performed one modality at a time. Perturbations are only applied to image and word embeddings, leaving other components (positional embeddings, segment embeddings etc.) of the multimodal features unchanged.

$$
\min_{\theta} \mathbb{E}_{(x_{\text{img}}, x_{\text{txt}}, y) \sim \mathcal{D}} \lbrack \mathcal{L}_{\text{std}}(\theta) + \mathcal{L}_{\text{at}}(\theta) + \alpha \cdot \mathcal{L}_{\text{kl}}(\theta) \rbrack
$$

where $$ \mathcal{L}_{\text{std}}(\theta) = L(f_{\theta}(x_{\text{img}}, x_{\text{txt}}), y) $$ is the cross-entropy loss on clean data, $$ \mathcal{L}_{\text{at}}(\theta)$$ is the label-preserving AT loss, and $$\mathcal{L}_{\text{kl}}(\theta)$$ is a finer-grained adversarial regularization term.

$$
\mathcal{L}_{\text{at}}(\theta) = \max_{||\delta_{\text{img}}|| \leq \epsilon} L(f_{\theta}(x_{\text{img}} + \delta_{\text{img}}, x_{\text{txt}}), y) + \max_{||\delta_{\text{txt}}|| \leq \epsilon} L(f_{\theta}(x_{\text{img}}, x_{\text{txt}} + \delta_{\text{txt}}), y) 
$$

$$
\mathcal{L}_{\text{kl}}(\theta) = \max_{||\delta_{\text{img}}|| \leq \epsilon} L_{\text{kl}}(f_{\theta}(x_{\text{img}} + \delta_{\text{img}}, x_{\text{txt}}), f_{\theta}(x_{\text{img}}, x_{\text{txt}})) + 
\max_{||\delta_{\text{txt}}|| \leq \epsilon} L_{\text{kl}}(f_{\theta}(x_{\text{img}}, x_{\text{txt}} + \delta_{\text{txt}}), f_{\theta}(x_{\text{img}}, x_{\text{txt}}))
$$

### ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph (2021)

* `Reference:` [Yu et al.,(2021)][Yu et al.,(2021)]
* `Leverage Scene Graph Parser to meaningfully/coherently mask part of the image-text input (New objectives: (1) Object prediction, (2) Attribute Prediction, and (3) Relationship Prediction. Nevertheless, they do also retrain the MLM and Masked Region Prediction losses. `

<figure>
<img src="/images/ernie_vil.png" alt="ERNIE-ViL" class="center">
</figure>


### UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning (2021)

* `Reference:` [Li et al.,(2021)][Li et al.,(2021)]
* `TODO`

### VIVO: Visual Vocabulary Pre-Training for Novel Object Captioning (2021)

* `Reference:` [Hu et al.,(2021)][Hu et al.,(2021)]
* `TODO`

### VL-T5: Unifying Vision-and-Language Tasks via Text Generation (2021)

* `Reference:` [Cho et al.,(2021)][Cho et al.,(2021)]
* `TODO`


### Seeing Out of tHe bOx: End-to-End Pre-training for Vision-Language Representation Learning (2021)

* `Reference:` 
* `TODO`

### E2E-VLP: End-to-End Vision-Language Pre-training Enhanced by Visual Learning (2021)

* `Reference:` 
* `TODO`


### An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021)

* `Reference:` [Dosovitskiy et al., (2021)][Dosovitskiy et al., (2021)]
* `TODO`


### CoAtNet: Marrying Convolution and Attention for All Data Sizes (2021)

* `Reference:` [Dai et al., (2021)][Dai et al., (2021)]
* `TODO`

### Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering (2018)

* `Reference:` [Anderson et al., (2018)][Anderson et al., (2018)]
* `TODO`

### Watch, Listen, and Describe: Globally and Locally Aligned Cross-Modal Attentions for Video Captioning (2018)

* `Reference:` [Wang et al., (2018)][Wang et al., (2018)]
* `TODO`


### Learning Visual Representations with Caption Annotations (2020)

* `Reference:` [Saryildiz et al., (2020)][Saryildiz et al., (2020)]
* `Design new pre-training tasks`

<figure>
<img src="/images/icmlm.png" alt="ICMLM" class="center">
</figure>


## References:
### Cross-Modal Architectures

1. [Multimodal Transformer for Unaligned Multimodal Language Sequences (Tsai et al., (2019))][Tsai et al., (2019)]
2. [VL-BERT: Pre-training of Generic Visual-Linguistic Representation (Su et al., (2019))][Su et al., (2019)]
3. [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representation fro Vision-and-Language Tasks (Su et al., (2019))][Su et al., (2019)]
3. [MAG-BERT: Integrating Multimodal Information in Large Pretrained Transformers (Rahman et al., (2019))][Rahman et al., (2019)]
13. [Tensor Fusion Network for Multimodal Sentiment Analysis (TFN) (Zadeh et al., (2017))][Zadeh et al., (2017)]
14. [Multi-attention Recurrent Network for Human Communication Comprehension (MARN) (Zadeh et al., (2018))][Zadeh et al., (2018)]
15. [Memory Fusion Network for Multi-view Sequential Learning (MFN) (Zadeh, Liang, et al., (2018))][Zadeh, Liang, et al., (2018)]
16. [Multimodal Language Analysis with Recurrent Multistage Fusion
 (RMFN) (Liang et al., (2018))][Liang et al., (2018)]
4. [Multi-Modality Cross Attention Network for Image and Sentence Matching (Wei et al., (2020)][Wei et al., (2020)]
5. [Cross-Modal Self-Attention Network for Referring Image Segmentation (Ye et al., (2019))][Ye et al., (2019)]
6. [Watch, Listen, and Describe: Globally and Locally Aligned Cross-Modal Attentions for Video Captioning (Wang et al., (2018))][Wang et al., (2018)]
9. [FusAtNet: Dual Attention based SpectroSpatial Multimodal Fusion Network for Hyperspectral and LiDAR Classification (Mohla et al., (2020))][Mohla et al., (2020)]
11.  [Fine-grained Visual Textual Alignment for Cross-Modal Retrieval using Transformer Encoders (Messina et al., (2020))][Messina et al., (2020)]
18. [Cross-Media Learning for Image Sentiment Analysis in the Wild (Vadicamo et al., (2017))][Vadicamo et al., (2017)]
10. [Perceiver: General Perception with Iterative Attention (Jaegle et al., (2021))][Jaegle et al., (2021)]
19. [Learning Transferable Visual Models From Natural Language Supervision (Radford et al., (2021))][Radford et al., (2021)]
19. [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks (Li et al., (2020))][Li et al., (2020)]
20. [VinVL: Revisiting Visual Representations in Vision-Language Models (Zhang et al., (2021))][Zhang et al., (2021)]
22. [VirTex: Learning Visual Representations from Textual Annotations (Desai et al., (2020))][Desai et al., (2020)]
23. [UNITER: UNiversal Image-TExt Representation Learning (Chen et al., (2020))][Chen et al., (2020)]
24. [Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training (Li et al., (2019))][Li et al., (2019)]
25. [VisualBERT: A Simple and Performant Baseline for Vision and Language (Harold Li et al., (2019))][Harold Li et al., (2019)]
26. [LXMERT: Learning Cross-Modality Encoder Representations from Transformers (Tan et al., (2019))][Tan et al., (2019)]
27. [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering (Anderson et al., (2018))][Anderson et al., (2018)]
29. [Learning Visual Representations with Caption Annotations (Saryildiz et al., (2020))][Saryildiz et al., (2020)]
30. [VideoBERT: A Joint Model for Video and Language Representation Learning (Sun et al., (2019))][Sun et al., (2019)]
31. [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision (Wang et al., (2021))][Wang et al., (2021)]

### Surveys:
1. [Multimodal Research in Vision and Language: A Review of Current and Emerging Trends (Uppal et al., (2020))][Uppal et al., (2020)]
2. [A Survey on Vision Transformer (Han et al., (2021))][Han et al., (2021)]
3. [Transformers in Vision: A Survey (Khan et al., (2021))][Khan et al., (2021)]


### Co-Learning (Transfer Learning approach to multi-modality)
1. [Foundations of Multimodal Co-learning (Zadeh et al., (2020))][Zadeh et al., (2020)]
2. [Multimodal Co-learning: Challenges, Applications with Datasets, Recent Advances and Future Directions (Rahate et al., (2021))][Rahate et al., (2021)] **Check this one once again**

### Extras:
21. [AdapterHub: A Framework for Adapting Transformers (Pfeiffer et al., (2020))][Pfeiffer et al., (2020)]
17. [xGQA: Cross-Lingual Visual Question Answering (Pfeiffer et al., (2021))][Pfeiffer et al., (2021)]
12. [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model (Wang et al., (2019))][Wang et al., (2019)]
13. [ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
(Dosovitskiy et al., (2021))][Dosovitskiy et al., (2021)]
14. [CoAtNet: Marrying Convolution and Attention for All Data Sizes
(Dai et al., (2021))][Dai et al., (2021)]

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

[Zadeh et al., (2017)]: https://arxiv.org/abs/1707.07250
[Zadeh et al., (2018)]: https://arxiv.org/abs/1802.00923
[Zadeh, Liang, et al., (2018)]: https://arxiv.org/abs/1802.00927
[Liang et al., (2018)]: https://arxiv.org/abs/1808.03920

[Pfeiffer et al., (2020)]: https://arxiv.org/abs/2007.07779
[Pfeiffer et al., (2021)]: https://arxiv.org/abs/2109.06082
[Radford et al., (2021)]: https://arxiv.org/abs/2103.00020
[Li et al., (2020)]: https://arxiv.org/abs/2004.06165
[Zhang et al., (2021)]: https://arxiv.org/abs/2101.00529
[Zhang et al., (2020)]: https://arxiv.org/abs/2010.00747
[Tian et al., (2019)]: https://arxiv.org/abs/1906.05849
[Chen et al. (2020)]: https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf
[Dosovitskiy et al., (2021)]: https://arxiv.org/abs/2010.11929
[Desai et al., (2020)]: https://arxiv.org/abs/2006.06666
[Chen et al., (2020)]: https://arxiv.org/abs/1909.11740
[Li et al., (2019)]: https://arxiv.org/abs/1908.06066
[Harold Li et al., (2019)]: https://arxiv.org/abs/1908.03557
[Tan et al., (2019)]: https://arxiv.org/abs/1908.07490
[Anderson et al., (2018)]: https://arxiv.org/abs/1707.07998
[Saryildiz et al., (2020)]: https://arxiv.org/abs/2008.01392
[Sun et al., (2019)]: https://arxiv.org/abs/1904.01766
[Wang et al., (2021)]: https://arxiv.org/abs/2108.10904
[Jia et al., (2021)]: https://arxiv.org/abs/2102.05918
[Dai et al., (2021)]: https://arxiv.org/abs/2106.04803
[Uppal et al., (2020)]: https://arxiv.org/abs/2010.09522
[Han et al., (2021)]: https://arxiv.org/abs/2012.12556
[Khan et al., (2021)]: https://arxiv.org/abs/2101.01169

[Gan et al.,(2020)]: https://arxiv.org/abs/2006.06195
[Yu et al.,(2021)]: https://arxiv.org/abs/2006.16934
[Li et al.,(2021)]: https://arxiv.org/abs/2012.15409
[Hu et al.,(2021)]: https://arxiv.org/abs/2009.13682 
[Cho et al.,(2021)]: https://arxiv.org/abs/2102.02779

[MulT GitHub]: https://github.com/yaohungt/Multimodal-Transformer
[covarep]: https://ieeexplore.ieee.org/abstract/document/6853739
[facet]: https://imotions.com/biosensor/fea-facial-expression-analysis/
[glove]: https://aclanthology.org/D14-1162.pdf
[AI Coffee Break]: https://www.youtube.com/c/AICoffeeBreak/videos
[LAION Dataset]: https://laion.ai/laion-400-open-dataset/

[Multimodal Tutorial (CVPR16)]: https://sites.google.com/site/multiml2016cvpr/
[Random Blog Post]: https://www.clarifai.com/blog/multimodal-deep-learning-approaches
[AI4MEDIA]: https://www.ai4media.eu/consortium/
[Giuseppe Amato]: https://scholar.google.com/citations?hl=it&user=dXcskhIAAAAJ&view_op=list_works&sortby=pubdate
[Nicola Messina]: https://scholar.google.com/citations?hl=it&user=g-UGCd8AAAAJ&view_op=list_works&sortby=pubdate
[Vadicamo et al., (2017)]: https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w5/Vadicamo_Cross-Media_Learning_for_ICCV_2017_paper.pdf
[GitHub Virtex]: https://github.com/kdexd/virtex

