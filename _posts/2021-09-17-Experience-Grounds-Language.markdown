---
layout: post
title:  "Experience Grounds Language"
date:   2021-09-17 13:51:45 +0200
categories: Natural-Language-Processing
description: Language understanding research is held back by a failure to relate language to the physical world it describes and to the social interactions it facilitates.
---
`Language understanding research is held back by a failure to relate language to the physical world it describes and to the social interactions it facilitates.`

#### Authors: **Yonatan Bisk, Ari Holtzman, Jesse Thomason, et al.**

Meaning does not arise from the statistical distribution of words, but from their use by people to communicate. Many of the assumptions and understandings on which communication relies lie outside of text.
There is a need for language to attach to "extralinguistic events" (Ervin-Tripp, 1973) and the requirements for social context (Baldwin et al., 1966) should guide NLP research.

#### Five levels of **World Scope** (WS):
1. WS1. Corpus (*the past*)
2. WS2. Internet (*most of current NLP*)
3. WS3. Perception (*multimodal NLP*)
4. WS4. Embodiment
5. WS5. Social

#### **WS1**: Corpora and Representations
Word representations have a long history predating the recent success of deep learning methods. Outside of NLP, philosophy (Austin, 1975) and linguistics (Lakoff, 1973; Coleman and Kay, 1981) recognized that **meaning is flexible yet structured**. Early experiments on NN (Elamn, 1990, Bengio et al., 2003) suggested that vector representations could capture both syntax and semantics.
[...] The question of **where** meaning resides in "connectionist" system like DNN is an old one (Pollack, 1987; James and Miikkulainen, 1995). Are concepts distributed through edges or local to units in an ANN?

The Brown Corpus (Francis, 1964) and Penn Treebank (Marcus et al., 1993) defined context and structure in NLP for decades.  Only relatively recently (Baroni et al., 2009) has the cost of annotations decreased enough, and have large-scale web-crawls become viable, to enable the introduction of more complex text-based tasks. This transition to larger, unstructured context (WS2) induced a richer semantics than was previously believed possible under the distributional hypothesis.

#### **WS2**: The Written World
This move towards using large scale raw data has led to substantial advances in performance on existing and novel community benchmarks. Scale in data and modeling has demonstrated that a single representation can discover both rich syntax and semantics without our hep (Tenney et al., 2019).
Traditionally, transfer learning relied on **our understanding** of model classes. Unsupervised representations today capture deep associations across multiple domains, and ca be used successfully transfer knowledge into surprisingly diverse context (Brown et al., 2020).
These representation require scale in terms of both data and parameters. Mikolov et al., (2013) trained on 1.6B tokens, Pennington et al., (2014) scaled up to 840B tokens from Common Crawl. Peters et al., (2018) introduced ELMo with roughly 10<sup>8</sup> parameters. Transformer models have continued to scale by orders of magnitude between papers.

However, modeling lexical co-occurrence, no matter what the scale, is still modelling the **written** world. Models constructed this way blindly search for symbolic co-occurrences void of meaning.

How can models yields both "impressive results" and "diminishing returns"? Language modeling - the modern workhorse of neural NLP systems - is a canonical example. [...] Continuing to expand hardware, data sized, and financial compute cost by orders of magnitude will yield further gains, but the slope of the increase is quickly decreasing. [...] Other forms of supervision, such as multi-modal perception (Illharco et al., 2019), are necessary to learn the remaining aspects of meaning in context.

#### **WS3**: The World of Sights and Sounds
Language learning needs perception, because perception forms the basis for many of our semantic axioms. Learned, physical heuristics, such as the fact that a falling cat will land quietly, are generalized and abstracted into language metaphors like **nimble as a cat** (Lakoff, 1980). Perception is the foremost of source of reporting bias. `The assumption that we all see and hear the same things informs not just what we name, but what we choose to assume and leave unwritten.`
Even restricted to purely linguistics signals, sarcasm, stress and meaning can be implied through prosody. Further, tactile senses lend meaning, both physical and abstract, to concepts like **heavy** and **soft**. Visual perception is a rich signal for modeling a vastness of experiences in the world that cannot be documented by text alone (Harnard, 1990).

An ideal WS3 agent will exhibit better long-tail generalization and understanding than any language-only system could. This generalization should manifest in existing benchmarks, but would be most prominent in a test of zero-shot circumstances, such as “Will this car fit through that tunnel?,” and rarely documented behaviors as examined in script learning.

#### **WS4**: Embodiment and Action
In human development, **interactive** multi-modal sensory experience forms the basis of action-oriented categories (Thelen and Smith, 1996) as children learn how to manipulate their perception by manipulating the environment. Language grounding enables an agent to connect words to these action-oriented categories for communication (Smith and Gasser, 2005), but requires action to fully discover such connections. Embodiment --situated action taking-- is therefore a natural next broader context. An embodied agent must translate from language to action.
Planning is where people first learn abstraction and simple example of post-conditions through trial and error. The most basic scripts humans learn start with moving our own bodies and achieving simple goals as children, such as stacking blocks. In this space, we have unlimited supervision from the environment and can learn to generalize across plan and actions.
`Part of the problem is that much of the knowledge humans hold about the world is intuitive, possibly incommunicable by language, but still required to understand language.`
Current representations have very limited utility in even the most basic robotic setting (Scalise et al., 2019), making collaborative robots largely a domain of engineering rather than science.

#### **WS5**: The Social World
Interpersonal communication is the foundation use case of natural language (Dunbar, 1993). Take J.L. Austin's classic example of "BULL" being written on the side of a fence in a large field (Austin, 1975). It is a fundamentally *social* inference to realize that this was to indicate the presence of a dangerous creature, and that the word is written on the opposite side of the fence from where that creature lives.
In order to learn the effects language has on the world, an agent must **participate** in a linguistic activity, such as negotiation, collaboration, visual disambiguation, or providing emotional support. These activities require inferring mental states and social outcomes.
Active experimentation with language starkly contrast with the disembodied chat bots that are the focus of current dialogue community, which often do not learn from individual experiences and whose environments are not persistent enough do learn the effects of actions.

`Current notions of ground truth in dataset construction are based on crowd consensus bereft of social context`
