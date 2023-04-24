---
id: 101
title: "Some of my favorite Recsys resources and links"
date: '2023-04-23T00:00:00+00:00'
author: xamat
##layout: post
permalink: /recsys-resources/
reading_time:
    - ''
    - ''
categories:
    - machine learning
    - recommender systems
image: /blog/images/101-01.png
---

![](/blog/images/02-01.jpeg)

This post is now an [ArXiV paper](https://arxiv.org/abs/2302.07730) that you can print and cite.

If you have any changes you want to propose to this catalog, feel free to file a PR [here](https://github.com/xamat/blog/blob/gh-pages/_posts/2023-01-16-transformer-models-an-introduction-and-catalog-2d1e9039f376.md) or directly in the LaTeX sources [here](https://github.com/xamat/TransformerCatalog)

**Update 01/16/2023**

Six months after my last update, it is clear that the world has been taken by storm by Transformers. Everyone is talking about [ChatGPT](https://amatriain.net/blog/chatGPT), so I thought I needed to add the models that got us there. I had already talked about GPTInstruct before, but I added [GPT3.5](#gpt35) and [ChatGPT](#chatgpt) as independent models although they don’t add too much to the former. I also added a couple of models from [Eleuther.ai](https://www.eleuther.ai/) and [Anthropic](https://www.anthropic.com/), the only two startups that seem to be even ready to challenge the OpenAI/Facebook/Google supremacy in language models. Because of what is happening with ChatGPT, I thought I should add the main competitors from the big labs: [Sparrow](#Sparrow) from Deepmind/Google, and Blenderbot3 from Facebook. Speaking of startups though, there has been a lot of talk of [Stability.ai](https://stability.ai/), so I felt I needed to add a reference to [StableDiffusion](#stablediffusion). Finally, and while not many details are known about [AlphaFold](#alphafold)'s architeccture, I thought I should add a reference to it since the problem of protein folding is very important, and Deepmind’s accomplishment in this regard is huge.

Also, there are two concepts that are becoming more and more important in the recent success of Transformers: On the one hand, [RLHF](#rlhf) (Reinforcement Learning with Human Feedback) for language models. On the other hand, [Diffusion models](#diffusion). I added a brief section on both these topics. 

Now that I am including over 50 Transformers I thought I should highlight those that for some reason I consider to be noteworthy. I hope the others don’t feel bad about it :-) I also felt that very often I was searching for Transformer model timelines and none was comprehensive enough, so I bit the bullet and added a [timeline view](#Timeline) to the catalog.

Enjoy! And, as always, human feedback is welcomed.

**Table of Contents**

- [Catalog Index](#CatalogIndex)
- [What are Transformers](#Transformers)
     - [Encoder/Decoder Architecture](#encoderdecoder)
     - [Attention](#attention)
     - [Reinforcement Learning with Human Feedback](#rlhf)
     - [Diffusion Models](#diffusion) 
- [The Transformers Catalog](#TransformersCatalog)
    - [Catalog Table](#CatalogTable)
    - [Catalog Family Tree](#FamilyTree)
    - [Catalog Timeline](#Timeline)
    - [Catalog List](#CatalogList)



### <a name="Transformers"></a>What are Transformers

Transformers are a class of deep learning models that are defined by some architectural traits. They were first introduced in the now famous [Attention is All you Need](https://arxiv.org/abs/1706.03762) paper by Google researchers in 2017 (the paper has accumulated a whooping 38k citations in only 5 years) and associated [blog post](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html).

The Transformer architecture is a specific instance of the [encoder-decoder models](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/) that had become popular just over the 2–3 years prior. Up until that point however, attention was just one of the mechanisms used by these models, which were mostly based on LSTM (Long Short Term Memory) and other RNN (Recurrent Neural Networks) variations. The key insight of the Transformers paper was that, as the title implies, attention could be used as the only mechanism to derive dependencies between input and output.

It is beyond the scope of this blog to go into all the details of the Transformer architecture. For that, I will refer you to the original paper above or to the wonderful [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) post. That being said, we will briefly describe the most important aspects since we will be referring to them in the catalog below. Let’s start with the basic architectural diagram from the original paper, and describe some of the components.

![](/blog/images/02-02.png)

