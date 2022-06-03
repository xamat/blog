---
id: 35
title: "Transformer models: an introduction and catalog — 2022\_Edition"
date: '2022-03-26T00:00:00+00:00'
author: xamat
layout: post
guid: 'http://localhost:8080/wordpress/?p=35'
permalink: /transformer-models-an-introduction-and-catalog-2d1e9039f376/
reading_time:
    - ''
categories:
    - Uncategorized
---

### Transformer models: an introduction and catalog — 2022 Edition

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/09XUAshzdXdpL4zdt.jpg)<figcaption>Transformer model catalog (see original table here)</figcaption></figure>**Update 04/02/2022**

Thanks for all the interest and feedback in this post. As a result of all your comments, today I am updating the catalog with the following Transformer models: Megatron, Jurassic-1, MT-NLG , Wu Dao 2.0, Gopher, SeeKer, GPTInstruct, Chinchilla, Turing-NLG, VIT, and Swin Transformer. I considered adding others like Vega V1, which is on top of the GLUE benchmark leaderboard, but decided against it due to lack of public information on the models.

Again, let me know if you have comments or feedback.

### Why this post

I have a terrible memory for names. In the past few years we have seen the meteoric appearance of dozens of models of the Transformer family, all of which have funny, but not self-explanatory, names. The goal of this post is to offer a short and simple catalog and classification of the most popular Transformer models. In other words, I needed a Transformers cheat-sheet and couldn’t find a good enough one online, so I thought I’d write my own. I hope it can be useful to you too.

### What are Transformers

Transformers are a class of deep learning models that are defined by some architectural traits. They were first introduced in the now famous [Attention is All you Need](https://arxiv.org/abs/1706.03762) paper by Google researchers in 2017 (the paper has accumulated a whooping 38k citations in only 5 years) and associated [blog post](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html).

The Transformer architecture is a specific instance of the [encoder-decoder models](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/) that had become popular just over the 2–3 years prior. Up until that point however, attention was just one of the mechanisms used by these models, which were mostly based on LSTM (Long Short Term Memory) and other RNN (Recurrent Neural Networks) variations. The key insight of the Transformers paper was that, as the title implies, attention could be used as the only mechanism to derive dependencies between input and output.

It is beyond the scope of this blog to go into all the details of the Transformer architecture. For that, I will refer you to the original paper above or to the wonderful [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) post. That being said, we will briefly describe the most important aspects since we will be referring to them in the catalog below. Let’s start with the basic architectural diagram from the original paper, and describe some of the components.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/09XUAshzdXdpL4zdt.jpg)<figcaption>Transformer model catalog (see original table here)</figcaption></figure>### Encoder/Decoder architecture

A generic encoder/decoder architecture is made up of two models. The encoder takes the input and encodes it into a fixed-length vector. The decoder takes that vector and decodes it into the output sequence. The encoder and decoder are jointly trained to minimize the conditional log-likelihood. Once trained the encoder/decoder can generate an output given an input sequence or can score a pair of input/output sequences.

In the case of the original Transformer architecture, both encoder and decoder had 6 identical layers. In each of those 6 layers the Encoder has two sub layers: a multi-head attention layer, and a simple feed forward network. Each sublayer has a residual connection and a layer normalization. The output size of the Encoder is 512. The Decoder adds a third sublayer, which is another multi-head attention layer over the output of the Encoder. Besides, the other multi-head layer in the decoder is masked to prevent attention to subsequent positions.

### Attention

It is clear from the description above that the only “exotic” elements of the model architecture are the multi-headed attention, but, as described above, that is where the whole power of the model lies! So, what is attention anyway? An attention function is a mapping between a query and a set of key-value pairs to an output. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. Transformers use multi-headed attention, which is a parallel computation of a specific attention function called scaled dot-product attention. I will refer you again to the [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) post for many more details on how the attention mechanism works, but will reproduce the diagram from the original paper here so you get the main idea

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/09XUAshzdXdpL4zdt.jpg)<figcaption>Transformer model catalog (see original table here)</figcaption></figure>There are several advantages of attention layers over recurrent and convolutional networks, the two most important being their lower computational complexity and their higher connectivity, especially useful for learning long-term dependencies in sequences.

### What are Transformers used for and why are they so popular

The original transformer was designed for language translation, particularly from English to German. But, already the original paper showed that the architecture generalized well to other language tasks. This particular trend became quickly noticed by the research community. Over the next few months most of the leaderboards for any language-related ML task became completely dominated by some version of the transformer architecture (see for example the well known [SQUAD leaderboard](https://rajpurkar.github.io/SQuAD-explorer/) for question answer where all models at the top are ensembles of Transformers).

One of the key reasons Transformers were able to so quickly take over most NLP leaderboards is their ability to quickly adapt to other tasks, a.k.a. Transfer learning. Pretrained Transformer models can adapt extremely easily and quickly to tasks they have not been trained on, and that has huge advantages. As an ML practitioner, you no longer need to train a large model on a huge dataset. All you need to do is re-use the pretrained model on your task, maybe just slightly adapting it with a much smaller data set. A specific technique used to adapt pretrained models to a different task is the so-called [fine tuning](https://huggingface.co/docs/transformers/training).

It turns out that the capability of Transformers to adapt to other tasks is so great, that, while they were initially developed for language related tasks, they quickly became useful for other tasks ranging from [vision](https://arxiv.org/abs/2101.01169) or audio and [music](https://magenta.tensorflow.org/music-transformer) applications all the way to [playing chess](https://arxiv.org/abs/2008.04057) or [doing math](https://arxiv.org/abs/2110.03501)!

Of course all these applications would have not been possible if it wasn’t because of the myriad of tools that made them readily available to anyone that could write a few lines of code. Not only were Transformers quickly integrated into the main AI frameworks (namely [Pytorch](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) and [TF](https://www.tensorflow.org/text/tutorials/transformer)), but they even enabled the creation of an entire company around them. [Huggingface](https://huggingface.co/docs), a startup that has raised over $60M to this day, is almost entirely built around the idea of commercializing their open source [Transformers library](https://github.com/huggingface/transformers).

Last but not least, I would be remiss if I did not mention the impact of [GPT-3](https://en.wikipedia.org/wiki/GPT-3) on the popularization of Transformers. GPT-3 is a Transformer model introduced by OpenAI in May 2020 as a follow up to their earlier GPT and GPT-2. The company made a big splash by introducing the model in a [preprint](https://arxiv.org/abs/2005.14165) in which they claimed that the model was so powerful that they were not in a position to release it to the world. Since then, the model has not only been released, but also commercialized through a very large [partnership](https://openai.com/blog/openai-licenses-gpt-3-technology-to-microsoft/) between OpenAI and Microsoft. GPT-3 powers over [300 different applications](https://openai.com/blog/gpt-3-apps/), and is the foundation for OpenAI’s commercial strategy (which is a lot to say for a company that has received over $1B in funding).

### The Transformers catalog

So hopefully by now you understand what Transformer models are, and why they are so popular and impactful. In this section I will introduce a catalog of the most important Transformer models that have been developed to this day. I will categorize each model according to the following properties: Pretraining Architecture, Pretraining Task, Compression, Application, Year, and Number of Parameters. Let’s briefly define each of those:

### Pretraining Architecture

We described the Transformer architecture as being made up of an Encoder and a Decoder, and that is true for the original Transformer. However, since then, different advances have been made that have revealed that in some cases it is beneficial to use only the encoder, only the decoder, or both.

**Encoder Pretraining**

These models, which are also called bi-directional or auto-encoding, only use the encoder during pretraining, which is usually accomplished by masking words in the input sentence and training the model to reconstruct. At each stage during pretraining, attention layers can access all the input words. This family of models are most useful for tasks that require understanding complete sentences such as sentence classification or extractive question answering.

**Decoder Pretraining**

Decoder models, often called auto-regressive, use only the decoder during a pretraining that is usually designed so the model is forced to predict the next word. The attention layers can only access the words positioned before a given word in the sentence. They are best suited for tasks involving text generation.

**Transformer (Encoder-Decoder) Pretraining**

Encoder-decoder models, also called sequence-to-sequence, use both parts of the Transformer architecture. Attention layers of the encoder can access all the words in the input, while those of the decoder can only access the words positioned before a given word in the input. The pretraining can be done using the objectives of encoder or decoder models, but usually involves something a bit more complex. These models are best suited for tasks revolving around generating new sentences depending on a given input, such as summarization, translation, or generative question answering.

### Pretraining Task

When training a model we need to define a task for the model to learn on. Some of the typical tasks, such as predicting the next word or learning to reconstruct masked words were already mentioned above. “[Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271)” includes a pretty comprehensive taxonomy of pretraining tasks, all of which can be considered self-supervised:

1. **Language Modeling (LM):** Predict next token (in the case of unidirectional LM) or previous and next token (in the case of bidirectional LM)
2. **Masked Language Modeling (MLM):** mask out some tokens from the input sentences and then trains the model to predict the masked tokens by the rest of the tokens
3. **Permuted Language Modeling (PLM):** same as LM but on a random permutation of input sequences. A permutation is randomly sampled from all possible permutations. Then some of the tokens are chosen as the target, and the model is trained to predict these targets.
4. **Denoising Autoencoder (DAE):** take a partially corrupted input (e.g. Randomly sampling tokens from the input and replacing them with \[MASK\] elements. randomly deleting tokens from the input, or shuffling sentences in random order) and aim to recover the original undistorted input.
5. **Contrastive Learning (CTL):** A score function for text pairs is learned by assuming some observed pairs of text that are more semantically similar than randomly sampled text. It includes: ***Deep InfoMax (DIM):*** *maximize mutual information between an image representation and local regions of the image;* ***Replaced Token Detection (RTD):*** *predict whether a token is replaced given its surroundings;* ***Next Sentence Prediction (NSP):*** *train the model to distinguish whether two input sentences are continuous segments from the training corpus; and* ***Sentence Order Prediction (SOP):*** *Similar to NSP, but uses two consecutive segments as positive examples, and the same segments but with their order swapped as negative examples*

### Application

Here we will note what are the main practical applications of the Transformer model. Most of these applications will be in the language domain (e.g. question answering, sentiment analysis, or entity recognition). However, as mentioned before, some Transformer models have also found applications well beyond NLP and are also included in the catalog.

### Catalog table

**Note:** For all the models available in Huggingface, I decided to directly link to the page in the documentation since they do a fantastic job of offering a consistent format and links to everything else you might need, including the original papers. Only a few of the models (e.g. GPT3) are not included in Huggingface.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/09XUAshzdXdpL4zdt.jpg)<figcaption>Transformer model catalog (see original table here)</figcaption></figure>You can access the original table [here](https://docs.google.com/spreadsheets/d/1ltyrAB6BL29cOv2fSpNQnnq2vbX8UrHl47d7FkIf6t4/edit#gid=0) for easier browsing across the different model features. If you prefer to read the full list see below.

### Family Tree

The diagram below is just a simple view that should highlight the different families of transformers and how they relate to each other.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/09XUAshzdXdpL4zdt.jpg)<figcaption>Transformer model catalog (see original table here)</figcaption></figure>### Chronological timeline

Another interesting perspective is this chronological timeline of the main Transformer models borrowed from Huggingface [here](https://huggingface.co/course/chapter1/4).

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/09XUAshzdXdpL4zdt.jpg)<figcaption>Transformer model catalog (see original table here)</figcaption></figure>### Catalog List

Finally, here is a list view that might be easier to follow along in some cases:

[ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)

> ***Family:*** *BERT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM/NSP*

> ***Extension:*** *Compressed version of BERT using parameter sharing, which is much more efficient given the same number of parameters*

> ***Application:*** *Same as BERT*

> ***Date (of first known publication):*** *09/2019*

> ***Num. Params:*** *Base = 12M, Large = 18M, XLarge = 60M*

> ***Corpus:*** *Same as BERT*

[BART](https://huggingface.co/docs/transformers/model_doc/bart)

> ***Family:*** *BERT for encoder, GPT for Decoder*

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE*

> ***Extension:*** *It can be seen as a generalization of BERT and GPT in that it combines ideas from both in the encoder and decoder*

> ***Application:*** *Mostly text generation but also some text understanding tasks*

> ***Date (of first known publication):*** *10/2019*

> ***Num. Params:*** *10% more than BERT*

> ***Corpus:*** *Same as RoBERTa (160Gb of news, books, stories,*

> *and web text)*

[BERT](https://huggingface.co/docs/transformers/model_doc/bert)

> ***Family:*** *BERT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM/NSP*

> ***Extension:***

> ***Application:*** *General Language Understanding and Question Answering. Many other language applications followed*

> ***Date (of first known publication):*** *10/2018*

> ***Num. Params:*** *Base = 110M, Large = 340M*

> ***Corpus:*** *Toronto Book Corpus and Wikipedia (3.3B Tokens)*

[Big Bird](https://huggingface.co/docs/transformers/model_doc/big_bird)

> ***Family:***

> ***Pretraining Architecture:*** *Encoder AND Encoder/Decoder (BigBird is mostly a way to implement sparse attention that is implemented both in an Encoder-only as wells as Encoder/Decoder architecture)*

> ***Pretraining Task:*** *MLM*

> ***Extension:*** *Big Bird can extend other architectures such as BERT, Pegasus, or RoBERTa by using a sparse attention mechanism that elminates the quadratic dependency thus making it more suitable for longer sequences*

> ***Application:*** *Particularly well suited for longer sequences, not only in text but also e.g. in genomics*

> ***Date (of first known publication):*** *07/2020*

> ***Num. Params:*** *Depends on the overall architecture*

> ***Corpus:*** *Books, CC-News, Stories and Wikipedia*

[Chinchilla](https://arxiv.org/abs/2203.15556)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Same as Gopher but with optimizations to reduce model size and therefore training/inference time with equal or superior performance

> **Application:** Same as Gopher/GPT3

> **Date (of first known publication):** 03/2022

> **Num. Params:** 70B

> **Corpus:** Massive Text

[CTRL](https://huggingface.co/docs/transformers/model_doc/ctrl)

> ***Family:***

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:***

> ***Extension:*** *model can generate text conditioned on control codes that specify domain, style,*

> *topics, dates, entities, relationships between entities, plot points, and task-related behavior*

> ***Application:*** *Controllable text generation*

> ***Date (of first known publication):*** *09/2019*

> ***Num. Params:*** *1.63B*

> ***Corpus:*** *140 GB of text including: Wikipedia (En, De, Es, Fr), Project Gutenberg, 45 subreddits, OpenWebText2, Amazon Reviews, Europarl and UN data from WMT, question-answer pairs from ELI5, and the MRQA shared task3, which includes the Stanford Question Answering Dataset, NewsQA, TriviaQA, SearchQA, HotpotQA , and Natural Questions*

[DialoGPT](https://huggingface.co/docs/transformers/model_doc/dialogpt)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *GPT-2 architecture trained on dialog data*

> ***Application:*** *Text generation in dialog settings*

> ***Date (of first known publication):*** *10/2019*

> ***Num. Params:*** *1.5B*

> ***Corpus:*** *140M Reddit conversations*

[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)

> ***Family:*** *BERT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM/NSP*

> ***Extension:*** *Compressed version of BERT using distillation, which is much more efficient given the same number of parameters*

> ***Application:*** *Same as BERT*

> ***Date (of first known publication):*** *10/2019*

> ***Num. Params:*** *66M*

> ***Corpus:*** *Same as BERT*

[ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra)

> ***Family:***

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *RTD*

> ***Extension:***

> ***Application:*** *Same as BERT*

> ***Date (of first known publication):*** *03/2020*

> ***Num. Params:*** *Base = 110M, Large = 330M*

> ***Corpus:*** *Same as BERT except for Large with is same as XLNet*

[ERNIE](https://arxiv.org/abs/1905.07129)

> ***Family:*** *BERT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM*

> ***Extension:*** *Uses BERT for Encoder architecture, but stacks and aggregates two of them for text and entities. This architecture could be understood as BERT for text + knowledge graphs*

> ***Application:*** *Knowledge intensive related tasks that might benefit from knowledge graphs or entities such as entity recognition*

> ***Date (of first known publication):*** *05/2019*

> ***Num. Params:*** *114M*

> ***Corpus:*** *English Wikipedia + Wikidata for entitites (note that they initialize model to original BERT parameter values*

[Gopher](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Same as GPT-2 but use RSNorm instead of LayerNorm and relative positional encoding rather than absolute

> **Application:** Mostly Language Modeling and NLU, but also extensible like GPT

> **Date (of first known publication):** 12/2021

> **Num. Params:** 280B

> **Corpus:** Massive Text (2.35 billion documents, or about 10.5 TB of text including Massive Web, Books, Github, News, C4, and Wikipedia.

[GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:***

> ***Application:*** *Text generation, but adaptable to many other NLP tasks when fine tuned.*

> ***Date (of first known publication):*** *06/2018*

> ***Num. Params:*** *117M*

> ***Corpus:*** *Unsupervised Pretraining on BookCorpus dataset. Supervised Finetuning on several task-specific datasets including SNLI, RACE, Quora…*

[GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Minor extensions to the GPT architecture (e.g. layer normalization moved to the input of each sub-layer, or increased context size from 512 to 1024)*

> ***Application:*** *Text generation, but adaptable to many other NLP tasks when fine tuned.*

> ***Date (of first known publication):*** *02/2019*

> ***Num. Params:*** *1.5B*

> ***Corpus:*** *8 million web pages (40 GB). 10X GPT . WebText dataset is created by crawling all links at Reddit with at least 3 Karma points.*

[GPT-3](https://github.com/openai/gpt-3)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Same as GPT-2 with the only addition of alternating dense and locally banded sparse*

> *attention patterns, inspired by the Sparse Transformer*

> ***Application:*** *Initially text generation, but has over time been used for a large range of applications in areas such as code generation, but also image and audio generation*

> ***Date (of first known publication):*** *05/2020*

> ***Num. Params:*** *175 B*

> ***Corpus:*** *~ 500B tokens including CommonCrawl (410B), WebText2 (19B), Books1 (12B), Books2 (55B), and Wikipedia (3B)*

[GPTInstruct](https://openai.com/blog/instruction-following/)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** GPTInstruct starts off with a pretrained GPT3 model and adds reward modeling through reinforcement learning after a supervised finetuning

> **Application:** Knowledge-intensive dialog or language tasks

> **Date (of first known publication):** 01/2022

> **Num. Params:** Same as GPT3

> **Corpus:** Same as GPT3 for pretraining, but finetuned and optimized using labeler data and prompts

[GPT-Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Similar to GPT-2 but uses local attention in every other layer with a window size of 256 tokens*

> ***Application:*** *Text generation, but adaptable to many other NLP tasks when fine tuned*

> ***Date (of first known publication):*** *03/2021*

> ***Num. Params:*** *5B, 2.7B (XL)*

> ***Corpus:*** *Pile — 840 GB open source text dataset that combines 22 pre existing datasets*

[Jurassic-1](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Very similar to GPT-3, but far more parameters and improved training efficiency mostly because of the improved tokenizer. Also, different ratio of depth to breadth

> **Application:** Similar to GPT-3

> **Date (of first known publication):** 09/2021

> **Num. Params:** 178B (Jumbo), 7.5B (Large)

> **Corpus:** 300B tokens (same as GPT-3)

[mBART](https://huggingface.co/docs/transformers/model_doc/mbart)

> ***Family:*** *BART*

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE*

> ***Extension:***

> ***Application:*** *Translation*

> ***Date (of first known publication):*** *01/2020*

> ***Num. Params:*** *Same as BART*

> ***Corpus:*** *CC25 Corpus includes 25 monolingual corpuses in different languages. Largest corpuses are English (300 GB) and Russian (280GB)*

[Megatron](https://github.com/NVIDIA/Megatron-LM)

> **Family:** GPT/BERT/T5

> **Pretraining Architecture:** Encoder or Decorder, depending on the base model

> **Pretraining Task:** Same as base model

> **Extension:** Megatron is a family of models that extend previously known architectures (namely GPT-2 and BERT originally, but also T5 more recently) by introducing model parallelism primitives. In the case of BERT, the authors also replace the next sentence prediction head with sentence order prediction and use whole word n-gram masking.

> **Application:** Same as base model

> **Date (of first known publication):** 03/2020

> **Num. Params:** 8.3B (GPT-like), 3.9B (BERT-like)

> **Corpus:** Original paper uses an aggregate dataset consisting of Wikipedia), CC-Stories), RealNews, and OpenWebtext

[MT-NLG](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) (Megatron Touring NLG)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Uses parallelization similar to Megatron to train a LM double the size of GPT-3

> **Application:** Language generation and others (similar to GPT-3)

> **Date (of first known publication):** 10/2021

> **Num. Params:** 530B

> **Corpus:** [The Pile](https://arxiv.org/abs/2101.00027) (800GB dataset) + 2 Common Crawl snapshots

[Pegasus](https://huggingface.co/docs/transformers/model_doc/pegasus)

> ***Family:***

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE (more concretely GSG) and MLM*

> ***Extension:*** *Extends vanilla Transformer by using a different pretraining task (GSG: Gap Sentence Generation) that is better suited for summarization*

> ***Application:*** *Summarization*

> ***Date (of first known publication):*** *12/2019*

> ***Num. Params:*** *Base = 223M, Large = 568M*

> ***Corpus:*** *C4 (750GB) + HugeNews (3.8 TB)*

[RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)

> ***Family:*** *BERT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM (Dynamic)*

> ***Extension:*** *Extension of BERT with optimized training procedure and more data*

> ***Application:*** *Same as BERT*

> ***Date (of first known publication):*** *07/2019*

> ***Num. Params:*** *356M*

> ***Corpus:*** *Same as BERT + CC News + OpenWebText + Stories (~33B Tokens)*

[SeeKer](https://parl.ai/projects/seeker/)

> **Family:** GPT (but can extend any family)

> **Pretraining Architecture:** Encoder/decoder or decoder only, depending on the base model it’s extending

> **Pretraining Task:** Depends on the base model

> **Extension:** SeeKer is an extension that can be applied to any Transformer architecture by introducing “search”, “knowledge”, and “response” modules that are introduced during pretraining

> **Application:** Same as base models

> **Date (of first known publication):** 03/2022

> **Num. Params:** Depends on the base model

> **Corpus:** Same as base model

[Swin Transformer](https://github.com/microsoft/Swin-Transformer)

> **Family:** ViT

> **Pretraining Architecture:** Encoder

> **Pretraining Task:** Same as ViT

> **Extension:** Extends ViT by replacing the standard multi-head self attention (MSA) module by a module based on shifted windows (Swin) allowing ViT-like architectures to generalize to higher resolution images

> **Application:** Image (object detection, image classification..)

> **Date (of first known publication):** 03/2021

> **Num. Params:** 29M-197M

> **Corpus:** Imagenet and Imagenet-22k

[T5](https://huggingface.co/docs/transformers/model_doc/t5)

> ***Family:***

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE*

> ***Extension:*** *Same as original Transformer with some additions such as relative positional embeddings like Transformer XL*

> ***Application:*** *General language tasks including machine translation, question answering, abstractive summarization, and text classification*

> ***Date (of first known publication):*** *10/2019*

> ***Num. Params:*** *11 B (up to)*

> ***Corpus:*** *Colossal Clean Crawled Corpus (C4) — Cleaned up version of the Common Crawl dataset — 750 GB*

[Transformer XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl)

> ***Family:***

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Relative positioned embeddings enable longer-context attention when compared to vanilla Transformer model*

> ***Application:*** *General language tasks*

> ***Date (of first known publication):*** *01/2019*

> ***Num. Params:*** *151M*

> ***Corpus:*** *Different training datasets depending on experiments, but baseline is Wikitext-103*

[Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Optimized version of GPT2 with optimal hyperparameters and software/hardware platform to improve training

> **Application:** Same as GPT-2/3

> **Date (of first known publication):** 02/2020

> **Num. Params:** 17B originally, up to 530B more recently

> **Corpus:** Highest quality subset from The Pile + 2 CC snapshots (339B tokens)

[VIT](https://huggingface.co/docs/transformers/model_doc/vit)

> **Family:** BERT

> **Pretraining Architecture:** Encoder

> **Pretraining Task:** Image classification

> **Extension:** Extension of BERT architecture to train on patches of images

> **Application:** Image classification

> **Date (of first known publication):** 10/2020

> **Num. Params:** 86M(Base) to 632M (Huge)

> **Corpus:** From standard Imagenet to JFT-300M (large inhouse dataset)

[Wu Dao 2.0](https://en.wikipedia.org/wiki/Wu_Dao)

> **Family:** GLM (General Language Model)

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** Autoregressive blank infilling

> **Extension:** Similar to GPT in that it uses a Decoder/autoregressive architecture but applies a different pretraining task proposed in the GLM family of models. Besides, Wu Dao uses a “[Fast Mixture of Experts](https://github.com/laekov/fastmoe)” approach to scale training to trillions of parameters

> **Application:** Language and multimodal (particularly image)

> **Date (of first known publication):** 06/2021

> **Num. Params:** 1.75T

> **Corpus:** ?

[XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)

> ***Family:*** *RoBERTa*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM (Dynamic)*

> ***Extension:*** *An extension of RoBERTa that introduces small parameter tuning insights in the context of multilingual applications*

> ***Application:*** *Translation and other cross-lingual language tasks*

> ***Date (of first known publication):*** *10/2019*

> ***Num. Params:*** *Base = 270M, Large = 550M*

> ***Corpus:*** *Cleaned Common Crawl in 100 languages*

[XLNet](https://huggingface.co/docs/transformers/model_doc/xlnet)

> ***Family:*** *Transformer XL*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *PLM*

> ***Extension:*** *This model basically adapts Transformer XL architecture to permutation-based LM*

> ***Application:*** *General language tasks*

> ***Date (of first known publication):*** *05/2019*

> ***Num. Params:*** *Base=117M, Large=360M*

> ***Corpus:*** *Same as BERT + Giga5 (16GB text) + and aggressively filtered ClueWeb 2012-B (19GB), Common Crawl (110 GB)*

### Further reading

Most of the following references have already been mentioned in the post. However, it is worth listing them here in case you need more details:

- The Huggingface Transformers [documentation](https://huggingface.co/course/chapter1/1?fw=pt) and course is extremely good and comprehensive. I have used myself in this post, and I can’t recommend enough as a natural follow up to what you will find here.
- [A survey of transformers](https://arxiv.org/abs/2106.04554) (Lin et al. 2021) includes a 40 page long survey wit over 170 references and a full blown taxonomy.
- [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf) is also a very comprehensive survey that includes many of the pretrained models with a particular focus on NLP

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/09XUAshzdXdpL4zdt.jpg)<figcaption>Transformer model catalog (see original table here)</figcaption></figure>- [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271) (Quiu et al. 2021) is another 30+ pages long survey that focuses on pretrained models for NLP.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/09XUAshzdXdpL4zdt.jpg)<figcaption>Transformer model catalog (see original table here)</figcaption></figure>