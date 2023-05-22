---
id: 35
title: "Transformer models: an introduction and catalog — 2023\_Edition"
date: '2023-01-16T00:00:00+00:00'
author: xamat
##layout: post
permalink: /transformer-models-an-introduction-and-catalog-2d1e9039f376/
reading_time:
    - ''
    - ''
categories:
    - machine learning
    - NLP
    - Transformers
image: /blog/images/02-05.png
---

![](/blog/images/02-01.jpeg)

This post is now an [ArXiV paper](https://arxiv.org/abs/2302.07730) that you can print and cite.


**Update 05/2023**
Another pretty large update after 4 months. I was invited to submit the [article] (https://arxiv.org/abs/2302.07730) to a journal, so I decided to enlist some help from some LinkedIn colleages and completely revamp it. First off, we added a whole lot of new models, including e.g. many from the Llama family. Because of this, we also included a new image that includes all the newer models we have included since February 2023. Since some of those models are fine-tuned or intstruction-tuned, we went into explaining a bit more what is the difference between a fine-tuned and a pre-trained model, and added a section on that discussion. Finally, there was a lot of editing throughout the paper that I have incorporated here too. Hope this makes it more useful!


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
     - [Foundation vs Fine-tuned models](#foundation)
     - [Reinforcement Learning with Human Feedback](#rlhf)
     - [Diffusion Models](#diffusion) 
- [The Transformers Catalog](#TransformersCatalog)
    - [Catalog Table](#CatalogTable)
    - [Catalog Family Tree](#FamilyTree)
    - [Catalog Timeline](#Timeline)
    - [Catalog List](#CatalogList)


<a name="CatalogIndex"></a>**Catalog Index**

Click on the list to access a Tranformer model directly, or keep reading below for more context and explanations. I am adding an * to those that I consider noteworthy in case you want to start with those.

- [ALBERT](#albert)
- [AlexaTM 20B](#alexa)
- [Alpaca](#alpaca)
- [AlphaFold*](#alphafold)
- [Anthropic Assistant](#anthropicassistant)
- [BART*](#BART)
- [BERT*](#BERT)
- [Big Bird](#BIGBIRD)
- [BlenderBot3*](#blenderbot3)
- [BLOOM*](#BLOOM)
- [ChatGPT*](#chatgpt)
- [Chinchilla](#CHINCHILLA)
- [CLIP](#CLIP)
- [CM3](#CM3)
- [CTRL](#CTRL)
- [Deberta](#deberta)
- [DALL-E](#DALLE)
- [DALL-E-2*](#DALLE2)
- [Decision Transformers](#DECISION)
- [DialoGPT](#DIALOGPT)
- [DistilBERT](#DISTILBERT)
- [DQ-BART](#DQBART)
- [Dolly](#dolly)
- [E5](#e5)
- [ELECTRA](#ELECTRA)
- [ERNIE](#ERNIE)
- [Flamingo](#FLAMINGO)
- [Flan-T5](#flant5)
- [Flan-PALM](#flanpalm)
- [Galactica](#galactica)
- [Gato*](#GATO)
- [GLaM](#GLAM)
- [GLIDE](#GLIDE)
- [GLM](#glm)
- [GC-ViT](#GCVIT)
- [Gopher](#GOPHER)
- [GopherCite](#gophercite)
- [GPT](#GPT)
- [GPT-2](#GPT2)
- [GPT-3](#GPT3)
- [GPT-3.5](#GPT35)
- [GPT-J](#gptj)
- [GPT-Neo](#GPTNEO)
- [GPT-NeoX 20B](#GPTNEOX)
- [GPTInstruct](#GPTINSTRUCT)
- [HTML](#html)
- [InstructOR](#instructor)
- [Imagen](#IMAGEN)
- [Jurassic-1](#JURASSIC1)
- [LAMDA*](#LAMDA)
- [LlaMA](#llama)
- [mBART](#MBART)
- [Megatron](#MEGATRON)
- [Minerva](#MINERVA)
- [MT-NLG](#MTNLG)
- [OpenAssistant LlAMA](#openassistantllama)
- [OPT](#OPT)
- [Palm](#PALM)
- [Pegasus](#PEGASUS)
- [Pythia](#pythia)
- [RoBERTa](#ROBERTA)
- [SeeKer](#SEEKER)
- [Sparrow*](#Sparrow)
- [StableDiffusion*](#stablediffusion)
- [Swin Transformer](#SWIN)
- [Switch](#SWITCH)
- [T0](#t0)
- [T5](#T5)
- [Trajectory Transformers](#TRAJECTORY)
- [Transformer XL](#TRANSFORMER)
- [Turing-NLG](#TURING)
- [UL2}(#ul2)
- [Vicuna](#vicuna)
- [ViT*](#VIT)
- [Wu Dao 2.0](#WUDAO)
- [XLM-RoBERTa](#XMLROBERTA)
- [XLNet](#XLNET)


# Why this post

I have a terrible memory for names. In the past few years we have seen the meteoric appearance of dozens of models of the Transformer family, all of which have funny, but not self-explanatory, names. The goal of this post is to offer a short and simple catalog and classification of the most popular Transformer models. In other words, I needed a Transformers cheat-sheet and couldn’t find a good enough one online, so I thought I’d write my own. I hope it can be useful to you too.

# <a name="Transformers"></a>What are Transformers


Transformers are a class of deep learning models that are defined by
some architectural traits. They were first introduced in he now famous [Attention is All you Need](https://arxiv.org/abs/1706.03762) paper by Google researchers in 2017 (the paper has accumulated a whooping 38k citations in only 5 years) and associated [blog post](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)) by
Google researchers in 2017. The paper has accumulated a whopping 38k
citations in only 5 years.

The original Transformer architecture is a specific instance of the
[encoder-decoder models](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)  that had become popular just over the 2--3
years prior. Up until that point however, attention was just one of the
mechanisms used by these models, which were mostly based on LSTM (Long
Short Term Memory)  and other RNN (Recurrent Neural Networks) 
variations. The key insight of the Transformers paper, as the title
implies, was that attention could be used as the only mechanism to
derive dependencies between input and output.

The input to the Transformer is a sequence of tokens. The output of the
encoder is a fixed-dimensional representation for each of the tokens
along with a separate embedding for the sequence as a whole. The decoder
takes the output of the encoder as input, and spits out a sequence of
tokens as its output. In natural language processing (NLP), the tokens
can be words or subwords. Subwords are used in all popular Transformer
NLP models because they enable us to address the out-of-vocabulary (OOV)
issue that is inherent in a word-based system. For simplicity, we will
use the term \"token\" to refer to the items in the input and output
sequences, understanding that these tokens are subwords for NLP systems.
When Transformers are used for processing images or video, the tokens
can represent sub-images or objects.

Since the publication of the paper, popular models like BERT and GPT
have used only the encoder or decoder aspects of the original
architecture. The core commonality of these models is, thus, not the
encoder-decoder aspect, but, rather, the architecture of the individual
layers in the encoders and decoders. The layer architecture of
Transformers is based on a self-attention mechanism and a feed-forward
layer, the core aspect of this being that each input token flows through
the layers in its own path, while, at the same time, being directly
dependent on every other token in the input sequence. This enables
parallel and direct computation of contextual token representations
which was previously not possible with sequential models like RNNs.

It is beyond the scope of this paper to go into all the details of the
Transformer architecture. For that, we will refer you to the original
paper or to  [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) post. That being said, we
will briefly describe the most important aspects since we will be
referring to them in the catalog below. Let's start with the basic
architectural diagram from the original paper, and describe some of the
components.

<a name="fig:transformer"></a>![](/blog/images/02-02.png)

## <a name="encoderdecoder"></a> Encoder/Decoder architecture

A generic encoder/decoder architecture (see
Figure [2](#fig:transformer) is composed of two models. The encoder
takes the input and encodes it into a fixed-length vector. The decoder
takes that vector and decodes it into the output sequence. The encoder
and decoder are jointly trained to maximize the conditional
log-likelihood of the output given the input. Once trained, the
encoder/decoder can generate an output given an input sequence or can
score a pair of input/output sequences.

In the case of the original Transformer architecture, both encoder and
decoder had 6 identical layers. In each of those 6 layers the Encoder
had two sub layers: a multi-head self attention layer, and a simple feed
forward network. The self attention layer computes the output
representation of each of its input tokens based on *all the input
tokens*. Each sublayer also has a residual connection and a layer
normalization. The output representation size of the Encoder was 512.
The multi-head self-attention layer in the decoder is slightly different
than that in the encoder. It masks all tokens to the right of the token
whose representation is being computed so as to ensure that the decoder
can only attend to tokens that come before the token it is trying to
predict. This is shown in
Figure  [2](#fig:transformer) as \"masked multi-head attention.\" The
Decoder also added a third sublayer, which is another multi-head
attention layer over all the outputs of the Encoder. Note that all those
specific details have since been modified in the many Transformer
variations we will discuss. For example, as we noted before, models like
BERT and GPT are based on only the encoder or decoder.

## <a name="attention"></a>Attention

It is clear from the description above that the only "exotic" elements
of the model architecture are the multi-head attention layers, but, as
described above, that is where the whole power of the model lies! So,
what is attention anyway? An attention function is a mapping between a
query and a set of key-value pairs to an output. Each token in the input
to the attention layer is converted to a query, key and value using
three corresponding matrices. The output representation of each token is
computed as a weighted sum of the values of all the tokens, where the
weight assigned to each value is computed by a compatibility function of
its associated key and the query of the token whose representation is
being computed. The compatibility function used in Transformers is just
a scaled dot product. A key aspect of this attention mechanism in
Transformers is that each token flows through its own computation path,
thus lending itself to parallel computation of the representation of all
the tokens in the input sequence. Now that we understand how attention
works, what is multi-head attention? Well, that is just multiple
attention blocks independently computing representations for each token.
All these representations are then aggregated to give the final
representation of the token. We will refer you again to the [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 
for many more details on how the
attention mechanism works, but will reproduce the diagram from the
original paper in Figure [3](#fig:attention) so you get the main idea.

<a name="fig:attention"></a>![The Attention Mechanism from. (left) Scaled Dot-Product Attention,
(right) Multi-Head Attention](/blog/images/02-03.png)

There are several advantages of attention layers over recurrent and
convolutional networks, the two most important being their lower
computational complexity and their higher connectivity, especially
useful for learning long-term dependencies in sequences.


## Foundation vs Fine-tuned models

A foundation model is defined as \"any model that is trained on broad
data (generally using self-supervision at scale) that can be adapted
(e.g., fine-tuned) to a wide range of downstream tasks" (see paper [here](https://arxiv.org/abs/2108.07258)) . When the
foundation model is further trained on a small amount of target-specific
data, it is called a [fine-tuned model](https://huggingface.co/docs/transformers/training) because it has been fine-tuned
to the specifics of the task at hand.

The [BERT paper](https://export.arxiv.org/abs/1810.04805) popularized this approach of pretraining and finetuning
for natural language processing, resulting in many researchers using
this approach for many different tasks. As a consequence, most of the
leaderboards for any language-related machine leartning (ML) task became
completely dominated by some version of the Transformer architecture
(see for example the well known [SQUAD leaderboard](https://rajpurkar.github.io/SQuAD-explorer) for question
answering or the [GLUE leaderboard](https://gluebenchmark.com/leaderboard) for general language
understanding, where all systems at the top employ Transformer-based
models).

In its original usage, \"fine-tuning\" referred to tweaking a foundation
model for a specific task, such as spam classification or question
answering. Models, such as BERT, produce representations of the input
tokens, but do not, by themselves, accomplish any task. Thus, it is
necessary to fine-tune them by adding extra neural layers on top of the
foundation model and training the model end to end.

With generative models like GPT, things are a little different. GPT is a
decoder language model trained to predict the next token of a sentence
given all the previous tokens. By training on huge amounts of web
corpora covering almost any topic one can think about, it was found that
GPT could actually produce reasonable outputs to input queries or
prompts. GPT accomplished this by simply predicting the next token given
the input prompt sequence and the output sequence GPT had already
predicted. This language generation actually did a somewhat reasonable
job of tasks like answering questions about general web knowledge,
writing poems etc. Notwithstanding, GPT's outputs were often untruthful
or really not very helpful to the user. To address this, OpenAI
researchers came up with the idea of training GPT to [follow human
instructions](https://arxiv.org/abs/2203.02155) . The resulting models are called InstructGPT. The authors
did this by using a small amount of human-labeled data from a large
variety of tasks to further train GPT. As before, this is a
\"fine-tuning\" process, but the resulting Instruct GPT model is capable
of doing a wide range of tasks, and is, in fact, the class of models
used by the popular ChatGPT engine. Since these models can accomplish a
myriad of tasks, we refer to them as foundation models.

Such additional fine-tuning has been used to generate other general
purpose model variants as well, specifically designed for uses cases
beyond language modeling (predicting the next token in a sequence). For
example, there is a subclass of models fined-tuned to learn text string
embeddings optimized for semantic-relatedness, making them directly
useful for higher-level semantic tasks (e.g. text classification,
clustering, search retrieval, etc.). Examples include [OpenAI's text
embedding models](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings}, [E5](https://huggingface.co/intfloat/e5-large), 
and [InstructOR](https://huggingface.co/hkunlp/instructor-xl). Transformer encoders
have also been successfully fined-tuned within multi-task learning
frameworks to be able to perform [multiple different semantic tasks using
a single shared Transformer model](https://arxiv.org/abs/2101.11038).

Thus, as we see, while originally foundation models were fine-tuned for
very specific target tasks for specific groups of users, today
fine-tuning is used to also create further versions of foundation models
that can be used by a huge number of users. The process used by ChatGPT
and similar dialog agents, like BlenderBot3 or Sparrow, is fairly
simple: Given a pretrained language model like GPT, we use it to
generate different responses to input prompts (or instructions) and have
humans rank the results. We then use those rankings (aka preferences or
feedback) to train a reward model. The reward model attaches a score to
each output for a given input instruction. After this, a reinforcement
learning with human feedback [(RLHF) process](https://arxiv.org/abs/1706.03741) is used to train the model
on more input instructions, but, rather than use a human to generate the
feedback, the reward model is used to rank the outputs of the model. You
can read much more in these two wonderful posts by [Huggingface](https://huggingface.co/blog/rlhf) and
[Ayush Thakur](https://wandb.ai/ayush-thakur/RLHF/reports/Understanding-Reinforcement-Learning-from-Human-Feedback-RLHF-Part-1--VmlldzoyODk5MTIx).

<a name="fig:rlhf"></a>![Reinforcement Learning with Human Feedback. From HuggingFace's RLHF
blog post at <https://huggingface.co/blog/rlhf>](rlhf.png)


## The impact of Transformers

The application demonstrated in the original Transformer paper  was
language translation. This seminal work also showed the architecture
generalized well to other language tasks. Over the next several months,
researchers figured out that Transformers could be used to capture a lot
of inherent knowledge about language by pretraining them on a very large
amount of unsupervised text. The knowledge captured in these models
could then be transferred to target tasks by training on a small amount
of labeled data.

While original Transformers were designed for language tasks, the same
Transformer architecture has been applied to many other applications
like the generation of images, audio, music, or even actions. Because of
that, Transformers are considered a key, if not the key, component to
the new wave of the so-called \"Generative AI\". Generative AI and its
many applications are already revolutionizing many aspects of society (see [here](https://www.nature.com/articles/d41586-023-00340-6) 
and [here](https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=4337484), for example).

Of course all these applications would not have been possible but for
the myriad of tools that made them readily available to anyone that
could write a few lines of code. Not only were Transformers quickly
integrated into the main AI frameworks (namely [Pytorch](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) and
[TensorFlow](https://www.tensorflow.org/text/tutorials/transformer)), but they even enabled the creation of an entire
company around them. [Huggingface](https://huggingface.co/docs), a startup that has raised over \$
60M to this day, is almost entirely built around the idea of
commercializing their open source [Transformers library](https://github.com/huggingface/transformers).

Transformer model adoption is further accelerated as specialized
hardware is developed by commercial players to improve model training
and inference speed. NVIDIA's [Hopper Tensor Cores](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet) can apply mixed
FP8 and FP16 precisions to dramatically accelerate AI calculations for
Transformers.

Last but not least, we would be remiss if we did not mention the impact
of ChatGPT on the popularization of Transformers. ChatGPT was released
by OpenAI in November 2022, and became the fastest growing app in
history, reaching 1 million users in less than a month, and 100 million
in [less than two](https://www.ubs.com/us/en/wealth-management/insights/article.1585717.html). ChatGPT was originally a chatbot application built on
top of the [Instruct-GPT model](https://arxiv.org/abs/2203.02155) also called GPT-3.5. Not much later,
OpenAI announced the release of the more powerful [GPT-4](https://openai.com/research/gpt-4), which
[achieves human level capabilities in tasks such as passing the USMLE exam for
medical doctors or the bar exam for lawyers](https://arxiv.org/abs/2303.08774) .


## <a name="diffusion"></a>A Note on Diffusion Models

Diffusion models have become the new state-of-the-art in image
generation, clearly pushing aside the previous approaches such as GANs
(Generative Adversarial Networks). It is important to note, though, that
the diffusion mechanism is not dependent on the Transformer
architecture. However, most modern diffusion approaches do include a
Transformer backbone .

Diffusion models are a class of latent variable models trained through
variational inference. What this means in practice is that we train a
deep neural network to denoise images blurred with some sort of noise
function. Networks that are trained this way are in fact learning the
latent space of what those images represent (see
Figure [4](#fig:diffusion).

<a name="fig:diffusion"></a>![Probabilistic diffusion model architecture from "Diffusion Models: A
Comprehensive Survey of Methods and Applications,\"](/blog/images/diffusion.png)

Diffusion models have relation to other generative models like Denoising
Autoencoders and the famous Generative Adversarial Networks [Generative Adversarial Networks (GAN)](https://en.wikipedia.org/wiki/Generative_adversarial_network),
which they have mostly replaced in many applications. Some [authors](https://benanne.github.io/2022/01/31/diffusion.html)
will go as far as saying that Diffusion models are just a specific
instance of autoencoders. However, they also admit that the small
differences do transform their application, from the latent
representation of autoencoders to the pure generative nature of
Diffusion models.

# <a name="TransformersCatalog"></a>The Transformers catalog

## Features of a Transformer

So hopefully by now you understand what Transformer models are, and why they are so popular and impactful. In this section we will introduce a catalog of the most important
Transformer models that have been developed to this day. We will
categorize each model according to the following properties: Family,
Pretraining Architecture, Pretraining or Fine-tuning Task, Extension,
Application, Date (of first known publication), Number of Parameters,
Corpus, License, and Lab. Some are relative simple to understand:
*Family* represents what original foundation model the specific model is
extending, *extension* describes what the model is adding to the one it
is deriving from, *Date* is when the model was firts published, *Number
of parameters* of the pretrained model, *Corpus* is what data sources
the model was pre-trained or fine-tuned on, *License* describes how the
model can be legally used, and *Lab* lists the institution that
published the model. The remaining propterties deserve a bit more
explanation. We do that in the paregraphs that follow:

### Pretraining Architecture

We described the Transformer architecture as being made up of an Encoder
and a Decoder, and that is true for the original Transformer. However,
since then, different advances have been made that have revealed that in
some cases it is beneficial to use only the encoder, only the decoder,
or both.

**Encoder Pretraining**

These models, which are also called bi-directional or auto-encoding,
only use the encoder during pretraining, which is usually accomplished
by masking tokens in the input sentence and training the model to
reconstruct those tokens. At each stage during pretraining,
self-attention layers can access all their input tokens. This family of
models are most useful for tasks that require understanding complete
sentences or passages, such as text classification, entailment, and
extractive question answering.

**Decoder Pretraining**

Decoder models use only the decoder during a pretraining. They are also
called auto-regressive language models because they are trained to
predict the next token based on the previous sequence of tokens.

The self-attention layers can only access the tokens positioned before a
given token in the sentence. They are best suited for tasks involving
text generation.

**Transformer (Encoder-Decoder) Pretraining**

Encoder-decoder models, also called sequence-to-sequence, use both parts
of the Transformer architecture.

Self-attention layers of the encoder can access all their input tokens,
while the self-attention layers of the decoder can only access the
tokens positioned before a given token. As explained before, the
additional attention layer in the decoder enables access to all encoder
token representations.

An encoder-decoder model can be pre-trained by optimizing denoising
objectives  or a combination of denoising and causal language modeling
objectives . These objective functions are complex in comparison to the
ones used to pretrain encoder only or decoder only models. The
encoder-decoder models are best suited for tasks revolving around
generating new sentences depending on a given input, such as
summarization, translation, or generative question answering.

### Pretraining Task

When training a model we need to define a task for the model to learn on. Some of the typical tasks, such as predicting the next word or learning to reconstruct masked words were already mentioned above. “[Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271)” includes a pretty comprehensive taxonomy of pretraining tasks, all of which can be considered self-supervised:

When training a model we need to define an objective, or task, for the
model to learn on. Some of the typical tasks, such as predicting the
next token or learning to reconstruct masked tokens were already
mentioned above. "Pre-trained Models for Natural Language Processing: A
Survey"  includes a pretty comprehensive taxonomy of pretraining tasks,
all of which can be considered self-supervised:

1.  **Language Modeling (LM):** Predict the next token (in the case of
    unidirectional LM) or the previous and next token (in the case of
    bidirectional LM).
2.  **Causal Language Modeling (Causality-masked LM):** Autoregressively
    (left-to-right, in general) predict a text sequence, similar to
    unidirectional LM.
3.  **Prefix Language Modeling (Prefix LM):** In this task, a separate
    'prefix' section is separated from the main sequence. Within the
    prefix, any token can attend to any other token (non-causal).
    Outside of the prefix, decoding proceeds autoregressively.
4.  **Masked Language Modeling (MLM):** Mask out some tokens from the
    input sentences and then train the model to predict the masked
    tokens using the surrounding context.
5.  **Permuted Language Modeling (PLM):** Same as LM, but on a random
    permutation of input sequences. A permutation is randomly sampled
    from all possible permutations. Then some of the tokens are chosen
    as the target, and the model is trained to predict these targets.
6.  **Denoising Autoencoder (DAE):** Take a partially corrupted input
    and aim to recover the original, undistorted input. Examples of
    corrupted input include randomly sampling tokens from the input and
    replacing them with \"\[MASK\]\" elements, randomly deleting tokens
    from the input, or shuffling sentences in random order.
7.  **Replaced Token Detection (RTD):** Using a \"generator\" model,
    randomly replace certain tokens in the text. The \"discriminator\"
    is tasked to predict whether a token comes from the original text,
    or the generator model.
8.  **Next Sentence Prediction (NSP):** Train the model to distinguish
    whether two input sentences are continuous segments from the
    training corpus.

Note that in the case of fine-tuned models, this property is used to
describe the task the model was fine-tuned to, not how it was
pre-trained.

### Application

Here we will note what are the main practical applications of the Transformer model. Most of these applications will be in the language domain (e.g. question answering, sentiment analysis, or entity recognition). However, as mentioned before, some Transformer models have also found applications well beyond NLP and are also included in the catalog.

### <a name="CatalogTable"></a>Catalog table

**Note:** For all the models available in Huggingface, I decided to directly link to the page in the documentation since they do a fantastic job of offering a consistent format and links to everything else you might need, including the original papers. Only a few of the models (e.g. GPT3) are not included in Huggingface.

![](/blog/images/02-04.png)

You can access the original table [here](https://docs.google.com/spreadsheets/d/1ltyrAB6BL29cOv2fSpNQnnq2vbX8UrHl47d7FkIf6t4/edit#gid=0) for easier browsing across the different model features. If you prefer to read the full list see below.

### <a name="FamilyTree"></a>Family Tree

The diagram below is a simple view that should highlight the different families of transformers and how they relate to each other.

<a name="fig:familytree"></a>![](/blog/images/02-05.png)

### <a name="Timeline"></a>Chronological timeline

Another interesting perspective of the catalog is to see it as a chronological timeline. Here you will find all the transformers in the catalog sorted by their date of publication. In this first visualization, the Y-axis is only used to cluster transformers of related heritage/family.

<a name="fig:timeline1"></a>![](/blog/images/02-06.png)

In this next visualization, the Y-axis represents model size in millions of parameters. You won't be able to see all the models in the catalog since many fall right on the same time and size, so please refer to the previous image for that.

<a name="fig:timeline2"></a>![](/blog/images/02-09.png)

Since the introduction of chatGPT, the LLM open-source community has
experienced a significant surge in activity. With each passing week, we
have observed a proliferation of refined models fine-tuned using the
latest technologies. As a result, these models are continuously
improving, growing more robust and powerful.
Figure [10](#fig:finetunedModels) demonstrates the recent emerged models
since Feb, 2023.

<a name="fig:timeline3"></a>![Recently published LLMs](02-10.png){#fig:finetunedModels}

### <a name="Catalog List"></a>Catalog List

Finally, here is a list view that might be easier to follow along in some cases:

<a name="albert"></a>[ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)

> ***Family:*** *BERT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM/NSP*

> ***Extension:*** *Compressed version of BERT using parameter sharing, which is much more efficient given the same number of parameters*

> ***Application:*** *Same as BERT*

> ***Date (of first known publication):*** *09/2019*

> ***Num. Params:*** *Base = 12M, Large = 18M, XLarge = 60M*

> ***Corpus:*** *Same as BERT*

> ***Lab:*** *Google*

<a name="alphafold"></a>[AlphaFold](https://www.deepmind.com/publications/highly-accurate-protein-structure-prediction-with-alphafold)

> ***Family:*** *[SE(3)-Transformer](https://arxiv.org/abs/2006.10503)*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *Protein folding prediction*

> ***Extension:*** *The original Alphafold used a BERT-style transformer. The details of Alphafold’s Transformer are not known, but it is believed it is an extension of the SE(3)-Tranformer, a 3-D equivariant Transformer (see [this blog post](https://fabianfuchsml.github.io/alphafold2/)).*

> ***Application:*** *Protein folding*

> ***Date (of first known publication):*** *07/2021*

> ***Num. Params:*** *21M*

> ***Corpus:*** *4170,000 proteins from a public repository of protein sequences and structures*

> ***Lab:*** *Deepmind*

<a name="anthropicassistant"></a>[Anthropic Assistant](https://arxiv.org/abs/2112.00861) (see [also](https://arxiv.org/abs/2204.05862) )

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *These models do not introduce novelties at the architecture/pretraining level and they are based on GPT-3 but rather focuses on how to improve alignment through fine-tuning and prompting. Note that the Anthropic Assistant includes several models optimized for different tasks. Latest versions of this work focus on the benefits of RLHF.*

> ***Application:*** *Different models with different applications from general dialog to code assistant.*

> ***Date (of first known publication):*** *12/2021*

> ***Num. Params:*** *10M to 52B*

> ***Corpus:*** *400B tokens from filtered Common Crawl and Books. They also create several Dialogue Preference datasets for the RLHF training.*

> ***Lab:*** *Anthropic*

<a name="BART"></a>[BART](https://huggingface.co/docs/transformers/model_doc/bart)

> ***Family:*** *BERT for encoder, GPT for Decoder*

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE*

> ***Extension:*** *It can be seen as a generalization of BERT and GPT in that it combines ideas from both in the encoder and decoder*

> ***Application:*** *Mostly text generation but also some text understanding tasks*

> ***Date (of first known publication):*** *10/2019*

> ***Num. Params:*** *10% more than BERT*

> ***Corpus:*** *Same as RoBERTa (160Gb of news, books, stories,*

> *and web text)*

> ***Lab:*** *Facebook*
 

<a name="BERT"></a>[BERT](https://huggingface.co/docs/transformers/model_doc/bert)

> ***Family:*** *BERT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM/NSP*

> ***Extension:***

> ***Application:*** *General Language Understanding and Question Answering. Many other language applications followed*

> ***Date (of first known publication):*** *10/2018*

> ***Num. Params:*** *Base = 110M, Large = 340M*

> ***Corpus:*** *Toronto Book Corpus and Wikipedia (3.3B Tokens)*

> ***Lab:*** *Google*


<a name="BIGBIRD"></a>[Big Bird](https://huggingface.co/docs/transformers/model_doc/big_bird)

> ***Family:***

> ***Pretraining Architecture:*** *Encoder AND Encoder/Decoder (BigBird is mostly a way to implement sparse attention that is implemented both in an Encoder-only as wells as Encoder/Decoder architecture)*

> ***Pretraining Task:*** *MLM*

> ***Extension:*** *Big Bird can extend other architectures such as BERT, Pegasus, or RoBERTa by using a sparse attention mechanism that elminates the quadratic dependency thus making it more suitable for longer sequences*

> ***Application:*** *Particularly well suited for longer sequences, not only in text but also e.g. in genomics*

> ***Date (of first known publication):*** *07/2020*

> ***Num. Params:*** *Depends on the overall architecture*

> ***Corpus:*** *Books, CC-News, Stories and Wikipedia*

> ***Lab:*** *Google*

<a name="blenderbot3"></a>[BlenderBot3](https://arxiv.org/abs/2208.03188)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** * BlenderBot 3 is based on a pre-trained OPT. It adds features needed for a dialog agent such as long-term memory or the ability to search the internet. It is also fine-tuned for some specific tasks given human feedback on them.*

> ***Application:*** *Same as GPT-3*

> ***Date (of first known publication):*** *08/2022*

> ***Num. Params:*** *175B*

> ***Corpus:*** *180B tokens = RoBERTa + the Pile + PushShift.io Reddit*

> ***Lab:*** *Facebook*

<a name="BLOOM"></a>[BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** * Main difference to GPT-3 is that it uses full attention instead of sparse attention*

> ***Application:*** *Same as GPT-3*

> ***Date (of first known publication):*** *07/2022*

> ***Num. Params:*** *176B*

> ***Corpus:*** *366B tokens (1.5 TB of text data) multilingual dataset*

> ***Lab:*** *Big Science/Huggingface*

<a name="chatgpt"></a>[ChatGPT](https://openai.com/blog/chatgpt/)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** ChatGPT takes a GPT3.5 (aka GPT3 Davinci-003) pretrained model and uses RLHF to finetune the model mostly like described in InstructGPT but with slight differences in the data collection. ChatGPT is also more than a model since it includes extensions for Memory Store and retrieval similar to BlenderBot3

> **Application:** Dialog agents

> **Date (of first known publication):** 10/2022

> **Num. Params:** Same as GPT3

> **Corpus:** Same as GPT3 + datasets generated for RLHF

> ***Lab:*** *OpenAI*

<a name="CHINCHILLA"></a>[Chinchilla](https://arxiv.org/abs/2203.15556)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Same as Gopher but with optimizations to reduce model size and therefore training/inference time with equal or superior performance

> **Application:** Same as Gopher/GPT3

> **Date (of first known publication):** 03/2022

> **Num. Params:** 70B

> **Corpus:** Massive Text

> ***Lab:*** *Deepmind*


<a name="CLIP"></a>[CLIP](https://huggingface.co/docs/transformers/model_doc/clip)

> ***Family:*** *CLIP (Also using Resnet, ViT, and vanilla transformer for text)*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *predict which of the N × N possible (image, text) pairings
across a batch actually occurred*

> ***Extension:*** * Combines Resnet and ViT for the visual encoding with Transformer for the Textual encoder*

> ***Application:*** *Image/object classification*

> ***Date (of first known publication):*** *02/2021*

> ***Num. Params:*** *?*

> ***Corpus:*** *WIT (WebImageText) - 400 million text,image pairs*

> ***Lab:*** *OpenAI*


<a name="CM3"></a>[CM3](https://arxiv.org/abs/2201.07520)

> ***Family:*** *HTML*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *Causality-masked LM*

> ***Extension:*** *This is somewhat similar to HTML in its use of structured training data. However, it is a different architecture and uses causal masking*

> ***Application:*** *Multimodal language model with the ability to do structured prompting*

> ***Date (of first known publication):*** *01/2022*

> ***Num. Params:*** *13B (largest)*

> ***Corpus:*** *CC-News, English Wikipedia*

> ***Lab:*** *Facebook*
 

<a name="CTRL"></a>[CTRL](https://huggingface.co/docs/transformers/model_doc/ctrl)

> ***Family:***

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:***

> ***Extension:*** *model can generate text conditioned on control codes that specify domain, style,*

> *topics, dates, entities, relationships between entities, plot points, and task-related behavior*

> ***Application:*** *Controllable text generation*

> ***Date (of first known publication):*** *09/2019*

> ***Num. Params:*** *1.63B*

> ***Corpus:*** *140 GB of text including: Wikipedia (En, De, Es, Fr), Project Gutenberg, 45 subreddits, OpenWebText2, Amazon Reviews, Europarl and UN data from WMT, question-answer pairs from ELI5, and the MRQA shared task3, which includes the Stanford Question Answering Dataset, NewsQA, TriviaQA, SearchQA, HotpotQA , and Natural Questions*

> ***Lab:*** *Salesforce*

<a name="DALLE"></a>[DALL-E](https://openai.com/blog/dall-e/)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *Caption prediction*

> ***Extension:*** * A differential variational auto-encoder is used to learn the visual codebook. The transformer is a variation of GPT-3*

> ***Application:*** *Text to image*

> ***Date (of first known publication):*** *01/2021*

> ***Num. Params:*** *12B*

> ***Corpus:*** *250 million text-images pairs from the internet*

> ***Lab:*** *OpenAI*
 

<a name="DALLE2"></a>[DALL-E-2](https://openai.com/dall-e-2/)

> ***Family:*** *CLIP, GLIDE*

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *Caption prediction*

> ***Extension:*** *Combines CLIP encoder and Diffusion decoder similar to GLIDE*

> ***Application:*** *Text to image*

> ***Date (of first known publication):*** *04/2022*

> ***Num. Params:*** *3.5B*

> ***Corpus:*** *Combination of the DALL-E and CLIP datasets*

> ***Lab:*** *OpenAI*
 
<a name="DECISION"></a>[Decision Transformers](https://arxiv.org/abs/2106.01345)

> ***Family:*** *GPT, Control Transformers” (not per se a family, but grouping here those transformers that try to model more general control, RL-like, tasks)*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *Next action prediction*

> ***Extension:*** *Decision transformers use a GPT architecture and extend it by encoding trajectories in a way that they can be learned by an auto-regressive task*

> ***Application:*** *General RL (reinforcement learning tasks)*

> ***Date (of first known publication):*** *06/2021*

> ***Num. Params:*** *Same as GPT*

> ***Corpus:*** *Different corpus for different experiments*

> ***Lab:*** *Google/UC Berkeley/Facebook* 


<a name="DIALOGPT"></a>[DialoGPT](https://huggingface.co/docs/transformers/model_doc/dialogpt)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *GPT-2 architecture trained on dialog data*

> ***Application:*** *Text generation in dialog settings*

> ***Date (of first known publication):*** *10/2019*

> ***Num. Params:*** *1.5B*

> ***Corpus:*** *140M Reddit conversations*

> ***Lab:*** *Microsoft*


<a name="DISTILBERT"></a>[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)

> ***Family:*** *BERT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM/NSP*

> ***Extension:*** *Compressed version of BERT using distillation, which is much more efficient given the same number of parameters*

> ***Application:*** *Same as BERT*

> ***Date (of first known publication):*** *10/2019*

> ***Num. Params:*** *66M*

> ***Corpus:*** *Same as BERT*

> ***Lab:*** *Huggingface*


<a name="DQBERT"></a>[DQ-BART](https://arxiv.org/abs/2203.11239)

> ***Family:*** *BART*

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE*

> ***Extension:*** *Adds quantization and distillation to a BART model to improve performance and model size*

> ***Application:*** *Text generation and understanding*

> ***Date (of first known publication):*** *03/2022*

> ***Num. Params:*** *Up to 30x reduction in parameters compared to standard BART*

> ***Corpus:*** *CNN/DM, XSUM, ELI5, WMT16 En-Ro (~1M tokens)*

> ***Lab:*** *Amazon*

<a name="ELECTRA"></a>[ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra)

> ***Family:***

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *RTD*

> ***Extension:***

> ***Application:*** *Same as BERT*

> ***Date (of first known publication):*** *03/2020*

> ***Num. Params:*** *Base = 110M, Large = 330M*

> ***Corpus:*** *Same as BERT except for Large with is same as XLNet*

> ***Lab:*** *Stanford/Google*


<a name="ERNIE"></a>[ERNIE](https://arxiv.org/abs/1905.07129)

> ***Family:*** *BERT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM*

> ***Extension:*** *Uses BERT for Encoder architecture, but stacks and aggregates two of them for text and entities. This architecture could be understood as BERT for text + knowledge graphs*

> ***Application:*** *Knowledge intensive related tasks that might benefit from knowledge graphs or entities such as entity recognition*

> ***Date (of first known publication):*** *05/2019*

> ***Num. Params:*** *114M*

> ***Corpus:*** *English Wikipedia + Wikidata for entitites (note that they initialize model to original BERT parameter values*

> ***Lab:*** *Various Chinese institutions*


<a name="FLAMINGO"></a>[Flamingo](https://arxiv.org/abs/2204.14198)

> ***Family:*** *Chinchilla*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *Log likelihood of text given some visual input*

> ***Extension:*** *It uses a frozen textual language model (like Chinchilla) conditioned on the visual representation, which is encoded from a Normalizer-Free ResNet*

> ***Application:*** *Text to image*

> ***Date (of first known publication):*** *04/2022*

> ***Num. Params:*** *80B (largest)*

> ***Corpus:*** *MultiModal MassiveWeb (M3W): 185 million images and 182 GB text + a number of text paired with image datasets: ALIGN + LTIP (Long Text & Image Pairs) = 312 million images, and VTP (Video & Text Pairs) = 27 million short videos (approximately 22 seconds on average)*

> ***Lab:*** *Deepmind*

<a name="GATO"></a>[Gato](https://www.deepmind.com/publications/a-generalist-agent)

> ***Family:*** *“Control Transformers” (not per se a family, but grouping here those transformers that try to model more general control, RL-like, tasks)*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *MLM (where tokens are either text or agent actions)*

> ***Extension:*** *The standard decoder-only transformer architecture is preceded by an embedding layer that can embed text and images, plus add position encodings to add spatial information when applicable.*

> ***Application:*** *Gato presents a generalizable agent that can be used beyond text to tasks such as playing Atari or controlling a robot arm. *

> ***Date (of first known publication):*** *05/2022*

> ***Num. Params:*** *1.2B*

> ***Corpus:*** *1.5T tokens including standard text (e.g. MassiveText), vision (e.g. ALIGN), and simulation environments (e.g. ALE Atari, or RGB Stacking Real Robot)*

> ***Lab:*** *Deepmind*


<a name="GLAM"></a>[GLaM](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)

> ***Family:*** *Transformer*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *GLaM introduces a Mixture of 64 Experts to increase parameter count and generalization properties in a somewhat standard decoder-only. Transformer architecture. Only two experts get activated at a time per token, which makes the model also more efficient in training and inference.*

> ***Application:*** *General language modeling*

> ***Date (of first known publication):*** *12/2021*

> ***Num. Params:*** *1.2T across 64 experts, but only 96B get activated for inference*

> ***Corpus:*** *1.6T tokens including web pages filtered by Wikipedia and books for quality*

> ***Lab:*** *Google*

<a name="GLIDE"></a>[GLIDE](https://arxiv.org/abs/2112.10741)

> ***Family:*** *Diffusion models*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** * Caption prediction*

> ***Extension:*** *GLIDE can be seen as an extension of the ADM (Ablated Diffusion Model) by the same authors. However, ADM is not per se a transformer architecture although it does resemble one in some of the configurations the authors use. Given that ADM is by the same authors and was quickly followed up by GLIDE, I think it is fair to consider GLIDE as the first of its kind.*

> ***Application:*** *Text to image*

> ***Date (of first known publication):*** *12/2021*

> ***Num. Params:*** *3.5B diffusion model (2.3B for visual encoding, 1.2B for textual) + 1.5B for model for upsampling*

> ***Corpus:*** *Same as DALL-E*

> ***Lab:*** *OpenAI*

[<a name="GCVIT"></a>Global Context ViT](https://arxiv.org/abs/2206.09959)

> ***Family:*** *ViT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *Image classification*

> ***Extension:*** *hierarchical ViT architecture consisting of local and global self-attention modules*

> ***Application:*** *APPLICATION*

> ***Date (of first known publication):*** *06/2022*

> ***Num. Params:*** *90M*

> ***Corpus:*** *Imagenet-1K and other task dependent dataasets*

> ***Lab:*** *NVidia*
 

<a name="GOPHER"></a>[Gopher](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Same as GPT-2 but use RSNorm instead of LayerNorm and relative positional encoding rather than absolute

> **Application:** Mostly Language Modeling and NLU, but also extensible like GPT

> **Date (of first known publication):** 12/2021

> **Num. Params:** 280B

> **Corpus:** Massive Text (2.35 billion documents, or about 10.5 TB of text including Massive Web, Books, Github, News, C4, and Wikipedia.

> ***Lab:*** *Deepmind*


<a name="gophercite"></a>[GopherCite](https://arxiv.org/abs/2203.11147)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** GopherCite is based on Gopher but adds a step using RLHP (Reinforcement Learning from Human Preferences) to learn whether not only a response is plausible but also supported

> **Application:** Dialog systems, Q&A, general language generation tasks

> **Date (of first known publication):** 03/2022

> **Num. Params:** 280B

> **Corpus:** Same as Gopher plus specific dataset generated in the RLHP process

> ***Lab:*** *Deepmind*

<a name="GPT"></a>[GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:***

> ***Application:*** *Text generation, but adaptable to many other NLP tasks when fine tuned.*

> ***Date (of first known publication):*** *06/2018*

> ***Num. Params:*** *117M*

> ***Corpus:*** *Unsupervised Pretraining on BookCorpus dataset. Supervised Finetuning on several task-specific datasets including SNLI, RACE, Quora…*

> ***Lab:*** *OpenAI*


<a name="GPT2"></a>[GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Minor extensions to the GPT architecture (e.g. layer normalization moved to the input of each sub-layer, or increased context size from 512 to 1024)*

> ***Application:*** *Text generation, but adaptable to many other NLP tasks when fine tuned.*

> ***Date (of first known publication):*** *02/2019*

> ***Num. Params:*** *1.5B*

> ***Corpus:*** *8 million web pages (40 GB). 10X GPT . WebText dataset is created by crawling all links at Reddit with at least 3 Karma points.*

> ***Lab:*** *OpenAI*


<a name="GPT3"></a>[GPT-3](https://github.com/openai/gpt-3)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Same as GPT-2 with the only addition of alternating dense and locally banded sparse*

> *attention patterns, inspired by the Sparse Transformer*

> ***Application:*** *Initially text generation, but has over time been used for a large range of applications in areas such as code generation, but also image and audio generation*

> ***Date (of first known publication):*** *05/2020*

> ***Num. Params:*** *175 B*

> ***Corpus:*** *~ 500B tokens including CommonCrawl (410B), WebText2 (19B), Books1 (12B), Books2 (55B), and Wikipedia (3B)*

> ***Lab:*** *OpenAI*

<a name="GPT35"></a>[GPT-3.5](https://beta.openai.com/docs/model-index-for-researchers)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *The GPT3.5 series includes a number of models like Davinci-003. They are basically versions of the InstructGPT model. See [here](https://scale.com/blog/gpt-3-davinci-003-comparison) for details on the comparison of the performance to older GPT3 models.*

> ***Application:*** *Dialog and general language, but there is a code specific model too*

> ***Date (of first known publication):*** *10/2022*

> ***Num. Params:*** *175B*

> ***Corpus:*** *Same as InstructGPT*

> ***Lab:*** *OpenAI*


<a name="GPTINSTRUCT"></a>[GPTInstruct](https://openai.com/blog/instruction-following/)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** GPTInstruct starts off with a pretrained GPT3 model and adds reward modeling through reinforcement learning after a supervised finetuning

> **Application:** Knowledge-intensive dialog or language tasks

> **Date (of first known publication):** 01/2022

> **Num. Params:** Same as GPT3

> **Corpus:** Same as GPT3 for pretraining, but finetuned and optimized using labeler data and prompts

> ***Lab:*** *OpenAI*


<a name="GPTNEO"></a>[GPT-Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Similar to GPT-2 but uses local attention in every other layer with a window size of 256 tokens*

> ***Application:*** *Text generation, but adaptable to many other NLP tasks when fine tuned*

> ***Date (of first known publication):*** *03/2021*

> ***Num. Params:*** *5B, 2.7B (XL)*

> ***Corpus:*** *Pile — 840 GB open source text dataset that combines 22 pre existing datasets*

> ***Lab:*** *EleutherAI*


<a name="GPTNEOX"></a>[GPT-NeoX-20B](https://arxiv.org/abs/2204.06745)

> ***Family:*** *GPT*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Similar to GPT-3 with rotary encoders instead of positional, parallel attention and feed forward layers, different initialization, and all dense layers instead of alternate dense/sparse*

> ***Application:*** *same as GPT-3*

> ***Date (of first known publication):*** *04/2022*

> ***Num. Params:*** *20B*

> ***Corpus:*** *Pile — 840 GB open source text dataset that combines 22 pre existing datasets*

> ***Lab:*** *EleutherAI*


<a name="html"></a>[HTML](https://arxiv.org/abs/2107.06955)

> ***Family:*** *BART*

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE*

> ***Extension:*** *As opposed to BART, they don’t do sentence shuffling*

> ***Application:*** *General purpose language model that allows structured HTML prompting *

> ***Date (of first known publication):*** *07/2021*

> ***Num. Params:*** *400M*

> ***Corpus:*** *23TB of simplified HTML extracted from CommonCrawl*

> ***Lab:*** *Facebook*


<a name="IMAGEN"></a>[Imagen](https://imagen.research.google/)

> ***Family:*** *T5, CLIP, Diffusion models*

> ***Pretraining Architecture:*** *T5 (or CLIP or BERT) for frozen text encoder + U-net architecture for cascaded diffusion models for text to image*

> ***Pretraining Task:*** *image/text pair prediction*

> ***Extension:*** *Imagen adds a few extensions to the U-net diffusion architecture (pooled embedding vector, cross attention over text embeddings, and Layer Normalizations)*

> ***Application:*** *Text to image*

> ***Date (of first known publication):*** *06/2022*

> ***Num. Params:*** *2B*

> ***Corpus:*** *a combination of internal datasets, with ≈ 460M image-text pairs, and the publicly available Laion dataset, with ≈ 400M image-text pairs*

> ***Lab:*** *Google*

<a name="JURASSIC1"></a>[Jurassic-1](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Very similar to GPT-3, but far more parameters and improved training efficiency mostly because of the improved tokenizer. Also, different ratio of depth to breadth

> **Application:** Similar to GPT-3

> **Date (of first known publication):** 09/2021

> **Num. Params:** 178B (Jumbo), 7.5B (Large)

> **Corpus:** 300B tokens (same as GPT-3)

> ***Lab:*** *AI21*


<a name="LAMDA"></a>[LAMDA](https://ai.googleblog.com/2022/01/lamda-towards-safe-grounded-and-high.html)

> ***Family:*** *Transformer*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *LAMDA focuses on how to improve safety, quality, and groundeness using different fine-tuning strategies*

> ***Application:*** *General language modeling*

> ***Date (of first known publication):*** *01/2022*

> ***Num. Params:*** *137B*

> ***Corpus:*** *1.56T words from public dialog data and other public web documents*

> ***Lab:*** *Google*

<a name="MBART"></a>[mBART](https://huggingface.co/docs/transformers/model_doc/mbart)

> ***Family:*** *BART*

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE*

> ***Extension:***

> ***Application:*** *Translation*

> ***Date (of first known publication):*** *01/2020*

> ***Num. Params:*** *Same as BART*

> ***Corpus:*** *CC25 Corpus includes 25 monolingual corpuses in different languages. Largest corpuses are English (300 GB) and Russian (280GB)*

> ***Lab:*** *Facebook*


<a name="MEGATRON"></a>[Megatron](https://github.com/NVIDIA/Megatron-LM)

> **Family:** GPT/BERT/T5

> **Pretraining Architecture:** Encoder or Decorder, depending on the base model

> **Pretraining Task:** Same as base model

> **Extension:** Megatron is a family of models that extend previously known architectures (namely GPT-2 and BERT originally, but also T5 more recently) by introducing model parallelism primitives. In the case of BERT, the authors also replace the next sentence prediction head with sentence order prediction and use whole word n-gram masking.

> **Application:** Same as base model

> **Date (of first known publication):** 03/2020

> **Num. Params:** 8.3B (GPT-like), 3.9B (BERT-like)

> **Corpus:** Original paper uses an aggregate dataset consisting of Wikipedia), CC-Stories), RealNews, and OpenWebtext

> ***Lab:*** *NVidia*

<a name="MINERVA"></a>[Minerva](https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html)

> ***Family:*** *PaLM*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Extends PaLM by fine-tuning on the mathematical dataset*

> ***Application:*** *Mathematical reasoning*

> ***Date (of first known publication):*** *06/2022*

> ***Num. Params:*** *540B*

> ***Corpus:*** * Same as PaLM + ​​118GB dataset of scientific papers from the arXiv preprint server and web pages that contain mathematical expressions using LaTeX, MathJax, or other mathematical typesetting formats*

> ***Lab:*** *Google*

<a name="MTNLG"></a>[MT-NLG](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) (Megatron Touring NLG)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Uses parallelization similar to Megatron to train a LM double the size of GPT-3

> **Application:** Language generation and others (similar to GPT-3)

> **Date (of first known publication):** 10/2021

> **Num. Params:** 530B

> **Corpus:** [The Pile](https://arxiv.org/abs/2101.00027) (800GB dataset) + 2 Common Crawl snapshots

> ***Lab:*** *NVidia*


<a name="OPT"></a>[OPT](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/)

> ***Family:*** *GPT-3*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Basically same architecture as GPT-3 but with some training improvements introduced in Megatron-LM*

> ***Application:*** *Same as GPT-3*

> ***Date (of first known publication):*** *05/2022*

> ***Num. Params:*** *175B (and other smaller versions)*

> ***Corpus:*** *180B tokens = RoBERTa + the Pile + PushShift.io Reddit*

> ***Lab:*** *Facebook*

<a name="PALM"></a>[PalM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)

> ***Family:*** *Transformer*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Palm uses a typical decoder-only transformer architecture, but adds quite a few extensions: SwiGLU activations, parallel layers, multi-query attention, RoPE embeddings, Shared Input-Output Embeddings, no biases, and a 256k SentencePiece vocabulary generated from the training data.*

> ***Application:*** *PalM is designed as a general purpose language model with applicability to hundreds of different language tasks*

> ***Date (of first known publication):*** *04/2022*

> ***Num. Params:*** *540B*

> ***Corpus:*** *780B tokens from filtered webpages, books, Wikipedia, news articles, source code, and social media conversations. Code includes 24 programming languages.*

> ***Lab:*** *Google*

<a name="PEGASUS"></a>[Pegasus](https://huggingface.co/docs/transformers/model_doc/pegasus)

> ***Family:***

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE (more concretely GSG) and MLM*

> ***Extension:*** *Extends vanilla Transformer by using a different pretraining task (GSG: Gap Sentence Generation) that is better suited for summarization*

> ***Application:*** *Summarization*

> ***Date (of first known publication):*** *12/2019*

> ***Num. Params:*** *Base = 223M, Large = 568M*

> ***Corpus:*** *C4 (750GB) + HugeNews (3.8 TB)*

> ***Lab:*** *UCL/Google*

<a name="ROBERTA"></a>[RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)

> ***Family:*** *BERT*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM (Dynamic)*

> ***Extension:*** *Extension of BERT with optimized training procedure and more data*

> ***Application:*** *Same as BERT*

> ***Date (of first known publication):*** *07/2019*

> ***Num. Params:*** *356M*

> ***Corpus:*** *Same as BERT + CC News + OpenWebText + Stories (~33B Tokens)*

> ***Lab:*** *UW/Google*

<a name="SEEKER"></a>[SeeKer](https://parl.ai/projects/seeker/)

> **Family:** GPT (but can extend any family)

> **Pretraining Architecture:** Encoder/decoder or decoder only, depending on the base model it’s extending

> **Pretraining Task:** Depends on the base model

> **Extension:** SeeKer is an extension that can be applied to any Transformer architecture by introducing “search”, “knowledge”, and “response” modules that are introduced during pretraining

> **Application:** Same as base models

> **Date (of first known publication):** 03/2022

> **Num. Params:** Depends on the base model

> **Corpus:** Same as base model

> ***Lab:*** *Facebook*

<a name="Sparrow"></a>[Sparrow](https://arxiv.org/abs/2209.14375)

> **Family:** GPT 

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Starts from the Chinchilla 70B model but adds RLHF (Reinforcement Learning with Human Feedback). It also adds inline evidence a la GopherCite

> **Application:** Dialog agents and general language generation applications like Q&A

> **Date (of first known publication):** 09/2022

> **Num. Params:** 70B

> **Corpus:** Same as Chinchilla + interactive data gathering with human annotators during the RLHF process

> ***Lab:*** *Deepmind*

<a name="stablediffusion"></a>[StableDiffusion](https://huggingface.co/CompVis/stable-diffusion)

> **Family:** Diffusion

> **Pretraining Architecture:** Encoder/Decoder

> **Pretraining Task:** Caption prediction

> **Extension:** Stable diffusion is basically the Latent Diffusion model developed by LMU Munich researchers + some learnings on conditional diffusion from DALL-e and Imagen

> **Application:** Text to image 

> **Date (of first known publication):** 12/2021

> **Num. Params:** 890M (although there are different, smaller, variants)

> **Corpus:** LAION-5B, a publicly available dataset derived from Common Crawl

> ***Lab:*** *LMU Munich + Stability.ai + Eleuther.ai*

<a name="SWIN"></a>[Swin Transformer](https://github.com/microsoft/Swin-Transformer)

> **Family:** ViT

> **Pretraining Architecture:** Encoder

> **Pretraining Task:** Same as ViT

> **Extension:** Extends ViT by replacing the standard multi-head self attention (MSA) module by a module based on shifted windows (Swin) allowing ViT-like architectures to generalize to higher resolution images

> **Application:** Image (object detection, image classification..)

> **Date (of first known publication):** 03/2021

> **Num. Params:** 29M-197M

> **Corpus:** Imagenet and Imagenet-22k

> ***Lab:*** *Facebook*

<a name="SWITCH"></a>[Switch](https://arxiv.org/abs/2101.03961)

> ***Family:*** *T5*

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE*

> ***Extension:*** *Goal to increase parameter count while keeping FLOP operations constant by using efficient routing of MoE (Mixture of Experts)*

> ***Application:*** *General language tasks (e.g. question answering)*

> ***Date (of first known publication):*** *01/2021*

> ***Num. Params:*** *1T*

> ***Corpus:*** *Colossal Clean Crawled Corpus*

> ***Lab:*** *Google* 

<a name="T5"></a>[T5](https://huggingface.co/docs/transformers/model_doc/t5)

> ***Family:***

> ***Pretraining Architecture:*** *Encoder/Decoder*

> ***Pretraining Task:*** *DAE*

> ***Extension:*** *Same as original Transformer with some additions such as relative positional embeddings like Transformer XL*

> ***Application:*** *General language tasks including machine translation, question answering, abstractive summarization, and text classification*

> ***Date (of first known publication):*** *10/2019*

> ***Num. Params:*** *11 B (up to)*

> ***Corpus:*** *Colossal Clean Crawled Corpus (C4) — Cleaned up version of the Common Crawl dataset — 750 GB*

> ***Lab:*** *Google*

<a name="TRAJECTORY"></a>[Trajectory Transformers](https://arxiv.org/abs/2106.02039)

> ***Family:*** *GPT, Control Transformers” (not per se a family, but grouping here those transformers that try to model more general control, RL-like, tasks)*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *predict most likely sequence*

> ***Extension:*** *Similarly to the Decision transformers, the main extension introduced by Trajectory Transformers is a way to encode a trajectory (state, actions, rewards)*

> ***Application:*** *General RL (reinforcement learning tasks)*

> ***Date (of first known publication):*** *06/2021*

> ***Num. Params:*** *Smaller architecture than GPT*

> ***Corpus:*** *D4RL dataset and other RL datasets depending on the task at hand*

> ***Lab:*** *UC Berkeley*

<a name="TRANSFORMERXL"></a>[Transformer XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl)

> ***Family:***

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *LM*

> ***Extension:*** *Relative positioned embeddings enable longer-context attention when compared to vanilla Transformer model*

> ***Application:*** *General language tasks*

> ***Date (of first known publication):*** *01/2019*

> ***Num. Params:*** *151M*

> ***Corpus:*** *Different training datasets depending on experiments, but baseline is Wikitext-103*

> ***Lab:*** *CMU/Google*

<a name="TURING"></a>[Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)

> **Family:** GPT

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** LM

> **Extension:** Optimized version of GPT2 with optimal hyperparameters and software/hardware platform to improve training

> **Application:** Same as GPT-2/3

> **Date (of first known publication):** 02/2020

> **Num. Params:** 17B originally, up to 530B more recently

> **Corpus:** Highest quality subset from The Pile + 2 CC snapshots (339B tokens)

> ***Lab:*** *Microsoft*

<a name="VIT"></a>[ViT](https://huggingface.co/docs/transformers/model_doc/vit)

> **Family:** BERT

> **Pretraining Architecture:** Encoder

> **Pretraining Task:** Image classification

> **Extension:** Extension of BERT architecture to train on patches of images

> **Application:** Image classification

> **Date (of first known publication):** 10/2020

> **Num. Params:** 86M(Base) to 632M (Huge)

> **Corpus:** From standard Imagenet to JFT-300M (large inhouse dataset)

> ***Lab:*** *Google*

<a name="WUDAO"></a>[Wu Dao 2.0](https://en.wikipedia.org/wiki/Wu_Dao)

> **Family:** GLM (General Language Model)

> **Pretraining Architecture:** Decoder

> **Pretraining Task:** Autoregressive blank infilling

> **Extension:** Similar to GPT in that it uses a Decoder/autoregressive architecture but applies a different pretraining task proposed in the GLM family of models. Besides, Wu Dao uses a “[Fast Mixture of Experts](https://github.com/laekov/fastmoe)” approach to scale training to trillions of parameters

> **Application:** Language and multimodal (particularly image)

> **Date (of first known publication):** 06/2021

> **Num. Params:** 1.75T

> **Corpus:** ?

> ***Lab:*** *Beijing Academy of Artificial Intelligence*

<a name="XMLROBERTA"></a>[XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)

> ***Family:*** *RoBERTa*

> ***Pretraining Architecture:*** *Encoder*

> ***Pretraining Task:*** *MLM (Dynamic)*

> ***Extension:*** *An extension of RoBERTa that introduces small parameter tuning insights in the context of multilingual applications*

> ***Application:*** *Translation and other cross-lingual language tasks*

> ***Date (of first known publication):*** *10/2019*

> ***Num. Params:*** *Base = 270M, Large = 550M*

> ***Corpus:*** *Cleaned Common Crawl in 100 languages*

> ***Lab:*** *Facebook*

<a name="XLNET"></a>[XLNet](https://huggingface.co/docs/transformers/model_doc/xlnet)

> ***Family:*** *Transformer XL*

> ***Pretraining Architecture:*** *Decoder*

> ***Pretraining Task:*** *PLM*

> ***Extension:*** *This model basically adapts Transformer XL architecture to permutation-based LM*

> ***Application:*** *General language tasks*

> ***Date (of first known publication):*** *05/2019*

> ***Num. Params:*** *Base=117M, Large=360M*

> ***Corpus:*** *Same as BERT + Giga5 (16GB text) + and aggressively filtered ClueWeb 2012-B (19GB), Common Crawl (110 GB)*

> ***Lab:*** *CMU/Google*

### Further reading

Most of the following references have already been mentioned in the post. However, it is worth listing them here in case you need more details:

- The Huggingface Transformers [documentation](https://huggingface.co/course/chapter1/1?fw=pt) and course is extremely good and comprehensive. I have used myself in this post, and I can’t recommend enough as a natural follow up to what you will find here.
- [A survey of transformers](https://arxiv.org/abs/2106.04554) (Lin et al. 2021) includes a 40 page long survey wit over 170 references and a full blown taxonomy.
- [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf) is also a very comprehensive survey that includes many of the pretrained models with a particular focus on NLP

![](/blog/images/02-07.png)

- [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271) (Quiu et al. 2021) is another 30+ pages long survey that focuses on pretrained models for NLP.

![](/blog/images/02-08.png)
