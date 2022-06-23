---
id: 31
title: "Data as Prior/Innate knowledge for Deep Learning\_models"
date: '2019-04-18T00:00:00+00:00'
author: xamat
##layout: post
permalink: /data-as-prior-innate-knowledge-for-deep-learning-models-23898363a71a/
reading_time:
    - ''
    - ''
categories:
    - Uncategorized
---

There is a long history of debate about how much of human knowledge is innate and how much is learned from experience/data. This is also known as the nature vs. nurture debate (See for example some of [Gary Markus’ recent papers](https://arxiv.org/pdf/1801.05667.pdf) for more details on this). Traditional Bayesian methods explicitly model these two components with a prior component that can be derived from existing knowledge and a likelihood that is learned from the data.

The recent rapid advances in deep learning continue to demonstrate the significance of end-to-end training with no apriori knowledge (such as domain-aware feature engineering) in many computer vision and NLP tasks. However, when models require reasoning or need to do forward prediction, or operate in low data regime, most AI researchers and practitioners will agree that incorporating prior knowledge, along with end-to-end training can introduce better inductive bias, beyond what is provided in the training data. This is especially true in complex domains such as healthcare where precision of the systems are central, and the combinatorics required for generalization is large.

Despite the general agreement that innate structure and learned knowledge need to be combined, there is no simple approach to incorporate this innate structure into learning systems. As a scientific community, we are just starting to research approaches to incorporate prior into deep learning models. (See some of these [recent approaches](https://openreview.net/forum?id=r1E0OsA9tX) or [discussions](https://www.quora.com/Is-there-a-way-to-encode-prior-knowledge-in-deep-neural-networks) on how to include prior knowledge in DL networks). In fact, not surprisingly, there is little consensus on whether priors needs to be hard-coded into the model or can be learned through, for example, using a reward function (see [this great summary](https://www.abigailsee.com/2018/02/21/deep-learning-structure-and-innate-priors.html) of the debate between LeCun and Manning)

The prior is, [generally speaking](https://en.wikipedia.org/wiki/Prior_probability), a probability distribution that expresses one’s beliefs about a quantity before some evidence is taken into account. If we restrict ourselves to an ML model, the prior can be thought as of the distribution that is imputed before the model starts to see any data. However, a more holistic view into the process should consider not only the model training, but the design of the system as a whole, including the choice of model and the data. Therefore, the prior includes anything from the choice of the algorithm itself to the way that the data is labeled (this is in fact related to what [Markus calls](https://arxiv.org/pdf/1801.05667.pdf) innate algorithms, representational formats, and knowledge).

Therefore, a perhaps not so obvious way to inject prior knowledge and beliefs into the process is through the training data selection. Indeed, by selecting and labeling training samples for any supervised learning model, we encode knowledge into the learning model. The data labeling process might be trivial in some domains, but will require a good deal of domain knowledge in others. For example, to label cats and dogs we need to understand the difference between these two animals, which might seem very “common sense”, but to label medical images as “cancer” or “not cancer”, we need deep medical expertise.

As yet another example, an interesting but negative case of injecting “prior belief” in the data selection/labeling process is related to some well-known examples of algorithmic bias. If the belief of whoever is creating the predictive model is that people of a certain background are more likely to commit a crime, this belief is likely to be fed into the training and testing process by selecting and labeling a dataset that confirms that prior. In fact, most existing datasets, like collections of written texts, have priors/biases that will affect the learned model (see [Semantics derived automatically from language corpora contain human-like biases](http://science.sciencemag.org/content/356/6334/183))

In [a recent paper](https://medium.com/curai-tech/the-science-of-assisting-medical-diagnosis-from-expert-systems-to-machine-learned-models-cc2ef0b03098), we presented an approach to incorporate prior knowledge into DL systems by using synthetic data. While we presented this approach for a particular application (medical diagnosis), I believe this has broader implications that can be used in many other domains. To be clear, synthetic data has been used to some extent in other domains. For example, self-driving researchers [have used synthetic data from the Grand Theft Auto](https://www.technologyreview.com/s/602317/self-driving-cars-can-learn-a-lot-by-playing-grand-theft-auto/) videogame to train machine learning systems. However, our approach follows a flexible and reproducible process that involves the following steps:

1. Use an existing rule-based expert system to generate data
2. Inject noise to the generated data to make learned system more resilient
3. Combine synthetic data from real-world data
4. Train a deep learning system

The diagram below illustrates the process.

![](/blog/images/11-01.png)

As we showed in the paper, the process of combining synthetic data with noise plus real world data to train a deep learning model has several advantages:

1. It allows to introduce innate domain-specific knowledge
2. It allows to control the balance of innate vs. real-world learned knowledge by simply controlling the balance between synthetic and real-world data.
3. Adding noise to the synthetic data allows us to learn a system that is more robust to noisy data in inference time
4. Introducing real-world data means that we can have an extensible continuous learning system that is better over time than the expert system specified by the domain experts.

In our paper, we show a concrete example of how this approach can be successfully applied to modeling a medical diagnosis system. However, again, I think that this could be useful for others and look forward to hearing about it being applied elsewhere or for feedback and comments.
