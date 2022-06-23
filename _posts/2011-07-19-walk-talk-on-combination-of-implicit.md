---
id: 86
title: 'Walk the Talk: On the Combination of Implicit and Explicit Feedback'
date: '2011-07-19T05:32:00+00:00'
author: 'Xavier Amatriain'
##layout: post
permalink: /walk-talk-on-combination-of-implicit/
image: /wp-content/uploads/2011/07/up-box.png
categories:
    - Uncategorized
tags:
    - explicit
    - 'explicit feedback'
    - implicit
    - 'music recommenders'
    - 'recommender systems'
    - 'user modeling'
---

Last week, [Denis Parra](http://www.sis.pitt.edu/%7Edparra/) presented our paper entitled “Walk the Talk: Analyzing the Relation between Implicit and Explicit Feedback for Preference Elicitation” at the [UMAP conference](http://www.umap2011.org/). The paper won Denis the best-student paper award (Congratulations!).

The paper presents our initial work in analyzing the relation between implicit and explicit feedback. In short, the main question we wanted to answer is how does the self-reported preferences users give in a typical 5-star interface relate to what they actually do when looking at their consumption patterns. Our hypothesis was that there should exist simple models that relate both kinds of feedback. Finding a way to robustly convert implicit feedback into explicit ratings would open up the door to applying well-known methods with implicit feedback. But, much more importantly, we could then combine both kinds of input in a single model.

In order to test our hypothesis, we prepared an experiment in the music domain. We asked last.fm users to take a [survey](http://localhost:8080/wordpress/2010/08/study-on-online-music-taste-call-for.html) in which we queried them about how much they liked albums that were already in their listening history. With this data in hand, we could analyze the relation between implicit and explicit feedback and try to fit a simple model.

I recommend you read the [full paper](http://bit.ly/r1mvkK) if you want to get the longer story of our findings, but here is a brief summary:

- There is a strong correlation between implicit feedback and self-reported preference (see figure below)
- Variables such as recentness of interaction or overall popularity do not have significant effect. Note that in [a previous study](http://www.princeton.edu/%7Emjs3/salganik_watts08.pdf) by Salganik &amp; Duncan Watts, global popularity was found to affect users perceived quality. However, in that case and as opposed to ours, users were made aware of the popularity.
- Interaction effect: When listening to music, some people prefer to listen to isolated songs or albums. The way they interact with music, affects the way they report their taste.

[![](http://localhost:8080/wordpress/wp-content/uploads/2011/07/up-box.png)](http://1.bp.blogspot.com/-gtrwSvrIpEI/TifEaf_CfYI/AAAAAAAAANI/IzeMVpRZbaQ/s1600/up-box.png)  
After our analysis, we then construct a linear model that takes into account these variables by performing a linear regression. Once we have built these models, we can evaluate their performance in a regular recommendation scenario by measuring the error in predicting ratings in a hold-out dataset.

This paper represents an initial but very promising line of work that we have already improved in several ways such as the use of logistic instead of linear regression to account for the non-linearity of the rating scale or the use of the regression model as a way to combine both implicit and explicit feedback. But I will leave those findings for a future post.