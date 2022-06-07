---
id: 82
title: 'The Recommender Problem &#038; the Presentation Context'
date: '2011-09-26T06:41:00+00:00'
author: 'Xavier Amatriain'
layout: post
permalink: /recommender-problem-presentation/
image: /wp-content/uploads/2011/09/NetflixInterface.jpg
categories:
    - Uncategorized
tags:
    - interaction
    - 'presentation data'
    - 'recommender systems'
---

<span style="font-size: 12pt;">In the traditional formulation of the “Recommender Problem”, we have pairs of items and users and user feedback values for very few of those dyads. The problem is formulated as the finding of a utility function or model to estimate the missing values.</span>

In many real-world situations, feedback will be implicit<span style="font-weight: bold;">\*\*</span> and binary in nature. For instance, in a web page you will have users visiting a url, or clicking on an add as a positive feedback. In a music service, a user will decide to listen to a song. Or in a movie service, like Netflix, you will have users deciding to watch a title as an indication that the user liked the movie. In these cases, the recommendation problem becomes the prediction of the probability a user will interact with a given item. There is a big shortcoming in using the standard recommendation formulation in such a setting: we don’t have negative feedback. All the data we have is either positive or missing. And the missing data includes both items that the user explicitly chose to ignore because they were not appealing and items that would have been perfect recommendations but were never presented to the user.

A similar issue has been dealt with in traditional data mining research, where classifiers need to be trained only using positive examples. In the ["Learning Classifiers from Only Positive and Unlabeled Examples"](http://www.cse.ucsd.edu/users/elkan/posonly.pdf) SIGKDD 08 paper, the authors present a method to convert unlabeled examples into both a positive and a negative example, each with a different weight related to the probability that a random exemplar is positive or negative. Another solutions to this issue is presented in the “[Collaborative Filtering for Implicit Feedback Datasets](http://research.yahoo.com/files/HuKorenVolinsky-ICDM08.pdf)” paper by Hu, Koren and Volinsky. In this work, the authors binarize the implicit feedback values: any feedback value greater than zero means positive preference, while any value equal to zero is converted to no preference. A greater value in the implicit feedback value is used to measure the “confidence” in the fact the user liked the item, but not in measuring “how much” the user liked it. Yet another approach to inferring positive and negative feedback from implicit data is presented in the paper I co-authored with Dennis Parra, and I presented in a ["previous pos](/_posts/walk-talk-on-combination-of-implicit.html). There, we argue that implicit data can be transformed to positive and negative feedback if aggregated at the right level. For example, the fact that somebody listened only once to a single track in an album can be interpreted as the user not liking that album.

In many practical situations, though, we have more information than the simple binary implicit feedback from the user. For unlabeled examples that the user did not directly interact with, we can expect to have other information. In particular, we might be able to know whether they were shown to the user or not. This adds very valuable information, but slightly complicates the formulation of our recommendation problem. We now have three different kinds of values for items: positive, presented but not chosen, and not presented. And this is only if we simplify the model. In reality, information related to the presentation can be much richer than this and we might be able to derive data like the probability the user actually saw the item or weigh in different interaction events such as mouse overs, scrolls…

| ![](/blog/images/NetflixInterface.jpg) |

At Netflix, we are working on different ways to add this rich information related to presentations and user interaction to the recommender problem. That is why I was especially interested in finding out that this year’s SIGIR best student paper award has been awarded to a paper that addresses this issue. In the paper [“Collaborative Competitive Filtering: Learning  
Recommender Using Context of User Choice"](http://www.cc.gatech.edu/%7Esyang46/papers/SIGIR11CCF.pdf), the authors present an extension to traditional Collaborative Filtering by encoding into the model not only the **collaboration** between similar users and items, but also the **competition** of items for user attention. They derive the model as an extension to standard latent factor models by taking into account the context in which the user makes the decision. That is, the probability I decide to select a given item depends on which are the other items I have as an alternative. Results are preliminary but promising. And, this work is definitely an interesting and appealing starting point for an area with many practical applications.

However, there are many possible improvements to the model. One of them, mentioned by the authors, is the need to take into account the so-called **position bias**. An item that is presented in the first position of a list has many more possibilities to be chosen than one that is farther down. This effect is well-known in the search community and has been studied from several angles. I would recommend, for instance to read some of the very interesting papers on this topic by Thorsten Joachims and his students. In the paper [“How Does Clickthrough Data Reflect Retrieval Quality?"](http://www.cs.cornell.edu/People/tj/publications/radlinski_etal_08b.pdf), for instance, they show how arbitrarily swapping items in a search result list has almost no effect. This proves that the positioning of the element can be a most important factor than how relevant the item is.

I would love to hear of other ideas or approaches to deal with this new version of the recommender problem that includes, and would encourage researchers in the area to address an issue of huge potential impact.

***Note**: I am using the word implicit here in the traditional sense in the recommendation literature. The truth is that a user selecting an item is in fact **explicit** information. However, it can be considered implicit in that the user is informing about the preferences indirectly by comparing the item to others in a context.
