---
id: 6
title: 'Football or Futbol? Or why Deep Learning will not make other Machine Learning approaches obsolete'
date: '2016-04-20T00:00:00+00:00'
author: xamat
layout: post
permalink: /football-or-futbol-or-why-deep-learning-will-not-make-other-machine-learning-approaches-obsolete-666658ed4167/
reading_time:
    - ''
    - ''
categories:
    - Uncategorized
---

You have probably heard a lot about Deep Learning and how it is taking over the world in general, and the area of Machine Learning in particular. But, does that mean that Deep Learning will be the solution to all our (machine learning) problems? No. There are several reasons why there will always be a place for other algorithms to be better suited than deep learning in some applications.

**Need for feature engineering**

There are many cases where you need to have an understanding of the domain in order to have optimal results. While some proponents of Deep Learning describe their approach as being general-purpose, I don’t think that will ever be true.

**Occam’s razor**

Given two models that perform more or less equally, you should always prefer the one that is less complex. Even the authors of the [Deep Learning](http://www.deeplearningbook.org/) book mention this. For this reason, there will always be cases where Deep Learning will not be preferred, even if it has managed to squeeze an extra 1% in accuracy on the testing set.

**Ensembles**

Even in the very unlikely case where most ML problems end up being suited for Deep Learning Approaches, there will always be a place for ensembles. Given the output of a Deep Learning prediction, I am sure you will be able to combine it with some other model or feature to improve the results.

**A simple example: Football or Futbol?**

Let’s imagine that your very demanding boss asks you to implement an image classifier to detect whether sport images contain scenes coming from a [Football (US)](https://www.quora.com/topic/Football-US) game or a [Football (Soccer)](https://www.quora.com/topic/Football-Soccer-2) game. You are well read in Deep Learning literature, so you train a two-class classifier using [Convolutional Neural Networks (CNNs)](https://www.quora.com/topic/Convolutional-Neural-Networks-CNNs) feeding it thousands of labeled images containing both sports. The classifier works out pretty good and you get an accuracy of, say, 95%.

![](/blog/images/16-01.png)
    
    Football or Futbol?</figcaption></figure>You present the work to your boss and she is super impressed. Next thing you know, your deep classifier is being used on a dataset of images from Spain. Now of course, every now and then, the image is really poor quality and blurry, so the classifier has a hard time. Let’s say it finds an image where the probability of it being Football is 60%, should it decide to classify it as such? Of course not! An image with 60% probability of being Football coming from a Spanish database should be classified as Futbol for sure… almost nobody watches Football in Spain!

As a matter of fact, in this example, a dumb classifier that just labels everything as Futbol for this dataset would manage to have much more than a 95% accuracy and would work much better than your fancy Deep Neural Network!

So, what’s going on? Well, you should have thought better about the problem you had at hand and realized that it it not only an image recognition problem. Things like “popularity” or “location” could make a huge difference. Of course these are simple features, you could think about many more, but you get the point.

You could feed those features into your problem by using an ensemble as mentioned above. Or you could use a simple Bayesian prior. You could even force it as a feature into the Neural Network.

In any case, this simple example proves that you will always have the need to understand your domain and do some feature engineering, favor simple models whenever possible, and most likely use ensembles.

*Originally published at* [*www.quora.com*](https://www.quora.com/Will-deep-learning-make-other-Machine-Learning-algorithms-obsolete/answer/Xavier-Amatriain)*.*
