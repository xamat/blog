---
id: 88
title: 'Recommender Systems: We&#8217;re doing it (all) wrong'
date: '2011-04-07T20:59:00+00:00'
author: 'Xavier Amatriain'
##layout: post
permalink: /recommender-systems-were-doing-it-all/
image: /wp-content/uploads/2011/04/fig2b.jpg
categories:
    - Uncategorized
tags:
    - error
    - 'recommender systems'
    - stats
---

A few days back, there was an interesting post by Judy Robertson in the Communications of the ACM blog. The post, entitled “[Stats: We’re doing it wrong](http://cacm.acm.org/blogs/blog-cacm/107125-stats-were-doing-it-wrong/fulltext)“, builds upon a paper from last year’s CHI conference in which they report that more than 90% of the HCI researchers used the wrong statistical tools when analyzing and reporting on likert scale type of data. A Likert scale is a unidimensional scale on which the respondent expresses the level of agreement to a statement – typically in a 1 to 5 scale in which 1 is strongly disagree and 5 is strongly agree.

Here is an excerpt from the post that I think is worth highlighting:

> Likert scales give ordinal data. That it (sic), the data is ranked “strongly agree” is usually better than “agree.” However, it’s not interval data. You can’t say the distances between “strongly agree” and “agree” would be the same as “neutral” and “disagree,” for example. People tend to think there is a bigger difference between items at the extremes of the scale than in the middle (there is some evidence cited in Kaptein’s paper that this is the case). **For ordinal data, one should use non-parametric statistical tests** which do not assume a normal distribution of the data. **Furthermore, because of this it makes no sense to report means of likert scale data–you should report the mode**.

As Judy, I have to admit that I am not a stats expert myself either. But in the general case I would agree with the previous: likert scale data is ordinal and cannot be treated as interval. However, whether treating it as interval is <span style="font-weight: bold;">always</span> a mistake or can be accepted under some circumstances is something that I am not sure and relates to the rest of this post.

So for instance, it is not uncommon to find references where they clearly state that likert data can be treated as interval. For example, look at what they say in [this handbook](http://www.fao.org/docrep/W3241E/w3241e04.htm) edited by the FAO.

> Likert scales are treated as yielding Interval data by the majority of marketing researchers.

Or look at [the answer](http://stats.stackexchange.com/questions/10/under-what-conditions-should-likert-scales-be-used-as-ordinal-or-interval-data) to the question of whether likert data can be treated as interval in stackexchange.

So there might be some circumstances in which, depending on the analysis, likert could be treated as interval… I guess. But not in the general case.

<span style="font-weight: bold;">Implications for Recommender Systems</span>

Now onto the big question: What does this have to do with Recommender Systems and how does it affect? To start with, let me ask you the question: Does the likert (1 to 5) scale relate to anything we use in recommender systems? You got it: <span style="font-weight: bold;">ratings</span> !

So our worry goes now to understanding whether ratings can be treated as interval or they should instead be treated as ordinal data, just as they are in the general case of the likert scale. In order to defend that ratings can be treated as interval, we should have some validation that the distance between different ratings is approximately equal. However, just as in the case of likert scales, we know this is not the case.

Look at this figure from [our previous work](http://localhost:8080/wordpress/2009/04/i-like-it-i-like-it-not-or-how-miss.html) on measuring noise in ratings.  
![](file:///home/xavier/Sandbox/data/articles/xamat_UMAP09/figs/fig2b.jpg)[![](http://localhost:8080/wordpress/wp-content/uploads/2011/04/fig2b.jpg)](http://4.bp.blogspot.com/-0BzBRalrIDo/TZ4xEitd5EI/AAAAAAAAAKs/1l2M5Og6dEo/s1600/fig2b.jpg)

Here we are plotting the probability of finding different kinds of inconsistencies between pairs of ratings. The probability that a user changes her rating between 2 and 3 is almost 0.35 while the probability she changes between 4 and 5 goes down to almost 0.1. This is a clear indication that users perceive that the distance between a 2 and a 3 is much lower than between a 4 and a 5.

<span style="font-weight: bold;">Consequences</span>

At this point, we can safely say that ratings are ordinal but not interval data. However, they are treated as a continuous interval scale in most of the recommender systems research! Let us stop to think a few of the consequences of ratings not being interval data.

<span style="font-weight: bold;">Distance Measures:</span> All the neighbor based methods in collaborative filtering are based on the use of some sort of distance measure. The most commonly used are Cosine distance and Pearson Correlation. However, both these distances assume a linear interval scale in their computations! We should conclude that using these distance measures with rating data is wrong. Other measures such as [Spearman’s rank correlation](http://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient), do not assume this. But to be honest, I don’t remember having read many papers using Spearman.

<span style="font-weight: bold;">Error Measures:</span> This is my favorite one… The most commonly accepted measure of success for recommender systems is the Root Mean Squared Error (RMSE). But wait, this measure is explicitly assuming that ratings are also interval data! Similar error measures such as MAE also fall in the same trap… banned! So what could we use? Standard Information Retrieval measures such as Precision and Recall do not necessarily assume interval scale on the ratings, although their mapping to recommendation efficiency may also be questioned. Rank-based measures such as [Discounted Cumulative Gain](http://en.wikipedia.org/wiki/Discounted_cumulative_gain) (nDCG) seem like our best bet for now.

<span style="font-weight: bold;">Matrix factorization</span>: Most MF techniques in Recommender Systems are in fact optimizing for RMSE. Therefore, we should discard them as statistically incorrect for the same reasons stated above. There are interesting alternatives to this though, like the [PureSVD](http://research.yahoo.com/files/recsys2010_submission_150.pdf) method presented in Recsys last year, that do not optimize for RMSE but rather for ranking.

<span style="font-weight: bold;">Conclusion</span>

It is clear that explicit ratings, just like likert scale data, have to be treated like ordinal (and not interval data). However, most of the methods and measures currently in use in recommender systems assume in some sense that there is a continuous linear scale in the ratings. Of course I am not advocating for throwing all of this research to the trash (among other things, it would include much of mine), but I would advice for a drastic change in the way we approach these issues.

I am writing this post, especially in the hope to get feedback and reactions from you. So I am looking forward to the comments.

<span style="color: rgb(204, 0, 0); font-weight: bold;">Update:</span> This post was featured in Ycombinator Hacker News. So far it has received over 6K views and there is a somewhat interesting [comment thread](http://news.ycombinator.com/item?id=2423313) in Ycombinator.

(I’d like to thank and acknowledge the contribution of [Denis Parra](http://www.sis.pitt.edu/%7Edparra/), [Alexandros Karatzoglou](http://www.ci.tuwien.ac.at/%7Ealexis/Welcome.html), and [Rodrigo Oliveira](http://www.ic.unicamp.br/%7Eoliveira/) to this post through previous very fruitful discussions)
