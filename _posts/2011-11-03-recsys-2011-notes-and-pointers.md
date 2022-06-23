---
id: 81
title: 'Recsys 2011 &#8211; Notes and Pointers'
date: '2011-11-03T04:29:00+00:00'
author: 'Xavier Amatriain'
##layout: post
permalink: /recsys-2011-notes-and-pointers/
categories:
    - Uncategorized
tags:
    - conference
    - 'recommender systems'
    - recsys11
---

[![](http://recsys.acm.org/2011/images/Chicago_night1.jpg)](http://recsys.acm.org/2011/images/Chicago_night1.jpg)  
I found [Recsys](http://recsys.acm.org/2011/index.shtml) this year of very high quality in general. There were many good papers and presentations. The [Industry track](http://recsys.acm.org/2011/industry_track.shtml) was also very high-quality, with very interesting talks from companies such as Twitter, Facebook, or eBay. Jon Sanders and I also gave two presentations explaining how recommendations have evolved since the Netflix Prize (more on this soon).

Here are my rough notes with pointers to some papers I considered especially interesting. I have grouped them in 5 categories that I think summarize the main topics in the conference: (1) Transparency and explanations, (2) Implicit Feedback, (3) Context, (4) Metrics and evaluation, and (5) Others. Note that the selection is completely biased towards my personal interests.

<span style="font-weight: bold;">(1) TRANSPARENCY &amp; EXPLANATIONS.</span> One of the recurring themes was the fact that user trust and perceived quality of the recommendations was very much influence not by accuracy alone, but by how transparent the system was, and the amount of “explanations” that were added.

- Daniel Tunkelang(LinkedIn) did a very interesting tutorial on “Recommendations as a Conversation with the User”, where he focused on these kinds of issues. See his slides in [his blog](http://thenoisychannel.com/2011/10/31/recsys-2011-tutorial-recommendations-as-a-conversation-with-the-user/).
- Neel Sundaresan (eBay) also stressed in his keynote that adding explanations can sometimes be more important than getting the recommendation right.
- In the paper “[Each to His Own: How Different Users Call for Different Interaction Methods in Recommender Systems](http://www.usabart.nl/portfolio/KnijnenburgReijmerWillemsen-recsys2011.pdf)“, the authors found that depending on how experts are users in the domain, they prefer different kind of recommendations and interaction models. For example, in one of the extremes, novices, prefer top-10 non-personalized to their personalized recommendations. In general a hybrid model of interaction is better than either implicit or explicit-only.

<span style="font-weight: bold;">(2) IMPLICIT FEEDBACK.</span> A lot of papers this years on using implicit consumption data instead of (or in combination with) ratings.

- The best paper, by Yehuda Koren and Joe Sill, addressed the issue of non-linearity in ratings. “[OrdRec: An Ordinal Model for Predicting Personalized Item Rating Distributions](http://labs.yahoo.com/node/640)” modifies the standard Matrix Factorization approach to adapt to the fact that user ratings are ordinal, but not numerical. The way they model ratings, with a set of thresholds, can be used in combination with any model, not only SVD-like approaches. This paper effectively addresses most of the issues I raised in my previous post “[We are doing everything wrong…](http://localhost:8080/wordpress/2011/04/recommender-systems-were-doing-it-all.html)“
- In “[Modeling Item Selection and Relevance for Accurate Recommendations: A Bayesian Approach](http://unical.academia.edu/NicolaBarbieri/Papers/803078/Modeling_Item_Selection_and_Relevance_for_Accurate_Recommendations)” they define the concept of a “Free probabilstic model” where they try to predict independently the probabilty of play and rating.
- In “Multi-Value Probabilistic Matrix Factorization for IP-TV Recommendations”, the authors present a Matrix Factorization model that allows for multiple observations of the same item. In particular, it is applied for IPTV recommendations where the fact that the user watched part of an episode is interpreted as negative feedback.
- “[Matrix Co-factorization for Recommendation with Rich Side Information and Implicit Feedback](http://www.cs.purdue.edu/homes/fangy/hetrec11-fang.pdf)” presents a combined Matrix Factorization model that includes ratings, content features, and implicit feedback. They use cosine item similarity for weighing negative examples.
- In “[Personalizing Tags: A Folksonomy-like Approach for Recommending Movies](http://www.slideshare.net/alansaid/personalizing-tags-a-folksonomylike-approach-for-recommending-movies/download)“, they use tags (or categories) as a very simple method of recommending movies: for each user compute average rating given to movies with a certain tag.

<span style="font-weight: bold;">(3) CONTEXT.</span> There were 2 workshops ([CARS](http://cars-workshop.org/) and [CAMRA](http://2011.camrachallenge.com/)), and several papers in the main conference, talking about how to add contextual information for the recommendations:

- “The Effect of Context-Aware Recommendations on Customer Purchasing Behavior and Trust” is an interesting paper, focusing on the evaluation side. They include an A/B test for measuring the effect of context-aware recommendations. Using context increased overall sales in $ but not in number. Therefore, users tend to spend more $ per item.
- In the [CAMRA](http://2011.camrachallenge.com/) workshop, many papers (such as “Temporal Rating Habits: A Valuable Tool for Rater Differentiation” or “Identifying Users From Their Rating Patterns”) were related to how to identify who the author of a rating in a household was, since this was one of the tasks for the contest.
- Also related to group recommendations, “Group Recommendation using Feature Space Representing Behavioral Tendency and Power Balance among Members”, tries to model what is a good recommendation for a group where each of the individuals does not have the same influence.

<span style="font-weight: bold;">(4) METRICS and EVALUATIONS: </span>There were several papers that offered different ways to measure accuracy for top-N ranked recommendations.

- “[Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems](http://www.slideshare.net/pcastells/acm-recsys-2011-rank-and-relevance-in-novelty-and-diversity-metrics-for-recommender-systems)” presents an interesting framework that includes metric for measuring not only accuracy, but also novelty, diversity….
- “Item Popularity and Recommendation Accuracy” is an interesting work on how to remove popularity bias from accuracy metrics. A user study validates the fact that recall measure is correlated with user perceived quality of recommendation. Besides proposing a recall metric that removes popularity bias, he also proposes a popularity stratified training method that weights negative examples according to how popular they are.
- “[Evaluating Rank Accuracy based on Incomplete Pairwise Preferences](http://ucersti.ieis.tue.nl/files/papers/3.pdf)” proposes a measure called expected discounted rank correlation for the specific case of implicit feedback.

<span style="font-weight: bold;">(5) OTHERS</span>

- eBay and UCSC presented “[Utilizing Related Products for Post-Purchase Recommendation in E-commerce](http://users.soe.ucsc.edu/%7Ejwang30/index.files/recsys175-wang.pdf)“. The paper won the best poster award
- There were many papers on Social Recommendations. Just to name one, in “Power to the People: Exploring Neighbourhood Formations in Social Recommender Systems”, they did a user study to figure out how much users would like and trust recommendations coming from different user groups (those they decided, friends, everyone…). Interestingly, the method of choice did not make much difference… until you told the users what it was.
- In “Wisdom of the Better Few: Cold Start Recommendation via Representative based Rating Elicitation” they discussed how to select most imformative users and items for cold start. I was surprised to see that our “Wisdom of the Few” approach got paraphrased in a paper title.
- There were a couple of very interesting workshops on [Music Recommendations](http://womrad.org/2011/) and [Mobile Recommendations](http://pema2011.cs.ucl.ac.uk/) that I had to miss since I was attending others. But, they are definitely worth looking into if you are into music or mobile.