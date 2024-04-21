---
id: 75
title: 'Recsys 2012: A long (and likely biased) summary'
date: '2012-09-17T17:47:00+00:00'
author: 'Xavier Amatriain'
##layout: post
permalink: /recsys-2012-long-and-likely-biased/
header:
  teaser: /images/RecsysPhoto.jpg
categories:
    - Uncategorized
---

After a great week in beautiful and sunny Dublin (yes, sunny), it is time to look back and recap on the most interesting things that happened in the 2012 Recsys Conference. I have been attending the conference since its first edition in Minnesota. And, it has been great to see the conference mature to become the premiere event for recommendation technologies. I can’t hide that this is my favorite conference for several reasons: perfect size, great community, good involvement from industry, and good side program of tutorials, workshops, and demos.

| ![](/blog/images/RecsysPhoto.jpg) |

This year I arrived a bit late and missed the first day of tutorials, and first day of the conference. But, was able to catch up after jumping right in with my 90 minute tutorial on “Building Industrial-scale Real-world Recommender Systems”

In my tutorial ([see slides here](http://www.slideshare.net/xamat/building-largescale-realworld-recommender-systems-recsys2012-tutorial)), I talked about the importance of four different issues in real-world recommender systems:

- Paying attention to user interaction models that support things like explanations, diversity, or novelty.
- Coming up with algorithms that, beyond rating prediction, focus on other aspects of recommendation such as similarity, or, in particular, ranking.
- Using results of online A/B tests, and coming up with offline model metrics that correlate with the former.
- Understanding the software architectures where your recommender system will be deployed.

I was happy to see that some of these issues not only were mentioned, but almost became conducting threads throughout the conference. Of course, this might be in the eye of the beholder, and others might have come back with the impression that the main topics were others (I recommend you read these two other Recsys 2012 summaries by [Daniel Tunkelang](http://thenoisychannel.com/2012/09/14/recsys-2012-beyond-five-stars/) and [Daniele Quercia](http://www.syslog.cl.cam.ac.uk/2012/09/14/recsys-2012-few-things-i-remember/)). In any case, grouping in topics will help me summarize the many things I found interesting.

### Online A/B Testing and offline metrics

I am glad to see that this has become a relevant topic for the conference, because many of us believe this is one of the most important topics that need to be addressed by both industry and academia. One of these people is Ron Kohavi, who delivered a great keynote on “[Online Controlled Experiments: Introduction, Learnings, and Humbling Statistics](http://www.exp-platform.com/Pages/2012RecSys.aspx)“, where he described his learnings of many years of AB Testing in Amazon and Microsoft. It is funny that I cited his KDD 2012 paper in two slides in my tutorial, not knowing that he was in the audience. I recommend you go through his slides, it was one of the best talks of the conference for sure.

The importance of finding relevant metrics was, as a matter of fact, the focus of a workshop we organized with Harald Steck (Netflix), Pablo Castells (UAM), Arjen de Vries, and Christian Posse (LikedIn). The title of the workshop was “Recommendation Utility Evaluation: Beyond RMSE”. Unfortunately, I was not able to attend. But, I do know the keynote by Carlos Gomez-Uribe, also from Netflix, was very well received. And, the workshop as a whole went very well with several interesting papers and even more interesting discussions. You can access the papers on the [website](http://ir.ii.uam.es/rue2012/).

A couple of papers in the main track of the conference also touched upon the importance of optimizing several objectives at the same time. In “[Multiple Objective Optimization in Recommender Systems](http://www.slideshare.net/mechanistician/recsys-2012-slides?ref=http://thenoisychannel.com/)“, Mario Rodriguez and others explain how they design LinkedIn recommendations by optimizing to several objectives at once (e.g. candidate that is good for the job + who is open to new opportunities). They report results from an AB Test run on LinkedIn. In “[Pareto-Efficient Hybridization for Multi-Objective Recommender Systems](http://www.slideshare.net/marcotulioribeiro54/presentation-recsys12)“, Marco Tulio Ribeiro and others from Universidade Federal de Minas Gerais &amp; Zunnit Technologies take the multi-objective a step further. In their case, they optimize the system to not only be accurate, but also present novel or diverse items.

Some other papers went beyond the academic experimental procedure and implemented real systems that were tested with users. A good example is “Finding a Needle in a Haystack of Reviews: Cold Start Context-Based Hotel Recommender System” by researchers from the Tel Aviv Yaffo College and Technicolor.

### Learning to Rank

Another hot topic in this year’s recsys was ranking (or Top-n Recommendations as some prefer to call it). It is good to see that after some time publicly speaking about the importance of ranking approaches, the community seems now to be much more focused on ranking than on rating prediction. Not only there was a whole session devoted to ranking, but actually many other papers in the conference dealt with the topic in some way or another.

I will start by mentioning the very good work by my former colleagues from Telefonica. Their paper “CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering” won the best-paper award. And, I think most of us thought that it was very well-deserved. It is a very good piece of work. Well motivated, evaluated, and, it addresses a very practical issue. It is great to see the Recsys team at Telefonica that I started be acknowledged with this award. You can access the paper [here](http://baltrunas.info/papers/Shi12-climf.pdf) and the slides [here](http://www.slideshare.net/kerveros99/climf-collaborative-lessismore-filtering).

In that same session, researchers from the Université Paris 6 presented “Ranking with Non-Random Missing Ratings: Influence of Popularity and Positivity on Evaluation Metrics”, an interesting study on the very important issue of negative sampling, and popularity bias in learning to rank. The paper discusses these effects on the AUC (Area Under the Curve) measure, a measure that is not very well-behaved, nor very much used in evaluating ranking algorithms. Still, it is a valuable first step in a very interesting line of work. It is interesting to point out that the CLiMF paper addressed the issue of negative sampling in a radically different way: only considering positive samples. Yet another interesting paper in that session was “[Sparse Linear Methods with Side Information for Top-N Recommendations](http://www-users.cs.umn.edu/%7Exning/papers/Ning2012.pdf)“, a model for multidimensional context-aware learning to rank.

Another ranking paper, “Alternating Least Squares for Personalized Ranking” by Gábor Takács from Széchenyi István University and Domonkos Tikk from Gravity R&amp;D, received an honorable mention. The main author coined an (un)popular sentence during his presentation when he invited anyone not interested in Mathematics to leave the room. An unnecessary invitation in a conference that prides itself for being inclusively multidisciplinary. In Recsys, psychologists are seating through systems presentations as much as mathematicians are seating through user-centric sessions, and that is what makes the conference appealing. In any case, the paper presents an interesting way to combines a ranking-based objective function introduced in last year’s kdd and the use of ALS instead of SGD to come up with another approach to learning to rank.

Two papers dealing with recommendations in Social Networks also focused on ranking. “[On Top-k Recommendation Using Social Networks](http://eeweb.poly.edu/faculty/yongliu/docs/topk_tr.pdf)” by researchers from NYU and Bell Labs, and “[Real-Time Top-N Recommendation in Social Streams](http://www.l3s.de/web/upload/documents/1/diaz_recsys2012.pdf)” by Ernesto Diaz-Aviles and other researchers from the University of Hannover. The same first author had an interesting short paper in the poster session: “[Swarming to Rank for Recommender System](http://www.l3s.de/web/upload/documents/1/sp197-diaz.pdf)“. In that poster he proposes the use of a Particle Swarm Optimization algorithm to directly optimize ranking metrics such as MAP. The method proposes an interesting alternative to the use of Genetic Algorithms or Simulated Annealing for this purpose.

Finally, the industry keynote by Ralf Herbrich from Facebook, also introduced the world of Bayesian Factor Models for large-scale distributed ranking. This method, introduced by the same author and others from MSR as “[Matchbox](http://www2009.eprints.org/12/1/p111.pdf)” is now used in different settings. For example, the poster “[The Xbox Recommendation System](http://www.eng.tau.ac.il/%7Enoamk/papers/KNPS12.pdf)” presented its applicability for recommending movies and games for the Xbox. And, in “[Collaborative Learning of Preference Rankings](http://research.microsoft.com/pubs/166623/RecSys-2012.pd)” the authors apply it to… sushi recommendation!

### User-centric, interfaces &amp; explanations

This was probably the third big area of focus of the conference, with many contributions in papers, tutorials, and workshops. The first day, there were actually two tutorials that would fall into this category. In “Personality-based Recommender Systems: An Overview”, the authors presented the idea of using personality traits for modeling user profiles. Among other things, they introduced their proposal to use PersonalityML, an XML-based language for personality description. Interestingly, in the industry session, we saw that this is actually a quite practical thing to do. Thore Graepel from Microsoft explained their experiments in using The Big Five personality traits for personalization. In the other tutorial, “[Conducting User Experiments in Recommender Systems](http://www.slideshare.net/usabart/tutorial-on-conducting-user-experiments-in-recommender-systems)“, Bart Knijnenburg gave a thorough overview of how to conduct user studies for recommender systems. He also introduced his model for using structural equations to model the effects to evaluate. Again, I missed this tutorial, but I was fortunate to hear a very similar presentation by him in Netflix.

In “[Inspectability and Control in Social Recommenders](http://bostandjiev.com/Content/publications/SocialRecStudy.pdf)“, Bart himself (and researchers from UCSB) analyze the effect of giving more information and control to users in the context of social recommendations. A similar idea is explored in the short paper “The Influence of Knowledgeable Explanations on Users’ Perception of a Recommender System” by Markus Zanker.

Two papers addressed the issue of how much information we should require from users. In “User Effort vs. Accuracy in Rating-Bbased Elicitation” Paolo Cremonesi and others analyze how many ratings “are enough” for producing satisfying recommendations in a cold-start setting. And, in “How Many Bits Per Rating?”, the Movielens crew try to quantify the amount of information and noise in user ratings from an information-theoretical perspecive. An interesting continuation to [my work on user ratings noise](http://localhost:8080/wordpress/2009/04/i-like-it-i-like-it-not-or-how-miss.html). However, as the first author himself admited, this is just initial work.

Other highlights of user-centric work that fell more on the UI side were the paper “TasteWeights: A Visual Interactive Hybrid Recommender System” by my friends at UCSB, as well as the many papers presented in the [Workshop on Interfaces for Recommender System](https://homepages.abdn.ac.uk/csc284/pages/InterfaceRS/).

### Data &amp; Machine Learning Challenges

If somebody thought that data and machine learning challenges would fade away after the Netflix Prize, this year’s Recys was a clear example that this is far from being the case. Many challenges have taken over after that: the yearly KDD Cups, Kagel, Overstock, last year’s MoviePilot challenge, the Mendeley Challenge… Recsys had this year a [Tutorial/Panel](http://recsys.acm.org/2012/tutorials.html#best) and a [Workshop](http://2012.recsyschallenge.com/) on Recommender Systems Challenges, both organized by Alan Said, Domonkos Tikk, and others. I could not attend the Tutorial since it was happening at the same time than mine. But, I was able to catch some interesting presentations in the Workshop. Domonkos Tikk from Gravity R&amp;D gave [a very interesting presentation](http://www.slideshare.net/domonkostikk/from-a-toolkit-of-recommendation-algorithms-into-a-real-business-the-gravity-rd-experience) on how they evolved from being a team in the Netflix Prize to a real-world company with very interesting projects. Kris Jack from Mendeley also gave two interesting talks on the Mendeley recommender systems. In [one of them](http://www.slideshare.net/KrisJack/mendeley-suggest-engineering-a-personalised-article-recommender-system), he explained how they make use of AWS and Mahout in a system that can generate personalized recommendations for about $60 a month. In [the other](http://www.slideshare.net/KrisJack/rec-sys12mendeleydatachallenges), he talked about their perspective on data challenges.

### Context-aware and location-based recommendations

This has become a traditional area of interest in Recsys. It has now matured to a point that it has it own session, and two workshops: “[Personalizing the Local Mobile Experience](http://loca.mobilelifecentre.org/)“, and the “[Workshop on Context-Aware Recommender Systems](http://cars-workshop.org/)“. But, besides having its own session in the conference, several other papers in others also deal with context-aware recommendations. I have already mentioned “Sparse Linear Methods with Side Information for Top-N Recommendations”, for example. Other interesting papers in this area were “Context-Aware Music Recommendation Based on Latent Topic Sequential Patterns”, on the issue of playlist generation, and “[Ads and the City: Considering Geographic Distance Goes a Long Way](http://www.cl.cam.ac.uk/%7Edq209/publications/trumper12ads.pdf)” for location-aware recommendations.

### Social

A similar area that has already matured over several Recsys is Social. It has its own session, and Workshop, “[Workshop on Recommender Systems and the Social Web](http://ls13-www.cs.uni-dortmund.de/homepage/rsweb2012/index.shtml)” , and trancends over many other papers. In this area, the paper that I have not mentioned in other categories and found interesting was “[Spotting Trends: The Wisdom of the Few](http://www.slideshare.net/daniele.quercia/recsys-14297597)“. One of the reasons I found the paper interesting is because it builds on our idea of using a reduced set of experts for recommendations, what we called “[The Wisdom of the Few](http://localhost:8080/wordpress/2009/05/wisdom-of-few.html)“.

### Others

And yes, I still have some interesting stuff from the poster session that I could not fit into any of the above categories.

First, the short paper “[Using Graph Partitioning Techniques for Neighbour Selection in User-Based Collaborative Filtering](http://ir.ii.uam.es/%7Ealejandro/2012/recsys.pdf)” by Alejandro Bellogin. Alejandro won the Best Short Paper Award, for a great piece of work and presentation. He described an approach to use the Normalized Cut graph clustering approach for grouping similar users, and improve neighborhood formation in standard kNN Collaborative Filtering.

I also liked the poster “[Local Learning of Item Dissimilarity Using Content and Link Structure](http://cse.iitkgp.ac.in/%7Epabitra/paper/recsys12.pdf)“, another graph-based approach, in this case to learn a similarity function.

Finally, in “When Recommenders Fail: Predicting Recommender Failure for Algorithm Selection and Combination”, Michael Ekstrand starts to tap into an extremely important question: when and why do some recommendation algorithms fail? This question has been informally discussed in the context of hybrid recommenders and ensembles. But, there is clearly much more work to do, and many things to understand.

———————-

Well, if you made it all the way to here, it means that you are really interested in Recommender Systems. So, chances are that I will be seeing you in next year’s Recsys. Hope to see you in Hong Kong!
