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




- [Introduction to recommender systems: Basics and classic techniques](#Intro)
- [Beyond the basics](#Beyond)
    - [Ranking](#ranking)
    - [Factorization machines](#fm)
    - [Explore/exploit](#explorerexploit)
    - [Full page optimization](#page)
    - [Context-aware recommendations and other approaches](#context)
    - [Reinforcement learning](#RL)
- [Deep Learning for recommendations](#DL)
    - [The Deep Basics](#DeepBasics)
    - [Embeddings](#embeddings)
    - [Graph Neural Networks](#gnn)
    - [Recommending Sequences](#sequences)
    - [LLMs for recommendations](#llms)
- [The “systems part” of recommender systems](#systems)
- [Evaluation and UX](#ux)
- [End-to-end examples of real-world industrial recommender systems](#e2e)


### <a name="Intro"></a> 1. Introduction to recommender systems: Basics and classic techniques 


[Introduction to Recommender Systems: A 4-hour lecture [VIDEO]](https://amatriain.net/blog/introduction-to-recommender-systems-4/)

[Data Mining Methods for Recommender Systems](https://amatriain.net/pubs/RecsysHandbookChapter.pdf)

[The recommender revolution](https://www.technologyreview.com/2022/04/27/1048517/the-recommender-revolution/)

<img src="/blog/images/101-1.png">

[On the “Usefulness” of the Netflix Prize](https://amatriain.net/blog/on-the-usefulness-of-the-netflix-prize-403d360aaf2/)

<img src="/blog/images/101-2.png">

[Kdd 2014 Tutorial - the recommender problem revisited](https://www.slideshare.net/xamat/kdd-2014-tutorial-the-recommender-problem-revisited)

<img src="/blog/images/101-3.png">

[Feature Engineering for Recommendation Systems -- Part 1](https://blog.fennel.ai/p/feature-engineering-for-recommendation)

### <a name="Beyond"></a> 2. Beyond the basics

#### <a name="ranking"></a>2.1 Ranking

[What is Learning To Rank?](https://opensourceconnections.com/blog/2017/02/24/what-is-learning-to-rank/)

<img src="/blog/images/101-4.png">

[Personalized ‘Complete the Look’ model by Walmart](https://medium.com/walmartglobaltech/personalized-complete-the-look-model-ea093aba0b73)

[Lamdbamart In depth](https://softwaredoug.com/blog/2022/01/17/lambdamart-in-depth.html)

#### <a name="fm"></a>2.2 Factorization machines

<img src="/blog/images/101-5.png">

[Factorization Machines for Item Recommendation with Implicit Feedback Data](https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db)

[Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

#### <a name="explorerexploit"></a>2.3 Explore/exploit

[Bandits for Recommender Systems](https://eugeneyan.com/writing/bandits/)

[Explore/Exploit Schemes for Web Content Optimization](https://sci-hub.se/10.1109/icdm.2009.52)

[Explore/Exploit for Personalized Recommendation [VIDEO]](https://www.youtube.com/watch?v=LvcoPy0QUuw&list=PLZSO_6-bSqHQCIYxE3ycGLXHMjK3XV7Iz)

<img src="/blog/images/101-6.png">

[Artwork Personalization at Netflix by Netflix](https://netflixtechblog.com/artwork-personalization-c589f074ad76)

[A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/pdf/1003.0146.pdf)

[Recommending Items to Users: An Explore Exploit Perspective](https://www.ueo-workshop.com/wp-content/uploads/2013/10/UEO-Deepak.pdf)

#### <a name="page"></a>2.4 Full page optimization

<img src="/blog/images/101-7.png">

[Learning a Personalized Homepage by Netflix](https://netflixtechblog.com/learning-a-personalized-homepage-aa8ec670359a)

<img src="/blog/images/101-8.png">

[Beyond Ranking: Optimizing Whole-Page Presentation [VIDEO]](https://www.youtube.com/watch?v=1LGJmFadtoI)

[Fair and Balanced: Learning to Present News Stories](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=07b73ac3e6881865e518b72cff6d82ea08456241)

#### <a name="context"></a>2.5 Context-aware recommendations and other approaches

[The Wisdom of the Few: A Collaborative Filtering Approach Based on Expert Opinions from the Web](https://amatriain.net/pubs/xamatriain_sigir09.pdf)

<img src="/blog/images/101-9.png">

[Multiverse Recommendation: N-dimensional Tensor Factorization for Context-aware Collaborative Filtering](https://amatriain.net/pubs/karatzoglu-recsys-2010.pdf)

[Temporal Diversity in Recommender Systems](https://amatriain.net/pubs/karatzoglu-recsys-2010.pdf)

[Towards Time-Dependant Recommendation based on Implicit Feedback](https://amatriain.net/pubs/karatzoglu-recsys-2010.pdf)

#### <a name="RL"></a>2.6 Reinforcement learning

[Reinforcement Learning for Recommendations and Search](https://eugeneyan.com/writing/reinforcement-learning-for-recsys-and-search/)

<img src="/blog/images/101-10.png">

[Deep Reinforcement Learning for Page-wise Recommendations](https://zhaoxyai.github.io/paper/recsys2018.pdf)

### <a name="DL"></a>3. Deep Learning for recommendations

#### <a name="DeepBasics"></a>3.1 The Deep Basics

<img src="/blog/images/101-11.png">

[Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)

<img src="/blog/images/101-12.png">

[Wide & Deep Learning for Recommender Systems by Google](https://arxiv.org/abs/1606.07792)
[
<img src="/blog/images/101-13.png">

[Deep Learning Recommendation Model for Personalization and Recommendation Systems by Facebook](https://arxiv.org/abs/1906.00091)

#### <a name="embeddings"></a>3.2 Embeddings

[Embedding-based Retrieval in Facebook Search by Facebook](https://arxiv.org/pdf/2006.11632.pdf)

<img src="/blog/images/101-14.png">

[Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba by Alibaba](https://arxiv.org/pdf/1803.02349.pdf)

#### <a name="gnn"></a>3.3 Graph Neural Networks

<img src="/blog/images/101-15.png">

[ATBRG: Adaptive Target-Behavior Relational Graph Network for Effective Recommendation](https://arxiv.org/pdf/2005.12002.pdf)

<img src="/blog/images/101-16.png">

[Using graph neural networks to recommend related products by Amazon](https://towardsdatascience.com/modern-recommendation-systems-with-neural-networks-3cc06a6ded2c)

[Modern Recommendation Systems with Neural Networks](https://towardsdatascience.com/modern-recommendation-systems-with-neural-networks-3cc06a6ded2c)

[Graph Neural Networks in Recommender Systems: A Survey](https://arxiv.org/abs/2011.02260)


#### <a name="sequences"></a>3.4 Recommending Sequences

[Behavior Sequence Transformer for E-commerce Recommendation in Alibaba by Alibaba](https://arxiv.org/pdf/1905.06874.pdf)

[Sequential Recommender Systems: Challenges, Progress and Prospects](https://www.ijcai.org/Proceedings/2019/0883.pdf)

[Recommending movies: retrieval using a sequential model a Tensorflow example](https://www.tensorflow.org/recommenders/examples/sequential_retrieval)


#### <a name="llms"></a>3.5 LLMs for recommendations

### <a name="systems"></a>4. The “systems part” of recommender systems

<img src="/blog/images/101-17.png">

[Recommender Systems, Not Just Recommender Models](https://medium.com/nvidia-merlin/recommender-systems-not-just-recommender-models-485c161c755e)

[Real World Recommendation System - Part 1 by Fennel.ai](https://blog.fennel.ai/p/real-world-recommendation-system)

<img src="/blog/images/101-18.png">

[Blueprints for recommender system architectures: 10th anniversary edition](https://amatriain.net/blog/RecsysArchitectures)

<img src="/blog/images/101-19.png">

[Pinterest Home Feed Unified Lightweight Scoring: A Two-tower Approach by Pinterest](https://medium.com/pinterest-engineering/pinterest-home-feed-unified-lightweight-scoring-a-two-tower-approach-b3143ac70b55)

[How NVidia supports Recommender Systems [VIDEO] by NVidia](https://www.youtube.com/watch?v=wPso35VkuCs)

<img src="/blog/images/101-20.png">

[System Design for Recommendations and Search](https://eugeneyan.com/writing/system-design-for-discovery/)

[Real-time Machine Learning For Recommendations](https://eugeneyan.com/writing/real-time-recommendations/#when-not-to-use-real-time-recommendations)

<img src="/blog/images/101-21.png">

[Near real-time features for near real-time personalization by LinkedIn](https://engineering.linkedin.com/blog/2022/near-real-time-features-for-near-real-time-personalization)

<img src="/blog/images/101-22.png">

[Introducing DreamShard: A reinforcement learning approach for embedding table sharding by Facebook](https://research.facebook.com/blog/2022/12/introducing-dreamshard-a-reinforcement-learning-approach-for-embedding-table-sharding/)

### <a name="UX"></a>5. Evaluation and UX

<img src="/blog/images/101-23.png">

[The death of the stars: A brief primer on online user ratings](https://amatriain.net/blog/the-death-of-the-stars-a-brief-primer-on-online-user-ratings-6740453f27ed/)

[EvalRS: a Rounded Evaluation of Recommender Systems](https://arxiv.org/pdf/2207.05772.pdf)

<img src="/blog/images/101-24.png">

[Beyond NDCG: behavioral testing of recommender systems with RecList](https://arxiv.org/abs/2111.09963)

<img src="/blog/images/101-25.png">

[How to Measure and Mitigate Position Bias](https://eugeneyan.com/writing/position-bias/)

[Rate it Again: Increasing Recommendation Accuracy by User re-Rating](https://amatriain.net/pubs/xamatriain_Recsys09.pdf)

[I like It... I like It Not: Measuring Users Ratings Noise in Recommender Systems](https://amatriain.net/pubs/umap09.pdf)


### <a name="e2e"></a>6. End-to-end examples of real-world industrial recommender systems

[Lessons Learned from building real life recommender systems](https://amatriain.net/blog/ten-lessons-learned-from-building-real/)

[Past, present, and future of recommender systems: An industry perspective [VIDEO]](https://dl.acm.org/doi/abs/10.1145/2959100.2959144)

<img src="/blog/images/101-26.png">

[On YouTube’s recommendation system by Youtube](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/)

[Deep Neural Networks for YouTube Recommendations by Youtube](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190)

[How Spotify Uses ML to Create the Future of Personalization by Spotify [VIDEO]](https://engineering.atspotify.com/2021/12/how-spotify-uses-ml-to-create-the-future-of-personalization/)

[Recommender systems in industry: A Netflix case study by Netflix](https://amatriain.net/pubs/Recsys-in-industry.pdf)

[Twitter’s recommendation algorithm by Twitter](https://blog.twitter.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm)

<img src="/blog/images/101-27.png">

[Evolving the Best Sort for Reddit’s Home Feed by Reddit](https://www.reddit.com/r/blog/comments/o5tjcn/evolving_the_best_sort_for_reddits_home_feed/)

[Intelligent Customer Preference engine with real-time ML systems by Walmart [VIDEO]](https://www.tecton.ai/apply/session-video-archive/intelligent-customer-preference-engine-with-real-time-ml-systems-2/)

<img src="/blog/images/101-28.png">

[Homepage Recommendation with Exploitation and Exploration by Doordash](https://doordash.engineering/2022/10/05/homepage-recommendation-with-exploitation-and-exploration/)

[Learning to rank restaurants by Swiggy](https://bytes.swiggy.com/learning-to-rank-restaurants-c6a69ba4b330)

<img src="/blog/images/101-29.png">

[How we use engagement-based embeddings to improve search and recommendation on Faire by Faire](https://craft.faire.com/how-we-use-engagement-based-embeddings-to-improve-search-and-recommendation-on-faire-912277de4e6d)

[Deep Recommender Systems at Facebook [VIDEO] by Facebook](https://www.youtube.com/live/5xcd0V9m6Xs?feature=share)

<img src="/blog/images/101-30.png">

[Building a heterogeneous social network recommendation system by LinkedIn](https://engineering.linkedin.com/blog/2020/building-a-heterogeneous-social-network-recommendation-system)

<img src="/blog/images/101-31.png">

[A closer look at the AI behind course recommendations on LinkedIn Learning by LinkedIn](https://engineering.linkedin.com/blog/2020/course-recommendations-ai-part-one)

<img src="/blog/images/101-32.png">

[Recommending the world's knowledge. Applications of Recommender Systems at Quora](https://www.slideshare.net/LeiYang27/recommending-the-worlds-knowledge)

[Monolith: Real Time Recommendation System With Collisionless Embedding Table by Bytedance/Tiktok](https://arxiv.org/abs/2209.07663)






