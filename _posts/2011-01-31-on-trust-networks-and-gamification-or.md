---
id: 92
title: 'On Trust Networks and Gamification. Or How Quora can overcome its Hype and embrace long-term Success'
date: '2011-01-31T21:26:00+00:00'
author: 'Xavier Amatriain'
##layout: post
permalink: /on-trust-networks-and-gamification-or/
header:
  teaser: /images/14-01.png
categories:
    - Uncategorized
tags:
    - 'game dynamics'
    - quora
    - 'Social Networks'
    - trust
---

If you are reading this blog I am pretty sure that you know quite a lot about [Quora](http://www.quora.com/) by now. If not, you should sign on and try it a bit before you continue reading the post.

I have to admit it, the first time I saw Quora I thought it looked like a watered-down version of [stackoverflow](http://http//stackoverflow.com/) , only with a much broader scope. The ability to follow was nice but… “big deal”, I thought. However, I was missing the important point of the seamless integration between Quora and existing OSN, namely Twitter and Facebook. I always say that for an OSN to succeed it needs to ride on all the previous successful ones (including email if you allow me to stretch the definition of OSN that far), but I missed that part in quora until its hype began. Having quick connection to Twitter and Facebook, allowed Quora to overcome the always feared cold-start problem. You sign to Quora and in no time you are “connected” to all your “friends” and can start following their questions, their answers, votes… cool!

Well, so it seemed. But in no time, just as quick as people starting hyping about the service they were complaining about it and predicting its failure. This [recent post at Techcrunch](http://techcrunch.com/2011/01/31/quora-quora-quora-quora-quora-quora-quora/) does a pretty good at summarizing and linking to the main Quora bitchmemes. Don’t miss the[ original post by Vivek Wadhwa](http://techcrunch.com/2011/01/23/why-i-don%E2%80%99t-buy-the-quora-hype/) or some of the [threads](http://www.quora.com/What-can-be-said-to-Vivek-Wadhwas-criticism-on-TechCrunch-Why-I-Don%E2%80%99t-Buy-the-Quora-Hype) at Quora itself. You should also read [this very illustrative piece](http://scobleizer.com/2011/01/30/why-i-was-wrong-about-quora-as-a-blogging-service/) on how Scobble went from love to hate in a matter of weeks.

To summarize, the two biggest complaints are the following: (1) Quora will inevitably be overtaken by spam and there will be no way to find good content anymore; and (2) producers of good content (answers) will become tired of the system and progressively leave making problem (1) even more inevitable.

While I do agree that these (and many other) issues are very important, I don’t see them as inevitable and, in the following paragraphs, I would like to describe two ways to address them. But to start with, let me just state that believing that Quora can survive on being an inside-moderated network is not the answer. So what can be done?

<span style="font-weight: bold;font-size:130%;">Trust networks  
</span>  
In a trust network, nodes (users) have an associated trust value that is somehow used to decide how its contribution will be taken into account by the rest of users. For instance, in a recommender system, I can push content by neighboring nodes I trust while filtering out that coming from nodes with a lower trust value. In more sophisticated versions, trust is not a unique value but can be topic-specific. That is, my trust value can be very high for independent music but very low for classic literature. (If you are interested in the general topic of Trust and Social Networks you can read [Golbeck’s book](http://www.amazon.com/exec/obidos/ASIN/1848003552/j16t3i5j15-20) or any of her many publications or presentations available online)

So let’s go back to Quora now: why should they implement a trust network overlay? and, how could they implement a useful one? There are several reasons for why they should be doing so. But let us focus on the spam issue. You do not want for bad answers to get promoted by bad/evil users. The way around it is to not give these users the power to promote answers. And you can do this quite easily by assigning trust values to users. It would take 100 votes by “level 1” users to get an answer to the level of another one with just one vote by a “level 100” user. Of course, as I was mentioning before, this trust level could be topic-sensitive. Makes sense, doesn’t it?

But, there a number of issues that are still unsolved on how to implement this trust network. The first one is who decides to promote and demote users? My answer is quite simple: users themselves. Whenever your answer gets voted up/down so would your trust level. And again, how much this level would go up/down would depend on the trust level of the voting user.

The only important remaining issue to such an approach is how to deal with the cold-start issue. But the answer to this would come from the integration to other OSN I was mentioning at the beginning. If I were implementing this kind of system, I would give users an initial trust level based on their [TunkRank](http://tunkrank.com/) or their [Klout](http://klout.com/) Score.

<span style="font-weight: bold;font-size:130%;">Gamification  
</span>  
The other major issue that still needs to be tackled is how we guarantee that users do not become tired of the system and abandon it. I hope it is clear by now that the approach I described above would make things much more interesting for users interested in promoting their trust level. In fact, this is very close to what is known as [gamification](http://gamification.org/wiki/Gamification) (see also game dynamics or game mechanics for very related concepts). Attach a badge to given levels of trust for some topics and you can start competing with Foursquare check-ins.

The use of badges, or game dynamics in general in Q&amp;A sites is by no means new. Actually, stackoverflow, that I was referring to earlier in the post, delivers topical [badges](http://stackoverflow.com/badges). And obtaining the first badge on a given topic can be an important accomplishment worth noting in your resume. But stackoverflow did not come up with this idea out of the blue: levels of expertise in forums have been used for a long time (see the[ Coffee Cups/Beans in Ubuntu forums](http://ubuntuforums.org/announcement.php?f=48), for instance).

I am not saying that implementing these two approaches would guarantee Quora’s success. But not implementing them will probably guarantee the opposite. We have all seen potential in Quora. Apart from the quick integration with existing OSN and the ability to follow, there is the real-time component that brings it closer to a Q&amp;A Twitter. If they don’t fix these potential issues, somebody else will come up with an improved version that can very well be the “next big thing” that some social media gurus were seeing in Quora just some weeks ago.
