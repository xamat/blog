---
id: 30
title: 'Making decisions in an organization'
date: '2019-08-20T00:00:00+00:00'
author: xamat
layout: post
permalink: /making-decisions-8e8a902c0d3b/
reading_time:
    - ''
    - ''
categories:
    - Uncategorized
---

![](/blog/images/10-01.png)

Nobody would probably argue that it is important for any organization to make the “right” decisions. However, if you ask anyone you know how are decisions made at their company, you will most likely get vague answers at best. Maybe something like “we empower everyone to make good decisions” or “we have a distributed decision-making process”. Ask them, “What does that actually mean? Who makes a decision? When and how?” and you are likely to get a silent stare for an answer.

Of course, strong well-oiled cultures like Netflix go a step beyond and describe a few more details of the process. Netflix [culture](https://jobs.netflix.com/culture) talks about “informed captains”, “disagree openly”, and “context not control”, all of which are great aspects of a good decision-making process. However, I still find the description a bit vague for anyone outside of Netflix to really understand.

What follows is my attempt to clarify what is the optimal decision-making process for most organizations. This is based on my experience not only at Netflix, but at several other companies, including, at this point, [Curai](https://www.curai.com). [As always](https://towardsdatascience.com/cultural-overfitting-and-underfitting-or-why-the-netflix-culture-wont-work-in-your-company-af2a62e41288), I am not claiming this is a silver bullet that works always. But, I do believe this model is general enough that it should work if you have a reasonable culture to support it.

### Decision-making models

There are many models of making decisions in a group setting, but most of them can be summarized in the following three categories:

1. **Consensus-based**: decision won’t be reached until everyone in the group agrees.
2. **Majority-based**: the group votes after a discussion and the proposal with the most votes is chosen. Variations of this include things like weighted-majority.
3. **Authoritarian**: the person with the most “authority” in the group decides disregarding the opinion of the rest.

Deciding what is the right model depends on what is most important in the process. In fact, when evaluating any decision-making process it is important to look at several dimensions. The most important ones are:

- **Optimality**: How likely the model is to produce an optimal (or good) decision
- **Cost**: How long it will take to reach a decision
- **Alignment**: How hard/easy it will be to align the group with the final decision
- **Accountability**: How much clarity there will be after the fact on who is accountable for the decision

Any model will have to trade-off among those dimensions. For example, consensus-based models are generally high on alignment, but have a high cost, low accountability, and, arguably, low optimality. On the other hand, authoritarian models have low cost and high accountability, but very low alignment.

Another interesting, and somewhat controversial example, is Democracy, which can be mostly understood as a majority-based model. Democracy has relatively low cost since all it takes is to count votes. It has low optimality, as we have all learned recently, and low accountability since nobody feels responsible for the result. However, it does index high on alignment, or so it hopes. In a majority-based model everyone has had a chance to get their voice heard and it is easy to convince people to go with what “most people” want.

Unfortunately, all the above models tend to work very poorly in any company. Consensus or majority-based models are extremely inefficient and lead to sub-optimal decisions and lack of accountability. I have heard companies say they make decisions by consensus, and I hardly believe that is possible. Would love to see counter examples. On the other hand, it should be clear that authoritarian models can only get you so far and, while low-cost, are very low on optimality and alignment.

Is there a model that can be seen as the ideal tradeoff between all those competing dimensions? Yes! Meet the **Consultative decision-making model**.

### The Consultative model

This model of decision-making is based on a series of simple premises:

- The final decision will be made by an individual who is appointed ahead of time
- This individual is sometimes called [**DRI**](https://about.gitlab.com/handbook/people-operations/directly-responsible-individuals/#targetText=Apple%20coined%20the%20term%20%22directly,or%20failure%29%20of%20that%20project.), a term that was coined by Apple, stands for Directly Responsible Individual, and is used at many tech companies.
- The DRI will be held accountable for the consequences of the decision
- It is the DRI’s responsibility to include whoever is needed in the discussion phase so as to reach the best possible decision
- Once the decision is made, it is the DRI’s responsibility to make sure people are aligned
- Anyone consulted in the discussion commits to offering the best information so as to help the DRI make the right decision. They should also follow the understand/discuss/commit model described below. Consultants will be held accountable for their performance during the process, particularly when that has an impact (positive or negative) on the quality of the decision.

### The Understand-&gt;Discuss-&gt;Decide-&gt;Commit cycle

Besides the concrete model used to make the decision itself, it is important to understand that the decision process includes other activities that start before and end after the decision has been taken. In order to have an efficient and collaborative environment, it is important for everyone involved in the process to understand what is expected of them during the different phases of the process:

UNDERSTAND-&gt;DISCUSS-&gt;MAKE DECISION-&gt;COMMIT

The first two phases are pre-decision, while the last one is post-decision.

(Note that this is an extension of the popular [Disagree and Commit](https://en.wikipedia.org/wiki/Disagree_and_commit) model popularized by Sun, Intel, and more recently Amazon).

#### Understand

Anyone involved in the discussion leading to the decision should do their best to understand the constraints and the goals of the decision. It is on the DRI to clearly communicate those, but other participants should make sure they understand them and ask questions or clarifications if needed. This first phase should seek **clarity**.

#### Discuss

Once constraints and goals have been understood, people should be invited to discuss options and possibilities as openly and broadly as possible. This phase is likely to include an initial **brainstorming** activity to generate ideas. Once those ideas are generated, they should be narrowed down to the most likely and preferably ranked by all participants and ultimately the DRI. It is important to also add a **challenge** activity in which people can openly challenge the initial assumptions or the existing ranking to make sure that nothing is being missed in the discussion. This phase can take 15 minutes, or a 3 weeks since the timeline depends on the complexity of the decision and the amount of information needed to make a good decision.

Note that the goal of this phase **is not** to have everyone agree. The goal is to surface all possible relevant aspects so the decision is made with as much information as possible.

#### Commit

After the decision is made, the DRI will communicate to whoever is affected. It is very important for people to commit to the final decision, regardless of whether they agree with it or not. Getting alignment by everyone involved is extremely important to make sure the decision is then executed properly with everyone giving their best to make it succeed. It is extremely important to avoid people disengaging or even boycotting a process/project because they disagree with the decision.

As mentioned above, it is mostly on the DRI to get people to commit. An important part of doing this is communicating the decision in the right way, giving enough reasons about why the decision was taken and also acknowledging there were other options that were considered, but did not seem as good. For important decisions, it is also good to mention that the decision can be iterated on or revised at later times and give a sense of the timeline.

### Escalation

An important aspect of any decision-making process is the need to have an ability to **escalate**. If you feel strongly that the DRI is making the wrong decision or did not run the process correctly, you should feel totally ok with escalating up the DRI chain and raise your concerns to their manager or their manager’s manager.

Escalation is a good thing in any system, and nobody should feel bad about doing it. Actually, many disasters in companies and history in general could have been avoided with proper escalation.

### A simple example

So, how does this work in practice? Let’s run through a simple example:

*We have to decide where to organize our next team lunch. Since it is unclear who should be driving this, the executive team gets together. They decide that Johnny is probably the best person so they assign the task to him. Johnny is overwhelmed at first since this is an important decision, but he then remembers how he should be using the consultative model. He identifies two people in the office who have strong opinions about food (Rob, and Lisa). He also remembers that Amber has a lot of knowledge of local restaurants and is a Yelp power user. He calls for a meeting with the 3 people. He states the goal of the meeting as follows: “Let’s come up with a ranked list of options for our team lunch with pros/cons for each one”. After the meeting, they end up with a ranked list of 5 options. The ranking is done roughly based on the pros/cons identified by Rob, Lisa, and Scott. At the end of the meeting, Johnny thanks them for the feedback, and tells them he will come back with a decision. He spends a couple of hours researching in detail the options and realizes that the highest option is over budget, and the second is temporarily closed. He gets back to Rob, Lisa, and Scott informing them of the final decision (originally ranked 3rd) and explains the reasons. He invites them again to voice any concern. They all respond saying that this seems great. Johny then confirms with the executive team. Finally, he sends an email to everyone in the company announcing the decision, thanking the people who helped them, and providing some brief background on how the decision was made.*
