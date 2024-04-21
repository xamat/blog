---
id: 102
title: "On why LLMs are just like medical doctors"
date: '2023-04-27T00:00:01+00:00'
author: Xavier
permalink: /llmsdoctors
image: /blog/images/102-1.png
header:
  teaser: /images/102-1.png
reading_time:
    - ''
    - ''
categories:
    - Artificial Intelligence
    - Machine Learning
    - LLMs
---


<img src="/blog/images/102-1.png">

There is a lot of talk nowadays on avoiding hallucination in LLMs: we need to get IA models to be truthful and furthermore, 
we need them to understand their own uncertainty. While I agree those are worthy research questions, 
I don’t think we should put all our eggs in the basket of building completely accurate and self-aware AI models. 
In fact, I don’t even think these are practical and beneficial goals.

What should we do instead? We should teach  users of AIs that there is no "absolute truth" in anything they get from an LLM. 
In fact, there is no absolute truth in anything a user gets from a search engine either. Or... from a medical doctor! 
I know, this analogy between human doctors and LLMs might sound a controversial, but allow me to explain:

# The problem

In a [fascinating study](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/1731967) published ten years ago in the JAMA Internal Medicine journal, Meyer et al. showed that doctors 
had a very poor diagnostic accuracy. In easy cases, doctors average accuracy was a bit over 55% while in harder cases 
it did not even get to 6%. While those results might sound astonishing, the "slightly better than 50%" accuracy has been replicated in similar studies. 
And, there is even a more astonishing finding in Meyer's study: the level of confidence from doctors is almost the same for easy cases (72%) 
and hard cases (64%). So, basically, doctors not only have a horrible diagnostic accuracy. They also have no clue on when they might be wrong.

So, how does that translate to LLMs. Well, LLMs also have very little to no notion of the uncertainty on their "reasoning".
While they are [way better](https://www.medrxiv.org/content/10.1101/2023.04.20.23288859v2) than average doctors at diagnosis, 
they can also very confidently make stuff up and spit it out.

So, going back to my initial point: users should treat the output of an LLM just as they should treat medical diagnosis: 
a qualified opinion, not a ground truth.

# The Solution: ensembles of expert/AI opinions

And, what is the appropriate way to deal with this? What do you do if you get an important diagnosis from a doctor and 
you have doubts? You ask for a second opinion (and maybe a third and a fourth). In fact, according to 
[some studies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6484633/), 
diagnostic accuracy can increase to 85% when adding the opinion of 9 doctors.

This same approach can be used with LLMs. In fact LLMs allow for many different variations of "second opinions". 
In the simplest one, you can simply ask the SAME LLM the same question several times and get a sense of the variability and then 
make up your mind. A bit more involved approach requires not only variability in the response of the same model, 
but asking different models, which can then be combined using some ensemble technique (with the simplest being majority voting). 
There are even more complex solutions where different models (or agents) can be designed to follow different roles. 
In ["DERA: Enhancing Large Language Model Completions with Dialog-Enabled Resolving Agents"](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=eoBHpj4AAAAJ&sortby=pubdate&citation_for_view=eoBHpj4AAAAJ:J-pR_7NvFogC), 
my former team at Curai develops medical research and medical decision maker agents and show that combining these two 
kinds of agents improves results over using a single GPT-4.

Interestingly, just a few hours after publishing the first version of this post, I read about NVidia's [guardrails](https://github.com/NVIDIA/NeMo-Guardrails) toolkit for minimizing hallucination. It turns out that they implement exactly this approach.

# The importance of messaging

While the solution outlined below is readily available, generalized, and scales well, it cannot be implemented unless it comes with a 
great deal of effort in user messaging.

<img src="/blog/images/102-2.png">

It is true that applications like ChatGPT do have a disclaimer about their “factuality” (see below), but this is not enough.

Chatbots based on LLMs should be explicit about their stochasticity and constantly invite the user to try again, reformulate their 
question, and do their research. I have heard concerns that doing so might erode user “trust” in AI, but that is exactly the point! 
Users should trust AI less and take more agency over making decisions and verifying the input that goes into making them.

I see a good opportunity for regulation here. While I think mandating transparency or explainability is a moot point, 
mandating AIs to explicitly and clearly remind users of their stochasticity and encouraging to ask again/differently and verify sources is easy, 
and reasonable.

# Conclusion

Just as many human experts, LLMs fail to be accurate and factual while conveying a high degree of confidence. 
Getting them to be more accurate and factual is good, but we cannot simply hope to get them 100% right all the time. 
In fact, even what is “right” might depend on cultural background and personal preferences and beliefs. What we need to do is teach 
humans to deal with uncertainty and don’t put the agency of decisions and control on AIs. We can do that by conveying the right message, 
with conviction, and repeatedly.

