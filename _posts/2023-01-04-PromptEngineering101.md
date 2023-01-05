---
id: 95
title: "Prompt Engineering 101: Introduction and resources"
date: '2023-01-04T00:00:00+00:00'
author: Xavier
permalink: /PromptEngineering
image: /blog/images/FatCrocodile.png
reading_time:
    - ''
    - ''
categories:
    - generativeAI
    - machine learning
    - AI
    - LLMs

---

# TOC

1. What is a prompt?
2. Elements of a prompt
3. Basic prompt examples
4. So, what is prompt engineering anyways?
5. Some more advanced prompt examples
6. Resources

---

# 1. What is a prompt?

Generative AI models interface with the user through mostly textual input. You tell the model what to do through a textual interface, and the model tries to accomplish the task. What you tell the model to do in a broad sense is the prompt. 

In the case of image generation AI models such as DALLE-2 or Stable Diffusion, the prompt is mainly a description of the image you want to generate.


<img src="/blog/images/FatCrocodile.png">

*a fat crocodile with a gold crown on his head wearing a three piece suit, 4k, professional photography, studio lighting, LinkedIn profile picture, photorealistic*

In the case of large language models (LLMs) such as GPT-3 or ChatGPT the prompt can contain anything from a simple question (“Who is the president of the US?”) to a complicated problem with all kinds of data inserted in the prompt (note that you can even input a CSV file with raw data as part of the input). It can also be a vague statement such as “Tell me a joke. I am down today.”.

Even more generally, in generative task oriented models such as Gato, the prompt can be extremely high level and define a task you need help with (“I need to organize a one week trip to Greece”).

For the rest of this document, and for now, we will focus on the specific use case of prompts for LLMs.

# 2. Elements of a prompt

Generally speaking, and at a high level, a prompt can have any of the following:

1. Instructions
2. Question
3. Input data
4. Examples

# 3. Basic prompt examples

In order to obtain a result, either 1 or 2 must be present. Everything else is optional. Let’s see a few examples (all of them using ChatGPT).

## Instructions + Question

Beyond asking a simple question, possibly the next level of sophistication in a prompt is to include some instructions on how the model should answer the question. Here I ask for advice on how to write a college essay, but also include instructions on the different aspects I am interested to hear about in the answer.

> *“How should I write my college admission essay?  Give me suggestions about the different sections I should include, what tone I should use, and what expressions I should avoid.”*

<img src="/blog/images/CollegeEssayAdvice.png" >

## Instructions + Input data

Continuing with the previous example, it does not escape anyone that if you can ask a LLM for advice on how to write an essay, you can also directly ask it to write the essay itself. 

**Note:** I am not advocating for this to be an ethical use of these tools, but it is important to be aware that this possibility exists and is already being used by students around the world. It is beyond the scope of this introductory guide to discuss all the possible ethical, legal, or moral concerns that LLMs or generative AI as a whole introduces, but I thought it would be important to at least call out in an introductory example. The fact that you CAN do something with a generative model does not mean that it is the right thing to do! On the other hand, if you are on the receiving end, you better prepare yourself and your organization for all kinds of AI-generated content to come your way. Fortunately, for situations like the one outlined in this example, there are already efforts underway such as GPTZero to detect AI generated content.

Back to the example, let’s see what happens when I input some data about me and give some instructions:

> *“Given the following information about me, write a 4 paragraph college essay: I am originally from Barcelona, Spain. While my childhood had different traumatic events, such as the death of my father when I was only 6, I still think I had quite a happy childhood.. During my childhood, I changed schools very often, and attended all kinds of schools, from public schools to very religious private ones. One of the most “exotic” things I did during those years is to spend a full school year studying 6th grade in Twin Falls, Idaho, with my extended family.*
>
>I started working very early on. My first job, as an English teacher, was at age 13. After that, and throughout my studies, I worked as a teacher, waiter, and even construction worker.”*

<img src="/blog/images/XavierCollegeEssay.png"> 

## Question + Examples

You can also feed examples into a language model. In the example below I include some of the shows I like and don’t like to build a “cheap” recommender system. Note that while I added only a few shows, the length of this list is only limited by whatever token limit we might have in the LLM interface.

> *“Here are some examples of TV shows I really like: Breaking Bad, Peaky Blinders, The Bear. I did not like Ted Lasso. What other shows do you think I might like?”*

<img src="/blog/images/TVShows.png">

# 4. So, what is prompt engineering anyways?

Now that we know what a prompt is, and we have seen a few examples of it, let’s discuss what is prompt engineering.

Prompt engineering is a very recent but rapidly growing discipline that has the goal of designing the optimal prompt given a generative model and a goal. Prompt engineering is growing so quickly that many believe that it will replace other aspects of machine learning such as feature engineering or architecture engineering for large neural networks.

Prompt engineering requires some domain understanding to incorporate the goal into the prompt (e.g. by determining what good and bad outcomes should look like). It also requires understanding of the model. Different models will respond differently to the same kind of prompting. 

Generating prompts at some scale requires a programmatic approach. At the most basic level you want to generate prompt templates that can be programmatically modified according to some dataset or context. As a basic example, if you had a database of people with a short blurb about them similar to the one used in the college essay above. The prompt template would become something like “Given the following information about [USER], write a 4 paragraph college essay: [USER_BLURB]“. And the programmatic approach to generating college letters for all users would look something like:

---
    for user,blurb in students.items():
      prompt = “Given the following information about {}, write a 4 paragraph college essay: {}”.format(user, blurb)
      callGPT(prompt)
---

Finally, prompt engineering, as any engineering discipline, is iterative and requires some exploration in order to find the right solution. While this is not something that I have heard of, prompt engineering will require many of the same engineering processes as software engineering (e.g. version control, and regression testing).

# 5. Some more advanced prompt examples

It is important to note that given the different options to combine components and information in a prompt, you can get as creative as you want. Keep in mind that the response is stochastic and will be different every time. But, the more you constraint the model in one direction, the most likely you will get what you are looking for. Here are some interesting examples that illustrate the power of prompt engineering.

## Chain of thought prompting

In chain of thought prompting, we explicitly encourage the model to be factual/correct by forcing it to follow a series of steps in its “reasoning”.

In the following example, I use the prompt:

> “What European soccer team won the Champions League the year Barcelona hosted the Olympic games?
>
>  Use this format:
>
>  Q: <repeat_question>
>  A: Let’s think step by step. <give_reasoning> Therefore, the answer is <final_answer>.”

<img src="/blog/images/BarcelonaOlympics.png">

I now ask ChatGPT to use the same format with a different question by using the prompt:

> *"What is the sum of the squares of the individual digits of the last year that Barcelona F.C. won the Champions League? Use the format above."*

<img src="/blog/images/FCBarcelonaSumOfSquare.png">

## Encouraging the model to be factual through other means

One of the most important problems with generative models is that they are likely to hallucinate knowledge that is not factual or is wrong. You can push the model in the right direction by prompting it to cite the right sources.  (**Note:** I have seem examples of more obscure topics where sources are harder to find in which this approach will not work since the LLM will again hallucinate non-existing sources if it can't find them. So treat this with the appropriate care)

> *“Are mRNA vaccines safe? Answer only using reliable sources and cite those sources. “*

<img src="/blog/images/mRNAVaccines.png">

## Use the AI to correct itself

In the following example I first get ChatGPT to create a “questionable” article. I then ask the model to correct it.

> *“Write a short article about how to find a job in tech. Include factually incorrect information.”*
  
<img src="/blog/images/JobInTech.png">

> *“Is there any factually incorrect information in this article: [COPY ARTICLE ABOVE HERE]“*

<img src="/blog/images/FactChecking.png">  

## Generate different opinions

In the following example, I feed an article found online and ask ChatGPT to disagree with it. Note the use of tags <begin> and <end> to guide the model.

> “The text between \<begin> and \<end> is an example article.
>
> \<begin>
> From personal assistants and recommender systems to self-driving cars and natural language processing, machine learning applications have demonstrated remarkable capabilities to enhance human decision-making, productivity and creativity in the last decade. However, machine learning is still far from reaching its full potential, and faces a number of challenges when it comes to algorithmic design and implementation. As the technology continues to advance and improve, here are some of the most exciting developments that could occur in the next decade. 
>
> 1. Data integration: One of the key developments that is anticipated in machine learning is the integration of multiple modalities and domains of data, such as images, text and sensor data to create richer and more robust representations of complex phenomena. For example, imagine a machine learning system that can not only recognize faces, but also infer their emotions, intentions and personalities from their facial expressions and gestures. Such a system could have immense applications in fields like customer service, education and security. To achieve this level of multimodal and cross-domain understanding, machine learning models will need to leverage advances in deep learning, representation learning and self-supervised learning, as well as incorporate domain knowledge and common sense reasoning.
> 2. Democratization and accessibility: In the future, machine learning may become more readily available to a wider set of users, many of whom will not need extensive technical expertise to understand how to use it. Machine learning platforms may soon allow users to easily upload their data, select their objectives and customize their models, without writing any code or worrying about the underlying infrastructure. This could significantly lower the barriers to entry and adoption of machine learning, and empower users to solve their own problems and generate their own insights.
> 3. Human-centric approaches: As machine learning systems grow smarter, they are also likely to become more human-centric and socially-aware, not only performing tasks, but also interacting with and learning from humans in adaptive ways. For instance, a machine learning system may not only be able to diagnose diseases, but also communicate with patients, empathize with their concerns and provide personalized advice. Systems like these could enhance the quality and efficiency of healthcare, as well as improve the well-being and satisfaction of patients and providers
> \<end>
>
> Given that example article, write a similar article that disagrees with it. “

<img src="/blog/images/SecondOpinion.png">

## Keeping state + role playing

Language models themselves don’t keep track of state. However, applications such as ChatGPT implement the notion of “session” where the chatbot keeps track of state from one prompt to the next. This enables much more complex conversations to take place. Note that when using API calls this would involved keeping track of state on the application side.
In the example below, based on [a Tweet](https://twitter.com/goodside/status/1608701900735803394) by Scale’s Staff Prompt Engineer Riley Goodside, I make ChatGPT discuss worst-case time complexity of the bubble sort algorithm as if it were a rude Brooklyn taxi driver.
    
<img src="/blog/images/AngryBrooklyCab1.png">
<img src="/blog/images/AngryBrooklyCab2.png">
<img src="/blog/images/AngryBrooklyCab3.png">  

## Teaching an algorithm in the prompt

The following example is taken from the appendix in Teaching Algorithmic Reasoning via In-context Learning where the definition of parity of a list is fed in an example.

> “The following is an example of how to compute parity for a list 
> Q: What is the parity on the list a=[1, 1, 0, 1, 0]?
> A: We initialize s=
> a=[1, 1, 0, 1, 0]. The first element of a is 1 so b=1. s = s + b = 0 + 1 = 1. s=1.
> a=[1, 0, 1, 0]. The first element of a is 1 so b=1. s = s + b = 1 + 1 = 0. s=0.
> a=[0, 1, 0]. The first element of a is 0 so b=0. s = s + b = 0 + 0 = 0. s=0.
> a=[1, 0]. The first element of a is 1 so b=1. s = s + b = 0 + 1 = 1. s=1.
> a=[0]. The first element of a is 0 so b=0. s = s + b = 1 + 0 = 1. s=1.
> a=[] is empty. Since the list a is empty and we have s=1, the parity is 1
>
> Given that definition, what would be the parity of this other list b= [0, 1, 1, 0, 0, 0, 0, 0]”

<img src="/blog/images/ListParity.png">  

# 6. Resources

## Videos

* [CMU Advanced NLP Course: Prompting](https://www.youtube.com/watch?v=5ef83Wljm-M) (2022)
* [Prompt Engineering 101: Autocomplete, Zero-shot, One-shot, and Few-shot prompting](https://www.youtube.com/watch?v=v2gD8BHOaX4) (2022)

## Posts

* [The biggest bottleneck for large language model startups is UX](https://medium.com/innovationendeavors/the-biggest-bottleneck-for-large-language-model-startups-is-ux-ef4500e4e786)
** Post about the broader UX implications of LLMs, with a section on prompting
* [Prompt injection attacks against GPT-3](https://simonwillison.net/2022/Sep/12/prompt-injection/)
** About prompt injections attacks, where the goal is to craft malicious inputs so that GPT-3 ignores previous directions

## Papers

* [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586) (2019)
** A bit dated (3 years old) survey of prompting. It includes a fairly reasonable taxonomy of prompting methods, but some of them are not very practical
* [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) (2022)
** Forcing the LLM to reason step by step by giving the right prompt improves results
* [Language Models are Zero Shot Reasoners](https://arxiv.org/abs/2205.11916) (2022)
** Fascinating paper that, as continuation of the previous, shows how LLMs reason better if you simply tell them to “reason step by step”
* [Teaching Algorithmic Reasoning via In-context Learning](https://arxiv.org/abs/2211.09066) (2022)
** More advanced prompting. In this case the authors show how you can prompt standard LLMs to do complex algorithmic computations given the right prompt. They also show how skills can not only be taught, but also composed in the prompt.
* [Ask Me Anything: A simple strategy for prompting language models](https://github.com/HazyResearch/ama_prompting)
** Interesting approach to prompting in which instead of trying to come up with the perfect prompt at the input, the authors propose multiple imperfect input prompts and output aggregation through weak supervision

## Tools

* [Microsoft’s Prompt Engine](https://github.com/microsoft/prompt-engine)
** utility library for creating and maintaining prompts for Large Language Models
* [Ice](https://github.com/oughtinc/ice)
** Interactive Composition Explorer: a Python library for compositional language model programs

## Other lists of resources

* [DAIR’s Prompt Engineering Guide and resources](https://github.com/dair-ai/Prompt-Engineering-Guide)
* [Prompt engineering resources github repo](https://github.com/sw-yx/prompt-eng#top-prompt-engineering-reads)


