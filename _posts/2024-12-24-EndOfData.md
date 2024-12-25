---
id: 116
title: 'The end of the "Age of Data"? Enter the age of superhuman data and AI'
date: '2024-12-24T00:00:01+00:00'
author: Xavier
permalink: /ageofdata
image: /blog/images/116-0.png
header:
  teaser: /images/116-0.png
reading_time:
    - ''
    - ''
categories:
    - ai
    - data
---

# Everything ends, many things start again

In the ever-shifting landscape of Artificial Intelligence, pronouncements of the 'end of an era' are surprisingly common. 
The latest such declaration comes from Ilya Sutskever who recently suggested that the 'age of data'—a period defined by the
relentless pursuit of ever-larger datasets— [is drawing to a close](https://www.youtube.com/watch?v=YD-9NG1Ke5Y)  . 
But is he right?  This post will argue that the age of data is far from over. Instead, it's transforming into something even more powerful: an age of superhuman data.

<img src="/blog/images/116-0.png">

Now, to be fair, Ilya is known to be a believer in the power of big data. In fact their 2014 Sequence to Sequence paper that Ilya was 
remembering at NeurIPS this year had the following conclusions:

<img src="/blog/images/116-1.png">

The “scaling hypothesis” that Ilya is referring to here was fully developed by several OpenAI researchers (including e.g. Dario Amodei now CEO of Anthropic) 
in their 2020 paper [“Scaling Laws for Neural Language Model”](https://arxiv.org/abs/2001.08361 ) where they showed how the performance of a large neural network 
could be predicted by the number of parameters, dataset size, and compute budget.

So, here we are, just 12 years later, being told that the age of data is over. We don't have any more data to pump into our models. We ran out of our 'fossil fuel' of AI.Or have we?"

 **(Historical digression**:
One of the interesting things when you have been around for a while or you study a bit of history is that you realize that many scientific 
ideas or trends die and resurrect again and again. This is particularly true in AI, where a couple of AI winters declared the whole field itself mostly 
dead. As a reminder, the perceptron, which is the basic unit of Artificial Neural Networks, was also dismissed by Minski in his 1969 book 
[Perceptrons](https://en.wikipedia.org/wiki/Perceptrons_(book)).

It is also important to note that the simpler scaling law that states that in order to improve your model performance all you need is more data 
and more parameters was discovered years earlier. One of the earliest papers to be cited in that context is 2001 Banko and Brill’s 
[“Scaling to Very Very Large Corpora for Natural Language Disambiguation”](https://aclanthology.org/P01-1005.pdf ). Another popular, but a bit more recent, 
reference is [“The Unreasonable Effectiveness of Data”](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf ) 
by Halevy, Norvig, and Pereira from Google. Both of these contributed to the beginning of the “Age of Data” that is probably best illustrated by Chris Anderson’s 
[“The End of Theory: The Data Deluge Makes the Scientific Method Obsolete”](https://www.wired.com/2008/06/pb-theory/ ) in Wired where he claimed that the world of 
discovery and science was going to be dominated by data. Models and algorithms were to become obsolete. I argued against this point-of-view in my 2012 post 
[“More data or better models?”](https://amatria.in/blog/more-data-or-better-models/) .
**)**

<img src="/blog/images/116-2.png">


# Not all data is created equal

It is important to underscore that for data, size is not all that matters. You can have a huge dataset with very low informational value, or a small one with 
lots of knowledge density. In an extreme, you could take a single data point and replicate it 1 trillion times to have a pretty large dataset that only contains one 
piece of information. Yes, that is an extreme, but most of the data on the Internet is very low value. In fact, some of it has negative value! When I used to work on AI 
for healthcare I used to half jokingly talk about how the internet had very little high quality medical data and lots of very dubious Reddit threads with questionable medical 
advice.

Not surprisingly, advances in models such as [Llama 3](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/ ) list better data as the number one reason 
behind their improvement. An important detail of their approach, which is common nowadays, is their focus on having specific domains such as coding and math specifically 
represented. They also pay special attention to having the right “data mix” that represents the different kinds of knowledge the model should learn. The final mix has 
“roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens”. Note that the category of 
“general knowledge” could be broken down into whatever domains are important or relevant.


# Beyond the Web: The Value of Proprietary and Labeled Data

While the vast expanse of the public internet has been the primary training ground for large language models, it represents only a fraction of the data that exists and, 
more importantly, only a fraction of the data that is valuable for pushing the boundaries of AI. A crucial point that gets overlooked in discussions about the "end of the 
age of data" is the existence of vast troves of proprietary data, often highly specialized and meticulously curated, that are not publicly available on the internet.
Consider the healthcare industry, where patient records, clinical trial results, and medical imaging data hold immense potential for training specialized AI models. Similarly, 
companies like Waymo and Tesla possess massive, proprietary datasets of driving data that are far more detailed and comprehensive than anything available publicly. This kind of 
high-quality, domain-specific data is becoming increasingly important as AI moves beyond general-purpose tasks towards more specialized applications.

The value of this proprietary data is undeniable, even in the age of powerful pre-trained models. While some early proponents of the GPT era argued that pre-trained models 
would negate the need for specialized datasets through zero-shot learning, this has not proven to be entirely true. Instead, proprietary data is being leveraged in new ways, 
such as through techniques like fine-tuning, retrieval-augmented generation (RAG), or by strategically injecting it into the model's context window. These methods allow models 
to specialize and adapt to specific domains and tasks, demonstrating that the value of data persists even if the mechanisms for extracting that value are evolving.

Another critical aspect of valuable data is whether it is labeled or not. While LLMs are typically pre-trained on vast quantities of unlabeled data using self-supervision, 
the subsequent steps of fine-tuning and reinforcement learning often rely on high-quality, labeled datasets. As highlighted in ["Beyond Token Prediction: the post-Pretraining 
journey of modern LLMs"](https://amatria.in/blog/postpretraining), much of the recent progress in GenAI has come from advances in these post-training stages, which require 
carefully curated datasets with explicit labels. Creating these labeled datasets, particularly at scale and with high accuracy, remains a significant challenge and a major 
bottleneck in many AI applications. This is precisely why companies specializing in data labeling, like Scale.ai, are seeing continued growth and high valuations.

It is important to note that the need for labeled data, while crucial, does not directly contradict the idea that the "age of pretraining data" might be waning, as Ilya 
suggested. Pretraining still relies heavily on the vast, unlabeled data of the public web. However, the increasing importance of post-training techniques and the value of 
proprietary, labeled datasets underscore a shift towards a more nuanced understanding of data's role in the future of AI. The age of data is not ending, but it is certainly 
evolving, with a growing emphasis on quality, specificity, and the strategic use of data beyond the initial pretraining phase.

<img src="/blog/images/116-3.png">

# Beyond human and synthetic data: the Age of Superhuman Data

The vast majority of data currently used to train AI models is generated by humans – from the text on Wikipedia, Quora and Reddit to the images, videos, and code found across 
the internet. Humans also play a crucial role in creating the labeled datasets used for post-training. However, the question of whether this reliance on human-generated data 
can or should continue is a subject of intense debate. While it is true that much of the existing web data has been "used up" for pretraining purposes, the kind of data that 
future AI will need is very different.

One promising avenue is **synthetic data**, artificially generated data that can augment or even replace real-world data. The potential of synthetic data has been met with both 
excitement and skepticism. Some research, such as the paper "AI models collapse when trained on recursively generated data" (https://www.nature.com/articles/s41586-024-07566-y), 
suggests that models trained solely on their own outputs can degrade. However, I believe these limitations can be overcome by generating synthetic data from diverse and more 
complex models, as argued in ["On the Diversity of Synthetic Data and its Impact on Training Large Language Models"](https://arxiv.org/abs/2410.15226). There is a rich and 
growing body of work on how to best generate and use synthetic data that I won't review here (see e.g. 
[“Machine Learning for Synthetic Data Generation: A Review”](https://arxiv.org/abs/2302.04062) and 
[“Generative AI for Synthetic Data Generation: Methods, Challenges and the Future”](https://arxiv.org/abs/2403.04190) ).

In fact, I believe we are rapidly approaching a turning point: **the end of the age of data as defined by human limitations**. While human-generated data has been essential in 
bootstrapping AI, it is inherently constrained by our own perceptions, biases, and the inefficiencies of human communication. As [Yann LeCun](https://www.noemamag.com/ai-and-the-limits-of-language/) 
has pointed out, human language is an imperfect tool for capturing the full complexity of reality. We are, in a sense, "lossy" encoders of the world around 
us.

The future of data lies in moving beyond these limitations. My hypothesis is that LLMs are not the end but the means to an end: they are good enough to build AI agents that will 
capture much better data. In fact, they will capture superhuman data. We are on the cusp of an era where AI agents, equipped with advanced sensors and sophisticated reasoning 
capabilities, will interact directly with the world, generating data that is richer, more accurate, and less filtered by human interpretation. Multimodal agents, such as Google 
Deepmind's [Astra](https://deepmind.google/technologies/project-astra/), will be deployed in real environments to create better data and representations of the world, enabling new 
forms of scientific discovery, as previewed in ["Scientific discovery in the age of artificial intelligence"](https://www.nature.com/articles/s41586-023-06221-2). 
**This is not to say that we should expect or even desire a monolithic Artificial General Intelligence. As I have argued elsewhere, specialized, superhuman agents are likely 
to be both more achievable and more beneficial in addressing specific, complex challenges** (see my 
[“Beyond Singular Intelligence: Exploring Multi-Agent Systems and Multi-LoRA in the Quest for AGI”](https://amatria.in/blog/multiagents)).
Consider autonomous vehicles like Waymo's. While currently limited in scope, they represent a first step towards this future. These vehicles collect vast amounts of real-time 
data about the world, data that can be used to improve their own performance and, crucially, could be used to create richer datasets beyond the task of driving. In the near 
future, we can envision AI agents designed explicitly for data collection and generation, operating across a wide range of domains, from materials science to medical research.

This transition to agent-generated data will not be without its challenges. **Importantly, we must acknowledge the continued need for human-generated data in specific areas, 
particularly for aligning AI systems with human values and preferences. Data reflecting human desires, feedback, and ethical judgments will remain crucial for ensuring that 
AI remains beneficial and aligned with our goals. This includes data used for techniques like Reinforcement Learning from Human Feedback (RLHF) and other alignment methods. 
In addition, human data will continue to be invaluable for personalizing AI experiences, ensuring that systems are responsive to individual needs and preferences. While 
superhuman data can provide a more accurate and comprehensive understanding of the world, it is human data that provides the crucial link to what matters to us as humans**.

It is not an overstatement to say that data is not only not over, but it is in fact about to get much bigger and better, thanks to AI.

# To summarize

While it is true that we have almost extracted all the value there is in the low-quality web data, that is far from being the only or the best kind of data we need to train AI. 
There is lots of valuable data hidden in proprietary data stores. All the existing data is limited by the fact that it is human mediated and generated. It is hard for AI to become 
superhuman at many tasks when the data it is trained on is probably sub-human. In the near future, agentic AI will generate streams of higher quality data that is not limited by 
our human abilities. As long as we make sure we still input human data in the form values and preferences for alignment and personalization, we should all look forward to this 
near future with agentic AI with super-human capabilities.

