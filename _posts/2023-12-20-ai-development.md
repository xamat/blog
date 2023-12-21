---
id: 111
title: "AI as the New Member of the Engineering Team: Crafting an End-to-End AI Application with AI"
date: '2023-12-20T00:00:01+00:00'
author: Xavier
permalink: /aidevelopment
image: /blog/images/111-0.png
header:
  teaser: https://amatriain.net/blog/images/111-0.png
reading_time:
    - ''
    - ''
categories:
    - Artificial Intelligence
    - LLMs
    - Prompt Engineering
    - software engineering
---

# Exploring the AI-Driven Future of Software Development

Embarking on an experiment that blends the boundaries of AI and software engineering, a few days back I set out to explore a provocative hypothesis. Will AI-generated code dominate software development in the near future, accounting for over 80% of all coding? This question is not just theoretical; it's a forecast based on the rapid advancements in AI, and I'm putting it to the test through a hands-on project.

<img src="/blog/images/111-0.png">

## The Challenge: AI as the Core Developer

In this unique endeavor, I wanted to construct a comprehensive state-of-the-art AI end-to-end application in mere hours, a task that traditionally would require extensive manual coding. The twist is my reliance on advanced AI, particularly ChatGPT from OpenAI, known for its robust natural language processing and coding capabilities. This experiment is more than an exploration of AI's utility in coding; it's a deep dive into the evolving synergy between AI tools and software development, challenging the conventional roles and methods in the field.

## AI: More Than a Tool, a Collaborator

ChatGPT with the latest GPT4 model in the background, isn't just a sophisticated piece of technology; it represents the pinnacle of AI's integration into creative and technical processes. My goal was for it to act not just as a tool, but as a collaborator, bringing to the table its capacity to understand, generate, and debug code. This collaboration is a glimpse into a future where AI's role in software development transcends assistance, becoming a core component of the creative and development process.

## Embarking on a Groundbreaking Journey

Join me in this exploration. This journey is more than a technical challenge; it's a quest to uncover the potential of AI in redefining the software development landscape, signifying a shift in how we conceptualize and execute coding in an AI-augmented future.

# The App and the Tools: Venturing into Unfamiliar Territory

In this segment of my journey, I decided to push my boundaries by choosing technologies I had never worked with before. The goal was to create a personalized chatbot, "Xavibot," designed to respond as if it were me. You can interact with this chatbot [here](https://amatriain.net/Xavibot). 

<img src="/blog/images/111-1.png">

*A Note on Performance and Feedback:* Should you experience timeouts or other issues while using the bot, it's worth noting that it's hosted on a lower tier of Azure, which may affect performance. I welcome any feedback or queries at xavier at amatriain dot net.

For this project, I utilized:

- Node.js for the Backend: A deliberate choice over Python to challenge myself with an unfamiliar environment and assess ChatGPT's effectiveness in aiding with new technologies.
- React for the Front-End: Leveraging the react-chatbot-kit, I ventured into modern UI design, an area where my experience is limited.
- OpenAI Assistants APIs with GPT4: A choice driven by the desire to explore this new and powerful technology from OpenAI, with which I had no prior experience.

<img src="/blog/images/111-2.png">

The source code for the application is available on my [my Github](https://github.com/xamat/Xavibot). I encourage others to use it for their own projects or adaptations.

Deployment involved Azure for the Node.js backend and GitHub pages for the front-end. This process included new challenges for me, such as configuring CORS and managing a secrets vault for remote keys – all first-time experiences.
Reflection on the Initial Phase:

This project was not just about building a state-of-the-art AI application; it was a test of how quickly and efficiently I could adapt to new technologies with AI assistance. Despite my extensive background, I approached most of the tools used in this project as a novice. This experience sheds light on the current capabilities of AI in supporting software development, especially when diving into unfamiliar tech waters.

## OpenAI's Assistants API

The OpenAI [Assistants APIs](https://platform.openai.com/docs/assistants/overview) represent a significant advancement in chatbot development. This API simplifies the process, eliminating the need for developers to delve into complex aspects like memory management and retrieval-augmented generation (RAG), or the intricacies of prompt engineering and orchestration.

In creating "Xavibot" available [here](https://amatriain.net/Xavibot), I configured the assistant to perform RAG using two specific files, complemented by a basic prompt structure. While this was sufficient to create a functional version 0.1 of the bot, more sophisticated prompt engineering could potentially elevate its capabilities. My initial approach was to prioritize speed and simplicity in deployment.

My experience with the Assistants API has been insightful. For simple chatbot applications, it is exceptionally efficient and user-friendly. However, when it comes to applications requiring greater control and flexibility, the API shows limitations. Future experiments and developments could explore how advanced customization might overcome these constraints and expand the API's utility in more complex scenarios.

# The Positive Impact of AI in Software Development in 2023

My foray into AI-first software development has been an overwhelmingly positive experience. ChatGPT, serving as both a knowledgeable pair programmer and coach, accelerated my project's development significantly. Within just a few hours, I had a locally running application with about 80% of the intended functionality. This rapid progress was encouraging, though I was aware that perfecting the remaining 20% would be more time-consuming, adhering to the familiar power law dynamics of software development.

Enhanced by RAG and Continuous Interaction

The Retrieval-Augmented Generation (RAG) feature in ChatGPT proved exceptionally useful. Even when faced with complex queries, such as specific API usage, directing ChatGPT to relevant documentation often resulted in accurate and helpful responses.

What truly stood out was the ability to engage in a persistent dialogue with ChatGPT, iterating over problems until they were resolved. This process sometimes involved multiple refinements of my code, with ChatGPT providing consistent and relevant suggestions. It was akin to having a dynamic, interactive version of StackOverflow, but with the added advantage of contextual understanding and memory retention.

## AI vs Traditional Resources

Interestingly, on the few occasions when I consulted StackOverflow, I found the solutions there to be less effective than ChatGPT's. This experience aligns with the growing perception of AI as a formidable tool in the realm of technical problem-solving.

## The Supportive Nature of AI

An aspect that particularly resonated with me was ChatGPT's unwaveringly supportive and positive tone. Even in moments of frustration, when I expressed dissatisfaction with the responses, ChatGPT maintained its composure, apologizing and continuing to offer alternative solutions. This emotional intelligence, often crucial in pair programming and mentorship scenarios, significantly enhances the collaborative experience.

# Navigating the Challenges of AI-Assisted Software Development

My journey with AI-first software development, while largely positive, also revealed several areas where AI, specifically ChatGPT, could be improved.

## Data Loss and Backup Issues

A significant issue was the loss of my chat history due to corruption in the ChatGPT thread, a common problem I discovered others facing as well. This not only disrupted my workflow but also erased valuable insights and progress. The lack of an integrated backup feature in OpenAI's platform is a notable gap, necessitating third-party solutions for chat backup and restoration.

## Limitations in Information Accuracy and Complexity

GPT-4's knowledge base often lacked up-to-date information, leading to recommendations of deprecated or incompatible resources, such as a React toolkit that wasn't suitable for recent React releases. Additionally, when directed to online documentation for APIs, ChatGPT's responses were sometimes vague or missed the mark. For instance, when I asked about adding a secret to the Azure vault based on Azure documentation, ChatGPT incorrectly guided me to use the wrong role assignment.

## Tendency to Overcomplicate Solutions

ChatGPT frequently suggested overly complex solutions. For example, it led me towards using React Context and Redux for a situation where a simpler global state approach would suffice. Similarly, it advised setting up an Azure vault for secrets management, which was an overkill for my specific needs. While ChatGPT eventually acknowledged simpler alternatives, the initial guidance towards more complex solutions often proved to be time-consuming.

## Difficulty with Error Analysis and Code Context

One of the most significant challenges was ChatGPT's struggle to diagnose errors in relation to the provided code. A striking example was when it suggested an extensive refactor of my code, while the issue was actually due to a misspelled variable. This inability to pinpoint simple issues, coupled with a tendency to suggest extensive code changes, often led to inefficient problem-solving.

# Conclusions: Embracing the AI Revolution in Software Development

After this deep dive into AI-first software development using ChatGPT with GPT4, I've come to appreciate its distinct advantages over other coding-specific AI tools. While solutions like GitHub Copilot have their merits, the versatility and depth of ChatGPT's assistance surpassed my expectations and demonstrated its potential as a superior tool for developers.

However, there's room for improvement. Fine-tuning AI models for specific coding tasks and integrating them more seamlessly into development environments like Visual Studio Code could address some of the challenges I encountered. Such enhancements would further streamline the development process and amplify the benefits of AI assistance.

I remain bullish about the transformative impact of AI on software development. The advancements I've witnessed and utilized are just the tip of the iceberg. For developers at any career stage, now is the time to embrace AI tools. With AI assistance, even developers with average skills can significantly boost their productivity and effectiveness, potentially becoming '10X engineers'. And for those already excelling in their field, these tools could amplify their capabilities, leading to unprecedented levels of efficiency and innovation.

In conclusion, the integration of AI into software development isn't just a trend; it's a paradigm shift. It's reshaping how we approach coding and problem-solving, offering new heights of potential for every developer willing to adapt and learn.



