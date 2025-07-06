---
id: 119
title: "The AI Co-Developer 18 Months Later"
date: '2025-07-06T00:00:01+00:00'
author: Xavier
permalink: /ai-code-refactor
image: /blog/images/119-0.png
header:
  teaser: /blog/images/119-0.png
reading_time:
  - ''
  - ''
categories:
  - Artificial Intelligence
  - LLMs
  - software engineering
---

Eighteen months ago, I wrote a post on how Large Language Models would change software development, sharing my early experiences with the technology. You can read those initial thoughts [here](https://amatriain.net/blog/aidevelopment). A lot of things have happened since then, and my early observations are by now way outdated. That is why I decided to spend some time during a recent break to work on my side project and see how much better things are now.

<img src="/blog/images/119-0.png">

This time, my primary tool was [Cursor](https://cursor.sh/), an AI-first code editor, which I used almost exclusively in its agentic mode. I experimented with various backend models, including Gemini 1.5 Pro and Claude 3 Sonnet, though I often found myself defaulting to the "Auto" setting. I also incorporated [Jules](https://jules.google.com/), Google's own software development agent, into my workflow.

Instead of starting a project from scratch, I decided to undertake a full refactoring of my old project, [Xavibot](https://github.com/xamat/Xavibot). The objective was to test the mettle of these AI agents on a relatively complex codebase while attempting substantial changes—a scenario often trickier than a clean-slate build.

The initial scope of the refactoring included migrating the chatbot's backend from Azure to Google Cloud and transitioning the AI model from OpenAI's GPT to Google's Gemini. This move wasn't a simple swap, as I was using the OpenAI Assistants API, which conveniently manages file uploading, vector databases, and conversation memory. For Gemini, I would need to implement this functionality myself.

I decided to complicate things a bit more and keep both the Gemini and GPT models, giving users the ability to dynamically switch between them. This is a neat feature not commonly seen in user-facing chat applications, and I was curious to see how hard it would be to implement.

The entire refactoring process took about three days, totaling around 15-20 hours of focused work. As is often the case, the last 10% of the changes consumed 90% of the time. Deploying to Google Cloud and implementing the dynamic model switching proved to be the most time-intensive parts of the project.

### What Went Well

The experience was, for the most part, very positive. I was particularly impressed by:

* **Holistic Code Refactoring:** The agents demonstrated a remarkable ability to propose and implement changes across numerous files simultaneously when prompted with a high-level refactoring goal.
* **Intelligent Debugging:** Having the Cursor agent read through logs to identify errors and suggest fixes was immensely helpful. It streamlined a process that is often tedious and time-consuming.

### What Didn't Go So Well

Despite the significant strides in AI-assisted development, there were several areas where the models still fell short:

* **Knowledge Gaps:** The models often lacked up-to-date information about API features. For instance, I had to explicitly inform the models that the Gemini API included a caching option.
* **API Versioning Issues:** The AI models have a hard time dealing with APIs that have different versions. On several occasions, I had to read the documentation myself and point the LLM to the correct URL. I anticipate that maintaining updated, machine-readable documentation for coding agents will be an important use case soon.
* **Context Loss:** During longer refactoring sessions, the models would lose context, and I found myself repeatedly reminding them of previously discussed details.
* **Tendency to Overcomplicate:** Models still default to overcomplicated solutions. Thanks to my experience, I was able to sidestep most of these proposals, but I can see how this could lead to convoluted and unnecessary code for less-seasoned developers. Even in my case, I had to prompt the models to clean up and delete unused code after we got to a working solution.

### The Verdict

All that said, this is a huge improvement in developer experience in only 18 months, and I was able to do a lot of work in just a few days. For my next iteration, I might refactor the Node.js backend to Python, since I have no reason to maintain the Node.js approach and Python APIs are usually updated sooner.

I invite you to try out the new and improved [Xavibot](https://amatriain.net/Xavibot/) and let me know your thoughts. Any other suggestions on what other features to add? Again, code is available [here](https://github.com/xamat/Xavibot)
