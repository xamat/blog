---
id: 126
title: "Beyond the Bot: Building a Multi-Agent Recomender for Actionable Intelligence"
date: '2026-04-02T00:00:01+00:00'
author: Xavier
permalink: /endofcode
image: /images/126-0.png
header:
  teaser: /images/126-0.png
categories:
- Artificial Intelligence
- Agents
- Software Engineering
reading_time:
    - ''
---


<img src="/images/126-0.png">

This is the third—and likely final—post in a series I’ve been writing over the past two and a half years on using AI for software development. The journey began with [my early experiences of how LLMs would change development](https://amatria.in/blog/aidevelopment), and continued 18 months later when I revisited those ideas during an [AI-assisted codebase refactor](https://amatria.in/blog/ai-code-refactor).

But this post is going to have a significantly different vibe. Instead of discussing AI as a helpful “co-developer,” my argument today is simple and perhaps a bit provocative: coding, as we know it, is ending.

To be clear, I do not mean that software engineering is disappearing, or that code no longer matters. Code will still exist, just as assembly still exists. But for most engineers, most of the time, code is becoming less of a language we write directly and more of an intermediate representation produced, inspected, tested, and modified by agents.

In other words, the important shift is not from “coding” to “not coding.” It is from writing syntax to expressing intent, constraints, architecture, and verification. That is the abstraction jump this post is about.

# The Arc of Abstraction: Letting Go of the Syntax

I took my first computer class when I was around 10. In hindsight, there was very little actual coding involved—mostly just doing calculations in binary and hexadecimals. My real love affair with code didn't start until college, when I finally found my mother tongue: C++.

I loved the combination of object-oriented design with the hardcore, low-level control inherited from C. I took this obsession to the absolute limit during my Ph.D. leading the CLAM project, where I dove into the darkest depths of template metaprogramming and signal processing optimization, eventually winning an international ACM prize.

**(Fun fact: I still proudly keep a copy of a C++ book signed by Bjarne Stroustrup, a prize I won for spotting an error in his slides during an advanced course years ago.)**

This was not a casual phase. I taught programming and software engineering in universities in both Europe and the US, and even while doing my postdoc I moonlighted writing professional C++ code for startups. For a long stretch of my career, code was not just a tool I used; it was one of the main ways I thought.

As my career progressed beyond academia—moving through Netflix, Quora, and the early days of building Curai—my day-to-day coding naturally decreased. I shifted to Java, then Python, but more importantly, I found myself moving steadily up the stack. The most valuable contribution was increasingly not writing the perfect class hierarchy or optimizing a memory pointer, but defining architecture, clarifying logic, setting strategic constraints, and building the teams that could turn those ideas into working systems.

For a long time, software engineering has been defined by our proximity to the machine syntax. But engineering is fundamentally about problem-solving, not writing boilerplate. My transition away from writing daily code wasn't a departure from engineering; it was just a step up the abstraction ladder.

And now, driven by agentic AI, the entire world is taking that exact same step.

# The Second AI Moment: My Agentic Sprint

If late 2022 and early 2023 was the first "AI moment" with the release of ChatGPT, the first half of 2026 will likely be remembered as the second. We have crossed the Rubicon from AI-assisted coding to agentic software development.

Claude Code became the symbolic center of this shift. I have to admit, it caught me by surprise. When I first saw a preview of Claude Code at Google Next back in April 2025, I was not impressed. I was unconvinced that developers would willingly go back to the CLI. Surely, I thought, you wouldn't want to ditch the comfort of visual IDEs to debug and supervise what an AI was doing! I was clearly wrong. In the last few months, agentic CLI and background execution have taken the world by storm, with alternatives like OpenAI’s Codex updates and Google’s Antigravity following suit.

To truly understand this frontier, I decided to do a coding sprint. But rather than using the standard Claude Code harness—which I already use on and off at work—I wanted to take a bit more risk and maximize my optionality. So, I went with OpenClaw.

OpenClaw was the tool that made the abstraction shift visceral for me. It gives me complete control over the orchestration layer. I hooked it up to both Codex and Gemini, meaning I can hot-swap models at any point without breaking my workflows. I even connected it to WhatsApp and Obsidian. Do I think OpenClaw is the definitive future of mainstream software development? No way. It has a very "Linux" vibe to it—incredibly cool for hackers and early adopters, but tools like Claude Cowork are already bringing these features to the masses via polished GUIs. Still, for a personal sprint, it was the perfect playground.

For this experiment, I tackled tasks across the complexity spectrum. On the everyday maintenance side, I fixed all the security warnings accumulated in my Xavibot project (a notoriously painful dependency-hell operation) and added a long-overdue like/comment feature to my Jekyll-based blog. But to truly test the limits of agentic abstraction, I also used this framework to build a [distributed recommender system](https://amatria.in/blog/agenticrecsys) from scratch. Architecting a multi-agent recommendation flow generally involves complex state management, data orchestration, and API design.

Here is where the paradigm shift hit me: Whether it was untangling old dependencies or architecting a distributed recommendation engine, I executed this entire sprint using only OpenClaw, interacting with it almost exclusively through WhatsApp messages. I only opened a local terminal for specific operations that required elevated permissions.

I did not look at a single line of code. I know this might get me in trouble with the purists. Anyone inspecting my repository will likely find some suboptimal implementations. I care, but less than I used to—and much less than the purists think I should. The resulting system—including regression tests proposed and written by the agent—was more robust than what I would likely have produced manually in a fraction of the time available. Not because the code was perfect, but because the loop was different: specify, deploy, observe, report symptoms, let the agent inspect and repair.

One reason we rely so heavily on IDEs is the tight feedback loop: code, run, debug, iterate. I assumed I would miss that visibility, but I didn't. Because I was communicating asynchronously via WhatsApp, I simply had the agent deploy to the cloud so I could test "the real thing" live. If something wasn’t working, I didn't pull up the logs or open a debugger. I just texted OpenClaw what I was experiencing on the front end, and it diagnosed and fixed the issue autonomously. It is hard to quantify exactly how long this sprint took because I never sat down for a dedicated "coding session." But from a practical, outcome-oriented perspective, it felt easily two orders of magnitude faster than doing it by hand.

# The Future of Software Engineering: From Code to Intent

A lot of software engineers are already writing significantly less code (or [no code at all](https://techcrunch.com/2026/02/12/spotify-says-its-best-developers-havent-written-a-line-of-code-since-december-thanks-to-ai/)), and it is my strong opinion that the large majority of them will very soon not even see a line of code in their day-to-day work. Just as looking at raw assembly became a rarity decades ago, the same will happen with high-level languages like Python, Java, or C++. Very soon.

For decades, source code was the canonical artifact of software engineering. Requirements were fuzzy, architecture diagrams were partial, tests were incomplete, but the code was the source of truth. In an agentic workflow, that hierarchy starts to invert. The durable artifacts become the spec, the constraints, the tests, the deployment environment, the issue reports, and the evaluation loop. Code still exists, of course, but it becomes an intermediate representation: something the agent manipulates on the way to a working system.

This does not mean the new workflow is risk-free. In fact, it shifts risk from syntax errors to specification errors. If your constraints are vague, your tests are weak, or your deployment environment is too permissive, agents will happily optimize toward the wrong outcome. The bottleneck moves from writing code to defining intent, setting guardrails, and verifying behavior. That is why deep technical knowledge still matters. You may not need to read every line of code, but you absolutely need to know what properties the system must satisfy. In that sense, the future engineer becomes less of a typist of logic and more of an architect of correctness.

Where does that leave us? Will software engineers be needed at all?

I think the answer is yes, and no. The kind of software engineer needed in a couple of years (or less) will be a hybrid between a software architect and an engineering manager who oversees a team of coding agents. We are already seeing the early signals of this in large enterprises, where the traditional lines between product managers and software engineers are blurring. Future engineers will spend much more of their time on the “what,” the “why,” and the “whether it actually works”: defining constraints, establishing product requirements, designing interfaces, setting evaluation criteria, and supervising execution. The agents will handle more and more of the “how.”

Does this mean it is a waste of time to study Computer Science? Absolutely not. In my opinion, Computer Science education is going to evolve into a foundational science, much like Mathematics. We don’t question the value of studying math, despite the fact that we seldom execute complex long division without the help of calculators or computers. You learn math to understand logic, structure, and problem-solving. Coding and software engineering education will progress in the exact same direction. Practical AI-assisted software engineering will be about orchestrating logic, understanding distributed systems, and architecting solutions, grounded in a theoretical understanding of how machines compute.

# Conclusions

We are experiencing a unique time in world history. Things are changing at a blistering pace, and the disappearance of manual coding as the center of software engineering is just one early signal.

So yes, I believe we are witnessing the end of coding as the central activity of software engineering. But this is not the end of software engineering. It is its next abstraction layer.

The engineer’s job is moving from producing code to producing intent: defining what should exist, why it should exist, what constraints it must satisfy, and how we will know whether it works. That inevitably pulls software engineering closer to product management and team leadership. The difference is that the future engineering team may include far more AI agents than humans, and the engineer’s role will be to orchestrate, constrain, evaluate, and ultimately own their output.

I spent decades learning to speak to machines through code. This year, for the first time, I felt that I could stop speaking their language directly.

Goodbye coding. Hello intent.
