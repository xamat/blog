---
id: 128
title: "Beyond the Model: Making Sense of the GenAI and Agentic AI Stack"
date: '2026-10-18T00:00:01+00:00'
author: Xavier
permalink: /agenticstack
image: /blog/images/128-0.png
header:
  teaser: /images/128-0.png
categories:
- Artificial Intelligence
- Agents
- Infrastructure
reading_time:
    - ''
---

Back in October 2023, I wrote about [the multi-layered cake of GenAI development](https://amatria.in/blog/multilayer). My point then was simple: building a serious GenAI application involved much more than prompt engineering. Models, data, tools, agent design, product experience, and evaluation all mattered. Focusing exclusively on the prompt meant missing most of the system.

Three years later, we have a much richer ecosystem—and a much more confusing one.

We are now swimming in agent frameworks, agent platforms, coding agents, agent builders, agent runtimes, memory systems, and model gateways, alongside an impressive collection of products that describe themselves using several of those terms at once. Ask where a particular vendor belongs in the stack today, and the only honest answer is: *"Which part of their product page are we talking about?"*

It felt like the right time to revisit that original picture. Less cake this time, more architecture, and concrete pointers to how current tools map to actual engineering problems.

The core thesis behind this updated map hasn't changed, even if the capabilities have: **the useful unit of analysis is no longer the model. It is the system that turns model capabilities into reliable outcomes.**

(As most of my other recent posts, this one has been written with AI assistance. In this one in particular I used GPT 6.1 Sol and Gemini 4 Argon)


## A map of capabilities, not a shopping list

Here is my current view of the GenAI and agentic AI platform landscape:

Including specific company and project names helps make abstract layers concrete, but **the boxes are much more important than the logos**.

First, this is not a suggestion that you need to buy or build a distinct product for every row, nor does every application require every capability. A straightforward retrieval application does not suddenly become better because you bolt on a multi-agent orchestration framework and a long-term episodic memory store.

Second, do not read this diagram as a linear request sequence. A single user interaction might hit retrieval three times, invoke two external tools, pause for human approval, and never touch persistent memory. These layers represent architectural responsibilities, not chronological steps.

With those caveats out of the way, let’s unpack the runtime stack from the ground up.

## The runtime: from compute to experience

### Hosting and inference

Something still has to run all of this, and it helps to separate three related—but distinct—infrastructure jobs: agent hosting, model serving, and underlying serverless compute. Hosting a stateful agent that calls external APIs is a very different operational problem from serving model weights efficiently on GPUs.

That is why the diagram separates managed agent hosting platforms like [AWS AgentCore](https://aws.amazon.com/bedrock/agentcore/), [Google Vertex AI](https://cloud.google.com/vertex-ai), and [Microsoft Foundry](https://azure.microsoft.com/en-us/products/ai-foundry) from dedicated model-serving providers such as [Together AI](https://www.together.ai/), [Fireworks AI](https://fireworks.ai/), and [Hugging Face](https://huggingface.co/). Meanwhile, platforms like [Modal](https://modal.com/resources/best-serverless-platforms-hosting-ai-agent-backends) highlight a third vital category: serverless CPU/GPU compute purpose-built for custom inference, agent backends, and bursty AI workloads.

### Foundation models

Providers like [OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/), [Google](https://deepmind.google/models/), and [xAI](https://x.ai/) supply frontier proprietary models, alongside an increasingly capable ecosystem of open-weight models from [Meta](https://ai.meta.com/) (Llama), [DeepSeek](https://www.deepseek.com/), [Qwen](https://qwenlm.github.io/), and others.

Model selection—including the trade-off between closed APIs and self-hosted or managed open-weight models—remains consequential. However, it should always be evaluated against your specific tasks, latency budgets, cost constraints, data-privacy requirements, and risk tolerance, rather than a model's latest position on a general academic leaderboard. A frontier model that tops a reasoning benchmark can easily be the wrong choice for a high-throughput production workflow where a smaller open-weight model excels, and vice versa.

### Model gateway and routing

As soon as an architecture relies on more than one model or provider, you need a control plane to govern how requests reach them. Gateways such as [LiteLLM](https://docs.litellm.ai/), [Portkey](https://portkey.ai/), [Kong AI Gateway](https://konghq.com/products/kong-ai-gateway), and [Vercel AI Gateway](https://vercel.com/ai-gateway) provide a central choke point for access management, dynamic routing, fallbacks, and token budgets.

This layer is also a prime example of why company names alone fail as architectural categories: [Vercel’s AI SDK](https://ai-sdk.dev/) sits at the very top of the stack in application experiences, while its AI Gateway sits near the bottom. Same company, entirely different architectural responsibility.

### Knowledge and RAG

Models need access to information that is current, domain-specific, and authorized for the task at hand. Providing that is not a single database call; it is a pipeline spanning ingestion, parsing, chunking, embeddings, retrieval, reranking, and context assembly—often combining internal corpora with live web retrieval. Dumping raw PDFs into a vector store addresses only a fraction of the problem.

Players like [Unstructured](https://unstructured.io/), [Cohere](https://cohere.com/), [Pinecone](https://www.pinecone.io/), [Qdrant](https://qdrant.tech/), [Weaviate](https://weaviate.io/), [Elasticsearch](https://www.elastic.co/elasticsearch), and AI-native web search and retrieval providers like [Parallel](https://parallel.ai/) each tackle different stages of this pipeline (see [Cohere’s retrieval walkthrough](https://docs.cohere.com/page/rag-with-chat-embed) for a solid example of how the pieces connect). Ultimately, however, the system questions matter far more than which index or search API you pick: *Did we retrieve the right evidence? Is it fresh? And is this specific user permitted to see it?*

### Memory and session state

While RAG handles external knowledge, memory answers a different question: *what should the system retain from its ongoing interactions?*

Here, conflating different timescales into a single undifferentiated store is asking for trouble. You have immediate conversational state, the working state of an unfinished multi-step task, and long-term episodic or preference memory across sessions. Tools like [Mem0](https://mem0.ai/), [Zep](https://www.getzep.com/)/[Graphiti](https://github.com/getzep/graphiti), and [AgentCore Memory](https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-agentcore-memory-building-context-aware-agents/) focus on this layer. But "remembering more" is a poor design objective on its own. Good memory architecture cares just as much about what to forget, when state expires, and how a user or administrator can inspect and correct stale assumptions.

### Prompts and context assembly

Prompting hasn't disappeared; it has matured into **context engineering**. System instructions, retrieved evidence, tool schemas, interaction history, and user memory all compete for finite attention and context window budgets. The real engineering challenge is no longer wordsmithing a clever static prompt—it is dynamically assembling the exact right information at each step of a task.

[DSPy](https://dspy.ai/) represents the programmatic optimization side of this space, while prompt-management suites inside [LangSmith](https://www.langchain.com/langsmith), [Braintrust](https://www.braintrust.dev/), and [Langfuse](https://langfuse.com/) handle versioning, collaboration, and operational rollout. They are valuable operational tools, provided we remember they don't replace core context design.

### Tools, connectors, and action

This is the inflection point where a system stops merely generating text and starts changing state in the outside world.

[Composio](https://composio.dev/) handles third-party SaaS integrations, [Browserbase](https://www.browserbase.com/) enables headless browser interaction, [E2B](https://e2b.dev/) and [Modal Sandboxes](https://modal.com/resources/best-serverless-sandboxes-ai-code-execution) provide isolated code-execution environments, and [AgentCore Gateway](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html) helps standardize how tools are exposed to agents.

Architecturally, the critical distinction here is between **capability** (being able to call a tool) and **authorization** (being permitted to take an action). Giving an agent an API connector to issue a refund is easy; enforcing strict boundaries around *which* refund, *for whom*, *under what policy*, and *with whose human approval* is where the real engineering lives.

### Orchestration and multi-agent coordination

Once a system can take actions, you need orchestration to govern how work progresses: what runs next, what executes in parallel, when to retry, when to bail out, and when to hand control back to a human.

Code-first frameworks like [LangChain](https://docs.langchain.com/oss/python/langchain/overview)/[LangGraph](https://www.langchain.com/langgraph), [LlamaIndex](https://www.llamaindex.ai/), [CrewAI](https://docs.crewai.com/), [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/), [Google ADK](https://adk.dev/), and [Strands Agents](https://strandsagents.com/) occupy different points along the autonomy spectrum, alongside visual workflow engines like [n8n](https://github.com/n8n-io/n8n) that both design and execute pipelines.

Do you actually need a multi-agent architecture? Maybe. But I always look for a justification grounded in clear task decomposition, context isolation, or fundamentally distinct security boundaries. Splitting a straightforward workflow into five chattering personas usually just gives you five new places to debug without improving the outcome.

### Voice

Voice earns its own dedicated layer in the map because audio fundamentally alters interaction constraints. Interruptions, barge-in handling, silence detection, turn-taking, background noise, and sub-second latency immediately become make-or-break product concerns. A two-second pause that feels completely natural in a chat UI makes a voice conversation feel broken.

Here, players like [ElevenLabs](https://elevenlabs.io/docs/overview/intro) and [LiveKit](https://docs.livekit.io/agents/) illustrate complementary, overlapping responsibilities: speech synthesis and voice-agent logic on one side, and low-latency realtime media transport and orchestration on the other.

### Experiences

At the top of the runtime is what users actually touch—and where we find out whether all the machinery underneath actually helps someone get a job done.

For teams building custom interfaces, toolkits like the [Vercel AI SDK](https://ai-sdk.dev/) streamline streaming UIs and state wiring. On the other end of the spectrum are vertically integrated customer-experience platforms like [Cognigy](https://docs.cognigy.com/ai/overview/about-cognigy-ai) and [Sierra](https://sierra.ai/), which bundle experiences together with voice, orchestration, and backend integrations. That overlap isn't a flaw in the taxonomy; it represents a core architectural decision between composing best-of-breed layers yourself or adopting an integrated platform.

## Building the agent is not the same as running it

While iterating on this diagram, it became obvious that "agent builders" was doing way too much work as a catch-all label.

On one hand, we have low-code/no-code environments for visually wiring workflows: [n8n](https://github.com/n8n-io/n8n), [Dify](https://dify.ai/), [Flowise](https://docs.flowiseai.com/), and [Langflow](https://www.langflow.org/). On the other, we have AI coding agents and development harnesses: [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview), [OpenAI Codex](https://openai.com/codex/), and [Google Antigravity](https://antigravity.google/docs/home/).

I pulled these out into a separate **Agent Design & Development** band at the top of the graphic. They are vital for authoring and maintaining the system, but they aren't necessarily sitting in the critical path of every production user request. (True to form in this ecosystem, the boundary is porous—n8n executes the workflows it authors, and tools like Claude Code and Antigravity expose SDKs that can be embedded into custom runtimes—but separating *authoring time* from *execution time* makes the architecture much cleaner to reason about.)

This distinction echoes what I wrote recently about [moving from coding to leading AI agents](https://amatria.in/blog/endofcode): as coding agents absorb more of the raw implementation work, human engineering leverage shifts decisively toward defining intent, constraints, system architecture, and verification. Generating code faster does not absolve us of designing the system.

## Three capabilities that cut across everything

The horizontal stack is only half the story. Evaluation, observability, and security aren't features you bolt on the week before launch; they cut across every single layer of the runtime.

### Evaluation and red teaming

In my 2023 cake post, evaluation was the "icing" holding the layers together. If anything, that understated its importance.

Rigorous engineering here requires curated golden datasets, regression suites, offline experiments, adversarial red teaming, and hard release gates—supported by platforms like [LangSmith](https://www.langchain.com/langsmith), [Braintrust](https://www.braintrust.dev/), [Arize Phoenix](https://arize.com/phoenix/), [Galileo](https://galileo.ai/), and [Promptfoo](https://www.promptfoo.dev/).

When evaluating agents, grading the final text response is completely insufficient. You have to evaluate the *trajectory*: Did the agent select the right tool with valid parameters? Did it respect approval gates? Did it recover gracefully when an API timed out, and did it know when to stop? **A polite, well-written response never compensates for an unauthorized or incorrect action.**

### Production monitoring and observability

Observability deserves its own seat at the table rather than being lumped blindly under "evals". Offline evaluation tells you how the system behaves on the scenarios you thought to test; production monitoring tells you what happens when real users, drifting data, and flaky downstream APIs collide.

In practice, you need visibility across three distinct altitudes:

* **Operations:** errors, uptime, latency, token consumption, and cost.

* **Quality and behavior:** grounding fidelity, tool-call accuracy, task completion, runaway loops, and human escalation rates.

* **Business outcomes:** actual issue resolution, conversion, user satisfaction, and net human intervention saved.

Tools like [Langfuse](https://langfuse.com/), [LangSmith](https://www.langchain.com/langsmith), [Braintrust](https://www.braintrust.dev/), [Arize Phoenix](https://arize.com/phoenix/), and [Datadog](https://www.datadoghq.com/) cover much of this ground, using [online evaluations](https://langfuse.com/docs/observability/overview) to score live production traces. Still, no tracing tool can define what "success" means for your product. A customer support agent can boast 99.99% uptime, low latency, and hallucination-free explanations while still failing to resolve a single customer problem. Standard infrastructure dashboards will report all green. That is not a working system.

### Security and governance

Model-level guardrails—such as [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails), [Lakera](https://www.lakera.ai/), [Bedrock Guardrails](https://aws.amazon.com/bedrock/guardrails/), and [Azure AI Content Safety](https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety)—remain essential for catching prompt injections, toxicity, and PII leaks.

However, agentic governance goes far beyond filtering input and output strings. It encompasses agent identity, least-privilege permissions, strict data boundaries, tool registries, human-in-the-loop approval policies, immutable audit logs, and kill switches. For an autonomous agent, the most critical security perimeter sits at the point of *action*, not the point of text generation.

## The most important part is the loop

If I had to highlight just one section of the entire diagram, it would be the continuous iteration loop at the very bottom:

**Production monitoring → Detect & investigate → Add failures to evals → Improve → Validate & release → Production monitoring.**

Say you discover in production that an agent enters an infinite retry loop whenever a specific payment API times out mid-transaction. Patching the immediate bug is table stakes. Extracting that exact failure trace and turning it into a permanent, reproducible test case in your offline eval suite is what prevents the next model upgrade or prompt tweak from reintroducing the bug three weeks later.

Crucially, "Improve" in that loop rarely means jumping straight to a bigger model or rewriting the system prompt. More often, the right fix is cleaner retrieval chunking, a tighter tool schema, explicit state tracking, a smarter retry policy, or an explicit human approval step. That is the real utility of having a clear architectural map: it helps you localize failures to the right system layer instead of blaming everything on the LLM.

(And to be clear, while **self-recursive improvement (SRI)** and online continual weight updates are top of mind across the industry right now, this production loop is something much more grounded. Even as agents increasingly help us diagnose traces, synthesize eval cases, and propose code or prompt fixes, a production system compounds in reliability through disciplined system engineering—better harnesses, tools, context, and eval-gated releases—long before it autonomously updates its own weights in live traffic.)

## Fine-tuning is an option, not an entrance requirement

That is also why I placed **Model Adaptation**—fine-tuning and post-training via tools like Hugging Face [TRL](https://huggingface.co/docs/trl/index)/[PEFT](https://huggingface.co/docs/peft/index), [Unsloth](https://unsloth.ai/), [Together AI](https://www.together.ai/), or [Fireworks AI](https://fireworks.ai/)—in a dashed, optional box outside the core runtime.

Post-training is powerful when you have a measured need to distill a smaller model for latency/cost, lock in a specialized output format, or teach domain-specific behavior. But if your system is failing because of stale knowledge, ambiguous tool definitions, or broken permissions, fine-tuning model weights is just an expensive way to avoid fixing your architecture.

## Wrapping up

The takeaway from this landscape is not that every team needs to go out and assemble a twelve-layer enterprise platform tomorrow. Quite the opposite: we need clearer mental models of the responsibilities an agentic system must fulfill, so we can make deliberate choices about where those responsibilities live.

Start with the user outcome. Build the simplest architecture that can reliably deliver it. Introduce new layers only when the problem demands them, and instrument everything so you can evaluate and observe what is actually happening in production.

The logos on this chart will inevitably churn. Vendors will consolidate neighboring boxes, categories will blur, and new abstractions will emerge. But the core lesson from that 2023 cake holds up: the model matters, and so does everything around it. The difference in 2026 is that our systems don't just talk—they act. And that makes the surrounding architecture, and our accountability for it, more important than ever.
