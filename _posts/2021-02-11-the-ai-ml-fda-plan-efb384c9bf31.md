---
id: 18
title: "The AI/ML FDA Plan"
date: '2021-02-11T00:00:00+00:00'
author: xamat
layout: post
permalink: /the-ai-ml-fda-plan-efb384c9bf31/
reading_time:
    - ''
    - ''
categories:
    - Uncategorized
---


In case you are not familiar with it, the [Food and Drugs Administration](https://en.wikipedia.org/wiki/Food_and_Drug_Administration) (FDA) is a federal agency of the Department of Health and Human Services. that is responsible for the control and supervision of food safety, drugs, vaccines, medical devices, and the likes. In recent times the FDA has been thinking about how to regulate medical devices that are driven or based on AI as a result of several recent efforts, they published their *“*[*Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan*](https://www.fda.gov/media/145022/download)*”* in January 2021. To understand this action plan, you need to go and read a number of related documents. I did that for you, and my hope is that this post summarizes the most important aspects of the FDA’s plan around Artificial Intelligence and Machine Learning. Make sure to read the last section for a surprise “ending” to it all.

I will start by saying that if you, like myself, come from a technical background, you might be coming into this thinking that the FDA is likely an old bureaucratic institution filled with old-school lawyers that know nothing about novel AI/ML approaches. I can neither confirm or deny that, but what I will say is that this particular action plan has very interesting aspects that I am sure you will agree with if you are working in applied AI in general, and AI for healthcare in particular. As a teaser, here is what the FDA’s vision is, in their own words:

> *“The ability for AI/ML software to learn from real-world feedback (training) and improve its performance (adaptation) makes these technologies uniquely situated among SaMD and a rapidly expanding area of research and development. Our vision is that with appropriately tailored regulatory oversight, AI/ML-based SaMD will deliver safe and effective software functionality that improves the quality of care that patients receive. “*

You probably have noted by now that this action plan is focusing in the so-called SaMD (Software as a medical device). SaMD is a piece of software that on its own performs a medical function. This is in contrast to SiMD (Software **in** a medical device) where the software needs to be embedded in a piece of hardware to perform its medical function.

The Action Plan that was recently presented is a continuation of the so-called *“*[*Proposed Regulatory Framework for Modifications to Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD)*](https://www.fda.gov/media/122535/download)*”* published in April 2019. The proposed framework included a number of questions for feedback from stakeholders that the current action plan takes into account and responds to by proposing a follow up plan in 5 different directions. So, in order to understand the current Action Plan, we need to first understand the proposed Framework.

### **The “Proposed Framework”?**

Traditionally, the FDA reviews medical devices through an appropriate premarket pathway, such as [premarket clearance (510(k)),](https://www.fda.gov/medical-devices/premarket-submissions/premarket-notification-510k "Premarket Notification 510(k)") [De Novo classification](https://www.fda.gov/medical-devices/premarket-submissions/de-novo-classification-request "De Novo Classification Request"), or [premarket approval](https://www.fda.gov/medical-devices/premarket-submissions/premarket-approval-pma "Premarket Approval (PMA)"). Generally speaking, these premarket approvals need to happen in 3 situations:

- A change that introduces a new risk or modifies an existing risk that could result in significant harm
- A change to risk controls to prevent significant harm
- A change that significantly affects clinical functionality or performance specifications of the device

However, this cannot be used in an AI/ML case where the “device” is expected to continuously improve as it gets more data and models are improved. The proposed framework addresses this particular concern as well as other requirements of AI-based medical devices.

In order to get there, it first defines a risk categorization framework based on: (1) the significance of information provided by the SaMD to the healthcare decision, and (2) the state of healthcare situation or condition, which identifies the intended user, disease or condition, and the population for the SaMD. See table below:

![](/blog/images/05-01.png)

The framework also defines 3 types of not mutually exclusive AI/ML-based SaMD modifications:

1. **Performance** (e.g. retraining, change in AI architecture…)
2. **Inputs used by the algorithm** (e.g. adding new input data types and sources)
3. **Intended use** (e.g. from a confidence score that is ‘an aid in diagnosis’ (drive clinical management) to a ‘definitive diagnosis’ (diagnose) or inclusion of pediatric population where the SaMD was initially intended for adults ages 18 years or older)

Finally, and very importantly, it describes a Total Product Lifecycle (**TPLC**) Regulatory Approach for AI/ML-Based SaMD illustrated below. The TPLC has 4 different regulatory components (i.e. touchpoints with the FDA): (1) A set of Good Machine Learning Practices, (2) a Premarket Assurance, (3) a Change Protocol Review Process, and (4) ways to define Real-World Performance. Let’s look briefly into these 4 components.

![](/blog/images/05-02.png)

1. **Quality Systems and Good Machine Learning Practices (GMLP)**

All SaMD, devices that rely on AI/ML are expected to demonstrate analytical and clinical validation, as described in the SaMD: Clinical Evaluation guidance (see figure below). Example considerations include: Relevance of available data to the clinical problem and current clinical practice; data acquired in a consistent, clinically relevant and generalizable manner; appropriate separation between training, tuning, and test datasets; and appropriate level of transparency (clarity) of the output and the algorithm aimed at users.

![](/blog/images/05-03.png)

**2. Initial Premarket Assurance of Safety and Effectiveness**

SaMD will require a “**Predetermined change plan**” that should include:

- **SaMD Pre-Specifications (SPS)**: manufacturer’s anticipated modifications to “performance”, “inputs,” or “intended use”
- **Algorithm Change Protocol (ACP)**: Specific methods in place to achieve and control the risks of the anticipated modifications in the SPS. See table below for more details on what is expected in the ACP.

![](/blog/images/05-04.png)

**3. Approach for modifications after initial review with established SPS &amp; ACP**

Besides the predetermined change plan, the SaMD also needs to specify an approach for modifications. The requirements depend on the concrete situation and whether the SPS and ACP above had been approved. See workflow below.

![](/blog/images/05-05.png)

**4. Transparency &amp; real-world performance monitoring of AI/ML-based SaMD**

Finally, the regulatory framework requires AI-based medical devices to address transparency and real-world performance. There are not many details of how that should be implemented in the framework, but mostly some examples on how to ensure transparency and monitoring. E.g.1. Transparency may include updates to FDA, device companies and collaborators of the manufacturer, and the public, such as clinicians, patients, and general users. E.g. 2. Real-world performance monitoring may also be achieved in a variety of suggested mechanisms that are currently employed or under pilot at FDA, such as adding to file or an annual report

Each of the sections above comes with a number of questions to the “stakeholders”. Besides, the document includes a number of real-world examples and scenarios including:

- Intensive Care Unit (ICU) SaMD
- Skin Lesion Mobile Medical App (MMA)
- X-Ray Feeding Tube Misplacement SaMD

### **Feedback**

After its publication, and for the following few months, the framework above received [133 public comments](https://www.regulations.gov/docketBrowser?rpp=25&so=DESC&sb=commentDueDate&po=0&dct=PS&D=FDA-2019-N-1185). Those came from large corporations such as GE Healthcare or Anthem, but also individuals. There are even anonymous comments. All of them are available in the public site.

![](/blog/images/05-06.png)

### **Pilot: AI-guided Cardiac Ultrasound**

As a way to test the framework in the real-world, the FDA applied it to [a pilot](https://www.fda.gov/news-events/press-announcements/fda-authorizes-marketing-first-cardiac-ultrasound-software-uses-artificial-intelligence-guide-user) for regulating a device developed by a startup called [Caption Health](https://captionhealth.com/). This device uses AI to guide ultrasounds, and to enable untrained registered nurses to perform ultrasounds at the same level of quality as trained specialists

![](/blog/images/05-07.png)

### **Digital Health Center of Excellence**

Also in the context of these AI/ML initiatives, the FDA started the [Digital Health Center of Excellence](https://www.fda.gov/medical-devices/digital-health-center-excellence) late 2020. The center includes many external collaborators from universities and other non-profits (not for profit companies as far as I can see). Its 3 listed objectives are:

- Connect and build partnerships
- Share knowledge
- Innovate regulatory approaches

### The Action Plan

Ok, so we are now that we understand the original framework, and the process, we are ready to go back to the January ’21 action plan. The plan includes a summary of the feedback received (“What they heard”), and a plan on what the next actions will be (“What they’ll do”), all of it grouped in 5 different areas:

1. **Tailored Regulatory Framework for AI/ML-based SaMD**

- **What they heard:** Many suggestions including re:Predetermined Change Control Plan
- **What they’ll do:** Update the proposed framework, including issuance of Draft Guidance on the Predetermined Change Control Plan for public comment

**2. Good Machine Learning Practices (GMLP)**

- **What they heard:** Strong general support for the GMLP + a call for FDA to encourage harmonization through consensus standards efforts.
- **What they’ll do:** Encourage harmonization of Good Machine Learning Practice development through institutions such as IEEE and AAMI

**3. Patient-Centered Approach Incorporating Transparency to Users**

- **What they heard:** Call for further discussion on how AI/ML-based technologies interact with people, including their transparency to users and to patients
- **What they’ll do:** Following up on recent Patient Engagement Advisory Committee meeting, next step: hold public workshop on how to support transparency to users and enhance trust in AI/ML-based devices

**4. Regulatory Science Methods Related to Algorithm Bias &amp; Robustness**

- **What they heard:** : Stakeholders described the need for improved methods to evaluate and address algorithmic bias and to promote algorithm robustness.
- **What they’ll do:** Support regulatory science efforts to develop methodology for the evaluation and improvement of algorithms, including for the identification and elimination of bias, and promotion of algorithm robustness (e.g. Centers for Excellence in Regulatory Science and Innovation (CERSIs) at UCSF, Stanford, and Johns Hopkins University)

**5. Real-World Performance (RWP)**

- **What they heard:** Stakeholders described the need for clarity on Real-World Performance (RWP) monitoring for AI/ML software
- **What they’ll do:** Work with stakeholders who are piloting the RWP process for AI/ML-based SaMD

To summarize, the action plan proposes to:

- Update the proposed regulatory framework in the AI/ML-based SaMD discussion paper, including issuance of a Draft Guidance on the Predetermined Change Control Plan.
- Harmonize development of GMLP through additional FDA participation in collaborative communities &amp; standards development efforts.
- Continue to host discussions on the role of transparency to users, including holding a public workshop on medical device labeling to support transparency
- Support regulatory efforts on methodology for evaluation &amp; improvement of algorithms, including for the identification &amp; elimination of bias, &amp; robustness and resilience of algorithms.
- Advance real-world performance pilots in coordination with stakeholders and other FDA programs, to provide additional clarity.

### A last-minute hurdle?

If you think everything you read until now makes sense, you will be surprised to read what happened next. In the middle of the chaos of Trump’s administration last days, the HHS published a request to exempt a large number AI devices from FDA regulation. This piece of news was pretty shocking for the community following these efforts and it is still unclear what it means, particularly since the new Biden administration will need to decide whether to approve or not the requested exemption (read more [here](https://www.statnews.com/2021/01/16/slippery-slope-territory-health-officials-propose-waiving-regulatory-review-of-medical-ai-tools/#:~:text=Days%20before%20leaving%20office,respiratory%20disease%20on%20medical%20images.), paywalled, or [here](https://medcitynews.com/2021/01/in-the-final-days-of-trump-administration-agencies-clashed-over-how-to-regulate-medical-ai/), free)

![](/blog/images/05-08.png)

In any case, these next few months promise to be really important and decisive for the whole space of AI in healthcare regulation. I look forward for the outcome of it all to provide a good avenue for aggressive innovation plus patient safety and medical quality.
