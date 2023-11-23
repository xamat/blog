---
id: 14
title: "The Memo\_Culture"
date: '2018-07-01T00:00:00+00:00'
author: xamat
##layout: post
permalink: /the-memo-culture-41a2debefe2d/
header:
  teaser: https://amatriain.net/blog/images/14-01.jpeg
reading_time:
    - ''
    - ''
categories:
    - Uncategorized
---

Some days back I was having a conversation about the importance of memos and documents in an agile/fast-moving startup company. The person I was talking with was surprised with my thoughts and asked me to send some links so they could follow up and share with others in the company. Unfortunately, after some searching around, I did not find anything online that was comprehensive enough to include all my thoughts. So, here I am trying to be good to the internet by filling in the void on a topic that is important but maybe not “cool” enough for tech leaders to spend much time sharing. To be clear, I don’t claim to be an expert on this, and, I would love to hear your feedback.

I have been leading technical teams for many years now, and something that I have learned to appreciate is the importance of… writing documents. Yes, you heard me right. I know what some of you might be thinking now: this guy has not read the Agile Manifesto and doesn’t understand that in-person meetings are much better than documentation or that code and tests are the best documentation for any piece of software. As a matter of fact, I have reached this conclusion after many years of trying many flavors of agile methodologies, both formal and informal. I now believe that having the **right** documentation is precisely one of the best tools to be agile and move fast. As Eric Ries describes in his [Lean Startup](http://theleanstartup.com/) book, the main goal of a past-paced organization is to maximize learning rate. And, in order to maximize learning rate, you need to use data and document your decisions.

But, you don’t need to trust my word on this though. The most agile and innovative companies that you can probably think of, from Netflix to Amazon, and also including Facebook, Google or the likes, they all use a flavor of what I will call “The Memo Culture”.

![](/blog/images/14-01.jpeg)

So, what is the memo culture? I use this term to loosely refer to company cultures in which generating some form of document or memo is an important part of that culture.

The most extreme and clear example is Bezos’ Amazon, where every meeting needs to be preceded by a short document written by the organizer of the meeting (see [here](https://www.cnbc.com/2018/04/23/what-jeff-bezos-learned-from-requiring-6-page-memos-at-amazon.html), for example). The document is shared right at the meeting and the first few minutes of every meeting are used for attendees to read the document. There is no expectation of anyone reading the document before the meeting. There are different goals of such process at Amazon:

- Make sure that any meeting that is called is actually needed. If you, as the organizer, are not willing to spend a few hours preparing a document it is very likely the meeting is a waste of time.
- Make sure the meeting has clear goals and focus
- Make sure everyone in the meeting has the right context before the discussion starts
- Use the memo as documentation of the meeting and its final resolutions

I have seen several variations of this particular culture. Most companies won’t really **require** a document for every single meeting, but rather establish a subset of meetings that do benefit from them. I find this to be a less extreme and more productive process. That said, it does have the downside of not strictly leading to a reduction in meetings.

Another variation that I like is sharing the document **before** the actual meeting. If you use an online collaborative tool like Googledocs or Quip, starting the discussion on the document before the meeting can be extremely useful. As a matter of fact, I have seen many times situations where the discussions on the comment threads in the doc where so useful that a meeting was no longer needed. The other benefit is that discussions on the doc give a clear preview to the organizer of what elements need more clarification and discussion during the meeting and avoid the need to start with an open-ended discussion to gauge general opinions. The obvious downside is that following this process you are not guaranteed to reduce the overall time spent on making a decision.

It is also important to address the two elephants in the room: **Powerpoint** and **meetings**.

I have seen organizations using Powerpoint (or equivalent) as their main documentation format. I hope I can easily convince you that this is plain wrong. Powerpoint is designed as a one-directional tool for a presenter to backup their presentation with visuals. Period. Any extension of this is likely to fail miserably. A set of slides, for example, tends to maximize visually appealing presentations and leave out many details that will be filled in by the presenter. Trying to reconstruct those details from only the slides is most of the time next to impossible. Also, slides are not a great vehicle for collaboration and online discussion. So, please, do your organization a favor and do limit the use of Powerpoint to external presentations where all the issues above do not matter that much. If you don’t trust me on the potential evils of Powerpoint, you can read [Wired](https://www.wired.com/2003/09/ppt2/), [New York Times](https://www.nytimes.com/2010/04/27/world/27powerpoint.html?src=me&ref=homepage), and [Harvard Business Review](https://hbr.org/2010/04/powerpoint-is-evil-redux) writing about it.

Regarding meetings, I know that some agile methodologies promote the value of “in-person communication and I agree there is some truth to this. However, it is also true that many organizations that attempt to become “agile” end up throwing their employees into a non-sustainable cadence of meetings. If your technical people are spending over 20% of their time on meetings, you probably have a problem (see [this article](https://dzone.com/articles/meetingsa-culture-of-excess-that-hurts-our-product) for some good thoughts on the problem with meetings heavy cultures). Collaborative documentation is a way to avoid this. Remember the Bezos rules for memos is precisely designed to avoid useless or costly meetings and to make them more useful. Yes, excessive useless documentation should be avoided, but the same is true for excessive useless meetings. And, the good news is that both these things tend to balance each other if managed right.

Documents used in the memo culture should be treated as “living documents”. This means that they should go through as many short iterations as possible during the initial phases. Those iterations should include comments and feedback from all stakeholders (see Jeff Bezos’ [recommendation](https://www.cnbc.com/2018/04/23/what-jeff-bezos-learned-from-requiring-6-page-memos-at-amazon.html) on how to write good memos). It is likely that the discussions on the document might lead to some meetings where those can be addressed and discussed in person with the relevant participants. Even once the initial iterations on the document are finalized, it should not be considered as “closed” since new revisions could come at any point in time.

This gets me to the important question of how to keep documents updated. I unfortunately don’t have a silver bullet for you here. I totally understand why developers prefer to keep documentation close and coupled to the code to avoid them drifting apart and updated. That is hard to accomplish with other kinds of documents. My suggestion here is to treat documents as an evolving entity and use versioning and change logs as much as possible. Furthermore, as you will see in the next section, some documents (e.g. postmortem or AB Test design) never become outdated. You should push your organization to write mostly evergreen documents and avoid as much as possible those that are likely to become easily outdated.

### Some example documents

As mentioned above, I am not a fan of enforcing a policy like Amazon’s where every single meeting should include a document. However, this does bring the follow up question of what meetings and what documents usually make sense. Here are some that I found useful. Do not treat this as an exhaustive list, but rather a collection of examples, which might be more or less relevant depending on your organizational details.

#### Software/system design doc

This document is used to describe a software component or system before it is implemented. It is important to understand that a design document should not attempt to capture all details in the final design but instead be used as a way to clarify that all people involved are on the same page before starting the project. This document should include the initial design agreement or “contract” and will then be updated as the project evolves to reflect the reality of the final design as closely as possible. Therefore, the document should be used as high-level technical reference (the low level reference being the code itself).

A software/system design doc will usually include the following sections:

1. Use cases
2. Requirements
3. Architecture
4. Tools/libraries/components
5. Risks/unknowns
6. Expected timeline

The software/system design document is used by many organizations. You will easily find many example of templates online designed by anyone from [NASA](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160011412.pdf) to the EU Commission. Keep in mind that you want to keep your document light, and alive. You do not need a 100 page document unless you are literally designing a rocket from scratch.

#### Build vs. buy doc

The goal of this doc is to document the pros/cons of building a piece of technology vs. acquiring it from a 3rd party. Note that because pros/cons of building inhouse are very related to the cost and complexity of the design, this document will be very connected to a Software/System Design doc. As a matter of fact, it could easily become a section of the document above. On the other hand, a build vs. buy document can also be written when design details of the inhouse system are not clear enough to write an independent document.

In any case, the build vs. buy documentation should include the following:

1. Supported use cases and high level requirements
2. High level description of in-house alternative (can be skipped if this is part of the system design doc)
3. Third party alternatives
4. Pros/cons of (2) and (3)
5. Decision and rationale

#### Post-mortem

As Malcom Forbes said “Failure is success if we learn from it.” and in a fast moving organization, as I described above, you want to keep your learning rate as high as possible. However, learning from failure is not so straightforward. You need to think about why things failed, analyze the details, and propose changes so the failure does not happen again. All those things are documented in the Post-mortem document.

A post-mortem document document usually includes the following information:

1. Date of the incident and timeline
2. Summary of the incident and impact
3. Root cause summary
4. Detailed description
5. What worked well/could have worked better
6. Corrective actions

See [this great post](https://medium.com/production-ready/writing-your-first-postmortem-8053c678b90f) for more details on the postmortem process as a whole and pointers to post-mortem templates.

#### AB Test design doc

Just as any software or system design process benefits from discussing and documenting the decisions, the same is true for AB tests. If you are in a data-driven organization I don’t need to convince you of the importance of AB tests. However, even organizations that are thoroughly convinced and executing on this mantra can be really bad about documenting decisions. I have seen people at companies debating endlessly about an AB test that happened a couple of years back and everyone remembers somewhat differently. I have also experienced miscommunication between PMs, engineers, and Data Scientists on how an ongoing AB test had been designed. Finally, and very importantly, I have also witnessed AB tests that are running without a clearly specified hypothesis and metrics! The AB test design doc addresses all of this and more.

The purpose of the AB test design doc is to make sure all the stakeholders are on the same page about what is being tested and how it is being tested. It is not very different from a traditional science experiment design document really. As always, the document should be initially considered as a vehicle for feedback and discussion and eventually evolve into being documentation that should be useful years from now.

Some of the sections an AB Test Design doc should include:

1. High level description of the goals
2. Hypothesis that are being tested
3. Primary and secondary metrics
4. Cohort selection and duration of the AB test
5. Engineering design or pointer to software/system design where this is described

See [this example](https://help.optimizely.com/Ideate_and_Hypothesize/Create_a_basic_experiment_plan) of AB Test design document from Optimizely.

#### AB Test results doc

Documenting the results of an AB Test is as important or more than documenting the AB Test design itself. That said, the AB Test results can be documented as an addendum to the AB Test Design doc. I generally would recommend that particular approach. Otherwise, both documents should definitely link to each other since they are tightly coupled and you want to avoid the overhead of copying and pasting the test design details into a new doc.

That being said, the goal of this doc/section is to explain the details of the results of test, document the final decision and its rationale. As such, the document should include the following:

1. Test result on primary/secondary metrics
2. Interpretation/discussion of the results
3. Final decision &amp; next steps

#### Data Analysis doc

The AB Test results doc is not the only data analysis document that you should be thinking about. As a matter of fact, documenting data analysis is, in my experience, of vital importance in a data-driven organization. The last thing you want is for people to be guessing at why a decision was taken a few months back now that nobody remembers the details of the data. Documenting data analysis also enables organizations to re-visit decisions in a much more rational way.

The Data Analysis doc is, in reality, a more general purpose document than the AB Test Results one. You can use it to document anything from exploratory data analysis using product data to market research. It is hard to standardize the format because its contents can be very diverse. That said, any data analysis doc should probably include:

1. Goal of the analysis
2. Description of the data analyzed including possible shortcomings, etc…
3. Data analysis results (of course with lots of pretty graphs and plots)
4. Recommendations

### Conclusions

I hope I have convinced you by now that writing the right kind of documents can increase your organization’s velocity and long-term success. To be clear, this is not an area I claim to be an expert on, so I would love to get feedback and other ideas from you.

#### More where this came from

This story is published in [Noteworthy](http://blog.usejournal.com), where thousands come every day to learn about the people &amp; ideas shaping the products we love.

Follow our publication to see more product &amp; design stories featured by the [Journal](https://usejournal.com/?utm_source=usejournal.com&utm_medium=blog&utm_campaign=guest_post) team.
