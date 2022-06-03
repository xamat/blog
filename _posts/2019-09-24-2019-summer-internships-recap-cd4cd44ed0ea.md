---
id: 7
title: '2019 summer internships recap'
date: '2019-09-24T00:00:00+00:00'
author: xamat
layout: post
guid: 'http://localhost:8080/wordpress/?p=7'
permalink: /2019-summer-internships-recap-cd4cd44ed0ea/
reading_time:
    - ''
    - ''
categories:
    - Uncategorized
---

### 2019 summer internships recap

This summer, we had a great class of interns at Curai. We accept interns on all our different teams all year round. As a matter of fact, we currently have a few interns for the fall and winter. This post focuses on the work done by the wonderful engineering interns we had over the summer. We will update this blog with future intern work, and you should look forward to reading publications from other interns in the coming months.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>In this post, we include work done by Markie and Andrea, who study at Stanford, Dylan and Mesi from MIT, Kyle from CMU, and Joseph, an MD and CS student at UPenn. Of course all of this work would not have been possible without the great work of their mentors in the engineering team, other colleagues from other teams (e.g. medical team) who teamed up with them in the projects, and those who volunteered to help edit this post. What follows is an edited short version of their work in their own words (among other things, our interns were so proud with our mission to “**Provide the world’s best healthcare to everyone**” that they all mentioned it when describing their work). Enjoy reading about their work, and, please send us your resume at <jobs@curai.com> if you are interested in helping us with our mission by doing an internship or taking on any of our [other positions](https://curai.com/careers).

### 1. Doctor Speech-to-Text

*Dylan Doblar (MIT)*

Over the course of this summer, I’ve had the pleasure to work as an Engineering Intern at [Curai](https://curai.com). Under the guidance of my mentor, Eric An, I’ve built a Speech-to-Text microservice to support doctors when interacting with patients on our chat-based platform, [Curai Health](https://www.curaihealth.com). This feature aims to increase doctor efficiency by providing real-time speech-to-text transcription functionality that allows doctors to dictate messages to patients. To serve its purpose, we decided that Doctor Speech-to-Text should be:

1. Accurate, especially with medical terms
2. Easy to use and seamlessly integrated into the product
3. Highly scalable and computationally feasible
4. Fast, with near real-time feedback
5. Secure, ensuring that patient privacy is maintained

We considered using pre-trained models (from IBM, nVoq, Microsoft, and Google) as well as the possibility of training our own model. We settled on Google Cloud’s Speech-to-Text engine because it demonstrated that it could recognize medical terms reasonably well during our experimentation, provided bidirectional streaming with intermittent results, and could scale to support all of the doctors in our global doctor network.

Enabling real-time transcription would require bidirectional streaming between the doctor’s browser and Google’s servers. To stream audio input to Google’s servers and receive responses, we needed to send chunks of audio via gRPC. Since modern browsers do not support gRPC, we set up our own server to act as middleware connecting the doctor’s browser to Google’s servers. Since privacy is a primary concern for this feature, we ensured that audio is only recorded at the doctors’ instruction. Additionally, the streaming of audio from the browser to our server and from our server to Google’s servers is done in a HIPAA compliant manner, and no audio files are ever saved.

For purposes of scalability, we decided to make Doctor STT a microservice instead of rolling its functionality into our main API server. This allows the resources allocated to this feature to grow independently of the rest of the platform. Although we initially built our middleware server using the Flask setup of all of our other microservices, we realized that an Express server was better suited for the task due to Node.js’s asynchronous, event-driven nature and graceful handling of multiple concurrent requests. One complication to this flow of information is related to audio encodings. Browsers record audio in a format that is not interpretable by Google’s STT engine, so we developed a scheme to transcode chunks of audio on the fly in the browser.

With this setup, we were able to achieve all of our desired feature characteristics. Google’s STT engine performs sufficiently well on medical terminology, the recording capability is built directly into the web app and is available at the click of a button, the microservice architecture enables efficient scaling, and our bidirectional streaming scheme makes the feature fast with near real-time feedback in a secure, privacy-preserving manner. I’d like to thank my mentor, Eric An, along with everyone else on the Engineering team for all of the help and advice that contributed to this project’s success.

### 2. Medical Embeddings

*Andrea Dahl (Stanford)*

I am Andrea, a rising Junior CS student at Stanford. This summer I worked on a very exciting NLP project at Curai. For some context, [this post](https://medium.com/curai-tech/nlp-healthcare-understanding-the-language-of-medicine-e9917bbf49e7) describes many of the NLP efforts we work on at Curai. Many of our NLP modeling tasks are reliant on the quality of the word embeddings that are incorporated into them. Initially we saved time by utilizing high quality pre-trained embeddings like BERT embeddings. Such embeddings are trained on generic English corpora such as the Wikipedia database or the Google News dataset (about 100 billion words) and are great for representing word use in language that is similar to those texts. However, to model the conversations between patients and doctors, such embeddings are not perfect. Thus, we decided to train custom word embeddings on Curai Health conversations in order to boost the performance of existing and future conversation models.

Curai Health is a conversation platform for patients to message with doctors. I trained embeddings on messages sent from patients, messages sent from doctors, and entire conversations including both parties. Messages sent from doctors are interesting because while doctors have the domain knowledge to use specific medical terminology, in Curai Health, they are communicating with users and have to write in a way that is understandable by non-medically trained laypeople.

In order for machine learning models to do math on language, words must be represented numerically. Therefore, we transform words into word embeddings which are high dimensional vector representations of words where words that are more similar have similar vector representations. These vectors can even be plotted so that we can see the spatial distance between different words. In order to learn the best vector representations for words, different embedding methods evaluate words at different levels. For instance, embeddings can consider language at the character level, subword level, word level, and sentence level, as well as with and without context.

Tokenization is the method of splitting sentence inputs into smaller strings. Since many medical terms are actually phrases, i.e. head pain or stomach ache, we hypothesized that learning embeddings on multi-word tokens would produce better performing embeddings than single-word tokens. One can imagine, if writing a sentence about cities, the embeddings for “new” and “york” would be an insufficient representation compared to the embedding “new\_york”. This method is similar to trigrams or bigrams but has a smaller and more manageable vocabulary because it only saves the most commonly appearing phrases. For instance, for the sentence “I have a runny nose”, a traditional bigrams tokenization method would add \[“i\_have”, “have\_a”, “a\_runny”, “runny\_nose”\] to the vocab, but this method might tokenize just to \[“i”, “have”, “a”, “runny\_nose”\]. The specific tokenization package I used is [Gensim Model Phrases](https://radimrehurek.com/gensim/models/phrases.html), which keeps any trigram and bigram phrases that surpass a certain probability threshold as well as all unigrams in each sentence.

After training embeddings on patient and doctor language, I was able to observe the differences in words that the two groups use and the way they use them. A lot of the terms that patients commonly use that don’t show up anywhere in doctor text are actually misspellings of words, rather than unique words. Below are some examples:

‘stsrt’, ‘circler’, ‘oncall’, ‘poofy’, ‘tep’, ‘parentheses’, ‘caked’, ‘sprry’, ‘playdoh’, ‘swallowinh’,

Interestingly, the embeddings still seem to be able to make sense of the misspelled words correctly. The nearest neighbors of misspelled words reflect an understanding of what the word is supposed to be.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure><figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>We can also observe what the embeddings look like in 2D space. Clearly the FO Word2Vec model on patient data here is capturing the correct meaning of these words and understands the context in which they exist. Here, words most similar to the words in the key (nearest neighbors) are plotted.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>Some of the terms that show up commonly in doctor text but don’t appear anywhere in patient text are highly specific medical terms, descriptions, or instructions. Below are some examples:

‘decomposed’, ‘acne\_rosacea’, ‘ceana’, ‘precipitates’, ‘hamartoma’, ‘emergency\_helpline\_number’, ‘antifungal\_dusting\_powders’, ‘essential\_minerals’, ‘silla’, ‘mobilizing’, ‘emotional\_states’, ‘counter\_tynelol\_advil’, ‘acidic\_sour’, ‘connective\_tissues\_attachments’, ‘premenstrual\_syndrome\_includes’, ‘neti\_pot\_plain’, ‘articles\_php’

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>Out of medical terms that appear in both corpuses, these words have embeddings that are far apart in the two embedding spaces, which could mean that they have different connotative meanings when used by doctors and patients. Below are some examples:

‘myself’, ‘email\_address’, ‘provide\_temporary\_relief’, ‘applying\_ice\_packs’, ‘triptans’, ‘cool\_wet\_cloth’, ‘eating\_smaller\_portions’, ‘various\_reasons’, ‘license’, ‘wearing\_loose\_clothes’, ‘surely’, ‘bodily\_fluid’, ‘scope’, ‘support\_network’, ‘relievers’, ‘real’, ‘character\_limit’, ‘fiber\_foods’, ‘killer’, ‘sounds\_familiar’,

What is surprising is the mix of specialized phrases like “wearing\_loose\_clothes” along with very everyday words like “basics” and “okay”. It shows that in many areas, doctors and patients utilize language differently.

After training embeddings for patient speech and doctor speech, we realized there were valuable insights to be had on how different these “languages” are from each other. Since the embeddings were trained using the same Word2Vec method and since the vocabularies have a high degree of overlap, in theory, their shapes should be pretty similar and a linear transformation of one shape could align it with the other. I used the [MUSE](https://github.com/facebookresearch/MUSE) supervised model code to align the embedding spaces in different ways to further explore the differences in how patients and doctors write. Specifically, the MUSE method involves an iterative [Procrustes](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem) method that defines the linear transformation applied to the source matrix. There is also an unsupervised GAN method that was less effective and that was not used for our data.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>Aligning of Italian and English embedding spaces from MUSE paper

We are able to tell the quality of these embeddings by comparing the locations of some medical words that have layperson translations in the embedding spaces before and after alignment. Below, before alignment, the medical terms in blue are separated spatially from their layperson translated terms in red. However, after running the alignment method, we are able to see that the embedding spaces shifted, and that the vocabulary in these embedding spaces spatially match up in a logical way.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure><figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>The embedding and text translation work I completed this summer at Curai will contribute to the NLP efforts of the ML team at Curai moving forward. Having never taken an NLP class in school, I learned an immense amount this summer by working on these individual projects as well as by contributing them to a larger team and codebase. It is incredibly rewarding to see that this work helped me grow as an engineer and will also benefit Curai’s machine learning team as well.

### 3. Interpreting natural language responses in Doctor-Patient Conversations

*Markie Wagner (Stanford)*

My project, Natural Language Next Question, focused on understanding doctor-patient conversations and extracting structured data from unstructured freeform text.

*Background*

When a patient first enters a chat, they are required to input their chief complaint. Our Next Question engine then generates a series of follow up questions for the patient, which ask the patient whether or not they have a symptom. For example, if the patient comes in and complains of a fever, the Next Question engine might ask, “Do you have a headache?” Currently, a patient can only respond by clicking either the yes, no, or unsure button. This is a missed opportunity to get additional information from users, who often want to contribute more information, such as the severity or duration of the symptom. This is also a less than ideal user experience, since a conversational dialogue system allows the patient more freedom, as opposed to being forced to choose between three options.

My goal for the summer was to build machine learning models that could understand free-text responses to the doctor’s symptom questions. The motivation behind this project was to improve the question answering experience, allow for more detailed findings, enable the diagnostic engine to output more relevant questions, and provide a path to fully automate the conversation. These models can interpret a patient’s response and extract information such as the duration, frequency, severity, and confirmation/negation of a symptom.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>*Using BERT To Detect Symptom Confirmation*

Our Next Question engine generates questions that follow the format ‘Do you have \[symptom\]?”. To replace the functionality of the radio buttons, I created a model that could extract, from a patient answer, whether or not they were saying “yes,” “no,” or “unsure” in their text response. This turns out to be a tricky problem since over 40% of users do not use a yes/no/unsure phrases (e.g., “yeah,” “no,” “nope,” or “not sure”) in their responses. Most of the time, one must infer whether the user is saying yes or no.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>In order to understand whether a patient was confirming or denying the presence of a symptom, I experimented with several of my own feature engineering strategies and popular NLP models. In my initial attempts, I tried to make use of publicly available question answering data, such as [Amazon’s Question Answering dataset](http://jmcauley.ucsd.edu/data/amazon/qa/), but these models struggled with such a domain-specific task. I ended up using a combination of weak supervision and Mechanical Turk labeled data to create my dataset.

After running experiments with several sequence classification models, I found the most success with the use of the [BERT](https://arxiv.org/abs/1810.04805) and [XLNet](https://arxiv.org/abs/1906.08237) language models released by Google. I ended up using [Huggingface’s implementation](https://github.com/huggingface/pytorch-transformers) of these transformer models in Pytorch. These models worked particularly well on small datasets because they leverage the use of transfer learning. BERT and XLNet are first trained on a large-scale dataset and then finetuned on a smaller dataset for a more specific target task. In my case, I was mapping sequence (doctor question, patient answer) to binary classification (patient confirming, patient denying.)

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>*Using Conditional Random Fields To Extract Severity, Frequency, and Duration*

For collecting the severity data, I decided to use a combination of the [hNLP dataset](https://healthnlp.hms.harvard.edu/center/pages/data-sets.html) (annotated de-identified clinical notes from several institutions) and weakly supervised labels from our own patient chat logs. The weakly supervised labels were generated using a hierarchy of labeling functions, which consisted of hardcoded heuristics, syntactics (specifically, [Spacys](https://spacy.io/api/dependencyparser/) dependency trees), and more. I also ended up bootstrapping more data from our chat logs using noisy manual labels from Mechanical Turk.

For the temporal dataset, I didn’t end up using data from our own app chat logs, since I had access to a small but high-quality dataset with annotated temporal entities. To further improve performance, I decided to resample from the specific categories I saw my validation set consistently struggling with.

For this entity extraction task, I found that [Conditional Random Fields](https://en.wikipedia.org/wiki/Conditional_random_field) enjoyed the greatest success. CRFs can do quite well even with limited datasets, and these models cover the majority of cases where the severity and temporal entities appear.

*Results*

In the end, I was able to create models that could consistently and accurately collect natural language next question data.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>*Integration*

At the end of the internship, I had to switch gears from research to product engineering and integrate API calls for each model into the backend. Curai uses Kubernetes, Prometheus, and recently [switched over to Helm](https://medium.com/curai-tech/constant-vigilance-a-step-by-step-guide-to-alerts-with-helm-and-prometheus-ae9554736031), so I had to learn how to use these technologies in order to successfully integrate, test, and productionize my models.

*Conclusion*

This project was a unique and gratifying learning experience because it was so interdisciplinary. I had to take into account the clinical ramifications, the actual engineering challenge of end-to-end integration, the data engineering aspect, as well as the ML research available to build the models. For those looking for your next summer internship or full-time position, I cannot recommend this opportunity highly enough.

### 4. Summer Internship Experience From A Doctor Turned Software Engineer

*Joseph Liu (UPenn)*

Hi. I’m Joseph, a first year master’s in CS student at the University of Pennsylvania. Previously I spent several years working as a doctor in the UK before making the career switch. This summer, I had the pleasure of utilizing both backgrounds in a software engineering internship at Curai. During my internship I worked on two projects: (1) Explaining diagnosis results with LIME, NS (2) Grouping findings based on similarity. I even spent some time combining both projects. However, in this post I will focus on the use of LIME for medical explanations, a project that required combining my understanding of software, machine learning, and medicine.

*Explaining Diagnosis Results With LIME*

Curai is working on developing state-of-the-art diagnosis algorithms (see [this post](https://medium.com/curai-tech/the-science-of-assisting-medical-diagnosis-from-expert-systems-to-machine-learned-models-cc2ef0b03098) in the Curai tech blog for more details). How it works almost seems like magic: a patient inputs a set of symptoms, answers a few questions and poof, just like that, the engine spits out a differential diagnosis.

But the question is, why should we trust these results? At no point has the engine explained its results and thus, given us a reason to trust it. When the stakes as high as they are in healthcare, it’s paramount that the user gets visibility into the analysis behind the predictions. Thankfully, with LIME we can do just that.

LIME stands for Local Interpretable Model-Agnostic Explanations. Put simply, it means it’s able to explain any black box classifier. It doesn’t matter if we’re dealing with a diagnosis model, a spam email classifier or a model that classifies an image as hot dog or not — LIME can do it!

In the following example, I’ve simulated a patient who’s having an allergic reaction due to a bee sting. After typing in our symptoms, our diagnosis model outputted the following diagnoses:

1. Anaphylaxis (fancy way of saying “life-threatening allergic reaction”)
2. Bee Sting
3. Insect Bite
4. Allergic Vasospastic Angina

All four diagnoses are related to either allergic reactions or bee stings. So far so good. But as the doctor overlooking these results, I would love to know which symptoms contributed most to each of these diagnoses. Let’s see what LIME had to say:

For anaphylaxis, we can see that out of the three symptoms, bee sting and upper lip swelling was very predictive for this diagnosis, while crying was not. That’s reassuring to know because crying isn’t a cause of anaphylaxis (unless you happen to be allergic to crying!)

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>Or take bee sting. We can see that bee sting causes bee sting (no duh!), while upper lip swelling and crying do not cause bee sting. Talk about stating the obvious! But it is precisely these simple conclusions that lets us know the model is on the right track and hence, gives us more reason to trust the results!

We used the LIME text classifier module; we converted the input (a list of symptom objects) into a single comma-delimited string (e.g. “upper\_lip\_swelling, bee\_sting, crying”) and created a wrapper function that parses the string and calls our own classifier function internally.

The output of the wrapper function needs to return probabilities for each label or disease. However, our classifier only outputs probabilities for diseases where the probability is greater than zero. So to fit it onto LIME, we had to include probabilities even when it was zero. That meant including the probability for every disease in our knowledge base.

One of the many benefits of working at Curai is the ability to wear many hats and interact with different kinds of people. The team is split into different squads, and each squad is multi-disciplinary: there are clinicians, product engineers, data scientists, machine learning researchers and product managers.

This setup is perfect who is a “jack of all trades” kind of person like myself and this internship had it all: I got to do a bit of back-end, front-end, machine learning, clinical informatics and even algorithms (as it turns out, all that technical interview prep was useful after all!)

But the best part of it all? The work you do will be seen in the final product. After my final presentation, it turns out a way to group findings is a feature users have been asking for recently. And really, there is nothing more satisfying than that — knowing that your work will have impact.

### 5. Internationalization and engineering improvements

*Kyle Chin, CMU*

Hi I’m Kyle and I’m writing about my experience as a software engineering intern at Curai this summer. As an early stage startup, there was (and is) no shortage of tasks to do week-to-week, and as a result I worked on a variety of different projects during the course of my 11-week internship. Enjoy!

*Internationalization &amp; Real Time Chat Translation*

My first project at Curai was to start the groundwork on internationalization (i18n). This means supporting the Curai Health App in different languages based on the user’s language preferences. For example, a button that says “Hello” by default should say “Hola” if the user’s language is set to Spanish. Supporting non-English languages is essential to the goal of offering the Curai Health App globally.

The first part of this is choosing and implementing an i18n framework. The three essential requirements we decided for any i18n library were that it must be scalable, flexible, and maintained. After researching various libraries, we decided to move forward with[ i18next](https://github.com/i18next/i18next), a popular i18n javascript library. A couple benefits that stood out were that it was well documented and maintained, and its implementation in the Curai Health code was straightforward and clean.

Since the Curai Health app is a chat app, in the long term, i18n also means facilitating conversation between a doctor and a user who may not be fluent in a common language. Ideally, users would be able to be matched with doctors who speak their native tongue, but given the economies of supply and demand (especially with respect to doctor response time) this will not always be possible or optimal. So, the second part of this project was to prototype a multilingual chat — so a non-english speaking user could chat with an english-speaking doctor.

This was fairly straightforward to implement. First, a “Preferred chat language” column needs to be added to a patient’s profile. Then, non-english messages coming from the patient need to be translated to english, and messages from the doctor need to be translated to the patient’s chat language. In either case, the language modeling and other NLP features that FirstOpinion utilizes need access to english text, so *both the english text and raw text are stored with messages*.

*Front-End Context Refactor*

At any fast-moving company, and even more so in one like Curai that acquired another startup, [cruft](https://en.wikipedia.org/wiki/Cruft) accumulates. A couple of inconveniences that existed in the frontend (React) codebase were:

1. Router.js, a file that was simply supposed to select the right component to display based on the URL, stretched to over 750 lines of code because it also handled the websocket connection to the chat in addition to a variety of other smaller tasks.
2. Prop Drilling: API functions to talk with the backend were all instantiated in Router, and passed down to components. For components that are low in the tree, this meant passing in the functions through as many as eight (!!) other components to get the function down to the one component that needed it.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>In the old architecture, Router had logic for websocket connection even if the component displayed was not using the websocket. In addition, api calls and informational props all came from Router. This meant if component B needed to make an api call, the function would need to be passed down through component A and Chat even if those components did not need it. To address these issues, I:

1. Moved the chat’s websocket handling to a separate component and out of Router.js
2. Created[ React contexts](https://reactjs.org/docs/context.html) for services like API calls so that components that needed them could subscribe to the context directly, as opposed to receiving them through prop-drilling.
3. Fetched and stored data relating to global state (such as the patient’s data) in a lightweight, custom state manager dubbed `Weedux` .
4. [Hookify’d](https://reactjs.org/docs/hooks-overview.html) some existing components so they could use (2) and (3).
5. Updated and added tests (an unexpected bonus!), and documented the new way to interact with these services.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>In the new architecture, a new wrapper component for Chat called ChatTooBig (an annoying name to motivate us to break Chat down into smaller pieces) handles websocket connection independently of Router, meaning Router can be just a router. Additionally, components like A and B can “hook” into the api or data they need, directly to contexts.

The refactor should make development on the application easier. Router and Chat are now decoupled, meaning that they can be tested separately and new components can be added to Router more easily. Components can now selectively subscribe to data and unnecessary prop drilling is on the way out. Finally, it lays the groundwork for a permanent move from `Weedux` to Redux for all of Curai Health’s global state management needs.

*Exploring Semantic Similarity in Patients’ Reasons for Encounters*

In a pivot from my first couple projects, I started this one close to the end of my internship. The premise of this experimental project is to be able to cluster together patient-doctor interactions by similarity, which could add significant value (for example, doctor’s could make sure that they’re not ruling out a potential disease that was spotted in a similar case).

Overall, it was an eventful, rewarding, and informative summer! I wasn’t sure what to expect about working at a tech startup, especially given the various stereotypes floating around, but I was extremely impressed by Curai’s culture and values. My teammates were dedicated not only to the mission and product, but also to emphasizing ethical growth, team cohesion, and life outside of work. I found it to be an inclusive and diverse environment where everyone’s opinion was valued (even the interns!) and I’m glad there’s at least one startup breaking the typical startup mold. Thanks for reading!

### 6. Improving an online dermatology workflow

*Mesert Kebede (MIT)*

I’m Mesi, currently a Master’s student in Computer Science at MIT. This summer I interned with the product engineering team at Curai.

Curai operates the [Curai Health](https://www.curaihealth.com) service, where users can chat with a medical professional and upload images. These images are especially useful for diagnosing dermatology cases. My primary project was to improve our image collection and processing workflow for our Curai Health app. This work was in line with our effort to improve the Curai Health experience for dermatology specific issues as 30% of all primary care cases are related to dermatology \[site\]. I worked alongside a cross-functional team to determine the main aspects to this project 1) streamline data collection and storage of images 2) improve the user experience to facilitate upload of high quality photos.

To address the first goal I created a datatable to store image(s), associated metadata such as the where the condition is located, if there is any pain, itching, bleeding…etc, and the medical findings and diagnosis associated with the interaction. This datatable consolidated all of the data we collected for images enabling future machine learning efforts.

In order to improve the quality of photos taken by users, we decided to present users with guidelines for taking high quality photos. These guidelines serve as a reminder to important elements of photography that are often ignored in chat mediums. For instance, it is better to take the photo with a solid colored background so that the focus is on the condition being photographed. Outlined below is the image collection flow before this project \[Figure 1\], and the final flow after the completion of my internship \[Figure 2\].

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>Figure 1: Initial image collection flow.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>Figure 2: Final image collection flow, from the conversation user initiates flow by pressing the camera button. The user is then asked if the photo is dermatology related, and presented with guidelines.

<figure>![](http://localhost:8080/wordpress/wp-content/uploads/2022/06/img_629a717f3bd51.jpg)<figcaption></figcaption></figure>Once the user uploads a photo they will be asked to fill out a survey, if this image was derm related.