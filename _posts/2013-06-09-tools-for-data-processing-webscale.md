---
id: 71
title: 'Tools for Data Processing @ Webscale'
date: '2013-06-09T18:19:00+00:00'
author: 'Xavier Amatriain'
##layout: post
permalink: /tools-for-data-processing-webscale/
header:
  teaser: https://amatriain.net/blog/images/ParquetTwitter-Poster.jpg

categories:
    - Uncategorized
---

A couple of days ago, I attended the [Analytics @Webscale workshop](http://analyticswebscale.splashthat.com/) at Facebook. I found this workshop to be very interesting from a technical perspective. This conference was mostly organized by Facebook Engineering, but they invited LinkedIn, and Twitter to present, and the result was pretty balanced. I think the presentations, though biased to what the 3 “Social Giants” do, were a good summary of many of the problems webscale companies face when dealing with Big Data. It is interesting to see how similar problems can be solved in different ways. I recently described how we address at Netflix many of these issues in our [Netflix Techblog](http://techblog.netflix.com/2013/03/system-architectures-for.html). It is also interesting to see how much sharing and interaction there is nowadays in the infrastructure space, with companies releasing most of what they do as open source, and using – and even building upon – what their main business competitors have created. These are my barely edited notes:

Twitter presented several components in their infrastructure. They use Thrift on HDFS to store their logs. They have now build [Twitter Parquet](http://parquet.io/), a columnar storage database that improves storage efficiency by allowing to read columns at a time. 

| ![](/blog/images/ParquetTwitter-Poster.jpg) |
|---|
| @squarecog talking about Parquet |

They also presented their DAL (Data Access Layer), built on top of [HCatalog](http://incubator.apache.org/hcatalog/).

| ![](/blog/images/TwitterDAL-HCatalog-Slide.jpg)|

| ![](/blog/images/DALTwitter-Poster.jpg)|

Of course, they also talked about [Twitter Storm](http://storm-project.net/), which is their approach to distributed/nearline computation. Every time I hear about Storm it sounds better. Storm now supports different parts of their production algorithms. For example, the ranking and scoring of tweets for real-time search is based on a Storm topology.

|![](/blog/images/TwitterRankingStorm-Slide.jpg) |

Finally, they also presented a new tool called [Summingbird](http://lanyrd.com/2013/lambda-jam/scghxb/). This is still not open sourced, but they are planning on doing so soon. Summingbird is a DSL on top of [Scalding](https://github.com/twitter/scalding) that allows to define workflows that integrate offline batch processing from Hadoop and near-line from Storm.

| ![](/blog/images/TwitterSummingbird-Slide.jpg) |

| ![](/blog/images/Summingbird-Poster.jpg) |

LinkedIn also talked about their approach to combining offline/near-line/real-time computation although I always get the sense that they are much more leaning towards the former. They talked about three main tools: [Kafka](http://kafka.apache.org/), their publish subscribe system; [Azkaban](http://data.linkedin.com/opensource/azkaban), a batch job scheduler we have talked about using in the past; and [Espresso](http://data.linkedin.com/projects/espresso) a timeline-consistent NOSQL database. 

| ![](/blog/images/Exa-scaleSystemsFacebook-Slide.jpg) |

Facebook also presented their whole stack. Some known tools, some not so much. [Facebook Scuba](https://www.facebook.com/notes/facebook-engineering/under-the-hood-data-diving-with-scuba/10150599692628920) is a distributed in-memory stats store that allows them to read distributed logs and query them fast. [Facebook Presto](http://gigaom.com/2013/06/06/facebook-unveils-presto-engine-for-querying-250-pb-data-warehouse/) was a new tool presented as the solution to get fast queries out of Exabyte-scale data stores. The sentence “A good day for me is when I can run 6 Hive queries” supposedly attributed to a FB data scientist stuck in my mind ;-). 

Morse is a different distributed approach to fast in-memory data loading. And, [Puma/ptail](http://www.quora.com/Why-did-Facebook-develop-Puma-pTail-instead-of-using-existing-ones-like-Flume) is a different approach to “tailing” logs, in this case into HBase.

| ![](/blog/images/FacebookScubaPoster.jpg) |
    
Another Facebook tool that was mentioned by all three companies is [Giraph](https://github.com/apache/giraph). (To be fair, Giraph was started at Yahoo, but Facebook hired the creator Avery Ching). Giraph is a graph-based distributed computation framework that works on top of Hadoop. Facebook claims they ran a Page Rank on a graph with a trillion edges on 200 machines in less than 6 minutes/iteration. Giraph is another alternative to [Graphlab](http://graphlab.org/). Both LinkedIn and Twitter are using it. In the case of Twitter, it is interesting to hear that they now prefer it to their own in-house (although single-node) [Cassovary](https://github.com/twitter/cassovary). It will be interesting to see all these graph processing tolls side by side in this year’s [Graphlab workshop](http://graphlab.org/graphlab-workshop-2013/).
    
Another interesting thread I heard from different speakers as well as coffee-break discussions was the use of [Mesos vs. Yarn](http://www.quora.com/How-does-YARN-compare-to-Mesos) or even [Spark](http://spark-project.org/). It is clear that many of us are looking forward to the NextGen Mapreduce tools to reach some level of maturity. 
