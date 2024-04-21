---
id: 90
title: "Migrating Medium and Blogger blogs to self-maintained Github Pages"
date: '2022-06-07T00:00:00+00:00'
author: xamat
permalink: /migrating-my-blog
header:
    teaser: /images/90-05.png
reading_time:
    - ''
    - ''
categories:
    - Uncategorized
image: /blog/images/90-05.png

---


In case you are reading this post in my now Medium mirror, you can access the result of this migration [here](https://xamat.github.io/blog/).

For quite some time I have been pretty disappointed with my current [Medium blog](https://xamat.medium.com). It has been sad to see how much Medium has failed as a product. On the one hand, the tools and services for writers are horrible (as an example, they recently removed the ability to edit on mobile!). On the other hand, Medium distribution is completely dead. I get most of my views through Twitter and LinkedIn, and any of my Quora answers gets orders of magnitude more reads than my Medium post. Finally, they are not friendly to readers either (and they make it harder and harder to find the damn Friends Link). I have felt captive to Medium for the only reason that I did not have enough time to invest into alternatives. But, I thought that I just HAD TO. Here’s what I learned in the process of “freeing” my posts from Medium. In doing so, I also thought it would be a good idea to bring along my old [Blogger posts](https://technocalifornia.blogspot.com/ ).

![](/blog/images/90-01.png)

## Alternatives

I have maintained my [webpage](https://xavier.amatriain.net ) in Github Pages for some time now. Going all in and finding a way to publish a blog using basic html or some other tool was top of mind. However, I did consider a couple others. The first one is [Wordpress](https://www.wordpress.org), which I will cover at length later on. The other one is [Substack](https://substack.com/), which I have seen used quite a lot lately, even for people who do not use it for its primary purpose of sending newsletters. If I were more serious about monetizing my writing, or I even liked newsletters at all, I might have been tempted to give it a try. But, the reality is that in any comparison you read, you will see that Wordpress comes on top on almost anything, but probably easiness of use. That was something I was not prioritizing. But, if you are not a techie and are looking to earn cash from your posts, Substack might be worth looking into.

## Why not Wordpress

[Wordpress.org](https://www.wordpress.org) is a powerful publishing platform that has existed for many years (almost 20 to be precise). It is open source, and has a very strong plug-in ecosystem that you can use for almost anything you can imagine. So, why not go with Wordpress and call it a day? For most people it will come to a very simple reason: you need a host. Wordpress is a software, and it needs to be hosted somewhere. Unless you have a server laying around in your basement, you are going to need to pay someone to host your blog, and that costs some money. Now, to be clear, it is not a lot of money. Recommended hosts such as [Bluehost](https://www.bluehost.com/wordpress-hosting) have plans starting under $3 a month. And [Wordpress.com](https://www.wordpress.com) even has a barebones free hosting option.

In my case though, it was not about the money. I was looking for a truly open solution where I felt like I owned the pipeline end to end. Having to pay someone to host Wordpress did not feel like the way to go. Plus, as I mentioned before, I already had my webpage in Githubs Pages and was looking to continue using Github’s infrastructure for publishing.

## Exporting your Medium (and Blogger) posts

Regardless of what your target platform is, you are going to have to start with exporting your existing blog posts. How do you do that? There are basically two options that I explored (and eventually got both to work): (1) Find a script that downloads your posts directly, and (2) Use Wordpress, and its many plugins, to export and aggregate your content.

### Exporting scripts

Here is the thing, there are many scripts out there that promise to do the job of exporting your posts from your blog to some common format (mostly markup or .md). Most of those scripts probably worked at some point, but they don’t anymore. I spent a bunch of time trying out scripts that did not work until I found one that did. As a rule of thumb, any script that has not been updated in say the last year or so is unlikely to work. 

I came up with the one that did work, [Medup](https://github.com/miry/medup/) a bit late in the game, when I had already figured things out through Wordpress. I still would recommend you to start there since it worked well, and the developer was super responsive. Now, if you are reading this in say 2 years, things might be different and the script might not work anymore. That’s why the second approach is always a good fallback to have.

### Exporting and merging through Wordpress

This is the more complicated route that I ended up following. However, it was worth it for me to get to install Wordpress locally, understand how plugins worked, and use Wordpress as an intermediate step to integrate posts coming both from Medium and Blogger into a single platform, even if that one was temporary.

The first thing you need to get in order to follow this route is to install Wordpress. While Wordpress.org boasts about their “5 minute installation, this was probably the most time consuming part of the process. You can indeed install Wordpress in 5 minutes… if, and only if, you already have a working AMP (Apache, MySQL, and PhP) installation in your system. That is a breeze in any Linux system, but it took way more than expected on my recently upgraded Mac OS Monterey. I could not get it to work until I “cheated” my way through [XAMPP](https://www.apachefriends.org/index.html), a cross-platform AMPP installer. I came across XAMPP through some tutorials recommending it for Wordpress (see [this one] (https://www.hostinger.com/tutorials/how-to-use-xampp-wordpress/) e.g.), and it did make everything much easier than trying to get AMP up and running directly.

Once you have XAMPP installed, you still need to install Wordpress. The most tricky part of that [“5 minutes installation”](https://wordpress.org/support/article/how-to-install-wordpress/ ) is to get the mySQL database started, but you can find good articles such as [this one](https://wordpress.org/support/article/creating-database-for-wordpress/ ) that will guide you through the process. 

![](/blog/images/90-02.png)

![](/blog/images/90-03.png)


Once you are done with the installation, you should be ready to go. You will be able to use Wordpress locally. From Wordpress, you can install plugins to import content from Medium and Blogger (or wherever you have your old blog) into Medium, and from Wordpress to markup for then taking to your Jekyll blog. Basically you can use Wordpress as the ultimate tool to import and export content from anything to anything. I ended up using a few plugins including Blogger Importer, Blogger Importer Extended, Wordpress Importer, WP Smart Import, Wordpress Importer, Wordpress to Jekyll Exporter, and Auto Upload Images.

I should have kept better notes of what exactly worked, since I had to try a combination of different things, but for importing the Medium content, I basically had used the [following tool](https://mediumtowp.com/) as explained [here](https://themeisle.com/blog/migrate-medium-to-wordpress/) .I did have to add auto upload image plugin though, and even with that I struggled with images and had to use the script above. But, a combination of the above should more or less work, and you should soon have all your posts in Wordpress ready to edit and/or export to Jekyll markup.

![](/blog/images/90-04.png)


## Creating your Jekyll blog on Github Pages

The final step, although you could really start here particularly if you don’t have much old content to transfer, is to create a Jekyll blog on Github pages. [Jekyll] (https://jekyllrb.com/) is a wonderful little framework that enables you to create webpages and blogs fairly easily by using a simplified markup language and some configuration files. It has the added benefit of being fully and directly compatible with Github pages. So, all you need to do is start a Jekyll compatible repository on Github, and Github will render it! 

Taking all of this into account, I decided that the easiest way to start would be to fork an existing Jekyll blog. Lo and behold, while reading [this howto guide](https://www.smashingmagazine.com/2014/08/build-blog-jekyll-github-pages/ ),  I found [Jekyll-now](https://github.com/barryclark/jekyll-now), a popular Jekyll blog template that has been forked a whooping 34k times. I forked the project into [my github](https://github.com/xamat/blog ), pushed the blogs that I had exported into it, and was up and running in no time. As a side note, because I did not want to host the blog directly in my Github Pages root (i.e. xamat.github.io), I had to create the fork into a gh-pages branch (see documentation [here](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site )).

## The result

![](/blog/images/90-05.png)

## The ugly

After going through all those steps, I did have to waste some time in formatting some of the older blog posts. The one thing that I spent the most time on was fixing image URL’s since moving posts from one place to another and using several plugins and scripts did not preserve the paths. That was a somewhat tedious process, but it did give me the opportunity to revisit content I had not read in a long time.

## What is next?

I have just edited my Jekyll posts with the bare minimum. I have not even added labels and categories, nor explored styling options. Similarly, Jekyll has a lot of extensions and plugins too, and I haven’t even scratched the surface. I plan on adding more capabilities to my new blog. In particular, I need to add stats to keep track of the visits as well as comments.

In the meantime, I will continue to cross-post in Medium just as an experiment. I am interested in seeing what happens on Medium if I don’t share any of my posts externally. Over time though I plan to shut it down. The only question is whether Medium itself will shut down sooner than that, judging by its lack of progress and promise.
