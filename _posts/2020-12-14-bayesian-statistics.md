---
layout: post
title: ISYE6420 Bayesian Statistics
---

[**Course Link**](https://omscs.gatech.edu/isye-6420-bayesian-statistics)

This is the 5th course I’ve taken for my study in Georgia Tech. Bayesian statistics has been an interesting topic that I would always want to learn. This semester I felt ready to take on this journey.

I finished the course with 95%, another A secured. The direct impact of this course is I’m no longer afraid of the math representations in the papers since there are tons of equations to be written in the assignments. The videos of the courses only quickly skim through all the topics, the valuable parts are the demos of using WinBUGS and OpenBUGS. Most of the time I’ve been reading the textbook - [http://statbook.gatech.edu/index.html](http://statbook.gatech.edu/index.html), which elaborates the topics in details. In the end, I turned out to write most of the assignments in Matlab instead of WinBUGS. And it is surprisingly efficient to translate math equations from paper to code in Matlab.

I did meet one problem using Matlab instead of using WinBUGS. In the final exam, we are asking to find influential observations or outliers in the sense of CPO or cumulative. But Matlab does not have CPO function. To use WinBUGS with Mac, I figured out how to use AWS workspace. It is a virtual desktop image and works exactly like a normal windows system. The only downside is it is a bit lagging. After installing WinBUGS in this virtual windows desktop, calculating CPO in WinBUGS is much easier, and potential outliers are defined as (CPO)i < 0.02.

Overall the course has met my expectation, though most of the stuff I learned from reading the textbook. I hate to say this but the lecture videos are so dry. The textbook is much better with detailed examples. The assignments are great for one to apply knowledge into practice. I couldn’t think of a lot of real-world applications other than a bayesian optimizer for hyperparameter fine-tuning. But it helps since most of my colleagues are from statistics background. It is useful to understand topics like MCMC methods or Hidden Markov Models to communicate with them. 