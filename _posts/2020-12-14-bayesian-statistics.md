---
layout: post
title: ISYE6420 Bayesian Statistics
archived: true
hidden: true
permalink: /2020/12/14/bayesian-statistics.html
last_modified_at: 2026-09-22
---

[**Course Link**](https://omscs.gatech.edu/isye-6420-bayesian-statistics)

The most useful thing I got from this course was feeling less afraid of the equations in research papers. There were enough of them in the assignments that translating notation into code became much less intimidating.

This was my fifth course at Georgia Tech. Bayesian statistics was something I'd wanted to learn for a while, and this semester I felt ready for it. I finished with 95%, another A secured.

I spent more time with the [textbook](http://statbook.gatech.edu/index.html) than with the lecture videos. I hate to say it, but the lectures were dry. They moved quickly through the topics, while the book gave me the detailed examples I needed. The WinBUGS and OpenBUGS demonstrations were useful, though, and the assignments made me put the material into practice.

Despite those demos, I wrote most of my assignments in MATLAB. It was surprisingly convenient for translating equations from paper into code.

WinBUGS was easier for one part of the final: finding potentially influential observations using conditional predictive ordinates, or CPO. I didn't have a convenient CPO implementation in my MATLAB workflow. On my Mac, I got around the WinBUGS setup problem by running it on a Windows desktop through AWS WorkSpaces. It worked, although the remote desktop felt a bit laggy. For that exercise, I used a CPO threshold below 0.02 to flag potential outliers; that was specific to the exercise, not a general cutoff.

At the time, Bayesian optimization for hyperparameter tuning was the application I could most readily connect to my work. That says more about my experience then than about the scope of Bayesian inference. Many of my colleagues had statistics backgrounds, and being more comfortable discussing MCMC and Hidden Markov Models was useful in itself.

Overall, the course met my expectations. For me, the textbook and the assignments were the best parts.

> **Tooling note — 22 September 2026:** The setup above describes my 2020 semester. The [Fall 2026 syllabus](https://omscs.gatech.edu/sites/default/files/documents/Syllabi/ISYE%206420%202026-3.pdf) specifies Python and Jupyter notebooks, with PyMC required for homeworks 5 and 6 and the final. The MATLAB and WinBUGS account here is historical.
