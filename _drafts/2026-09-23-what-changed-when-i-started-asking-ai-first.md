---
layout: post
title: What Changed When I Started Asking AI First
categories: [AI Systems]
tags: [agents]
comments: false
---

<!--
AUTHOR REVIEW BEFORE PUBLICATION
- Confirm exactly what the agent did in the Japan investigation before attributing the discovery to it; keep the company and metric anonymized.
- Replace the marked gap with an ordinary-looking AI failure. Do not invent outcomes or metrics.
- Confirm the teacher's relationship to the author and permission to describe the routine anonymously.
- Keep the Waymo driving and fare judgments clearly personal; do not generalize either from one ride.
- Check whether the coding and PR-review wording accurately describes your own workflow and what can be shared publicly.
-->

A few years ago, a request for analysis usually meant opening a SQL editor. I would find the right tables, write a query, inspect the rows, change the query, make a chart, and go back again when the chart raised another question. Much of the work was in the back and forth.

Now I am more likely to start by asking what an agent would need to do the analysis. Which source should it use? What does the metric actually mean? Is the data good enough? What would convince me that its answer is right? The agent can write SQL, inspect results, and make the visualization. My first job is increasingly to give it the context to do those things well.

One real investigation involved a persistent gap between Japan and other markets. For a question like that, an agent can segment the difference by app version, channel, and date, then check whether a release or campaign lines up with it. Those clues may live across several systems and departments. With well-documented data and access to those operational records, the agent can pursue many branches together and build a broader picture than a single dashboard gives me.

The cause in this case was a security or compliance check applied in Japan even though it should not have been. That was a localization problem as much as a data problem. Someone unfamiliar with the market could miss it, and so could an agent if that local knowledge was absent from its context. Polishing the SQL alone would not reveal the answer.

It can also get almost to the answer and leave the hardest part behind. Sometimes the analysis feels 95% right, yet the last 5% demands all of my attention. Those numbers are figurative; the frustration is real. One unexplained discrepancy may still take a full round of human investigation to resolve, and it is exactly the part that prevents me from trusting the conclusion. Simple questions are increasingly self-service; root-cause analysis still depends on the quality of the context and the final check.

I work in machine learning, NLP, LLMs, and data science, so I see this change close up. I also see it when I leave work. ChatGPT has become a first stop for questions I might once have typed into Google: a bit of troubleshooting, something I am curious about, a purchase I am weighing, or the start of a trip. I can ask a follow-up without starting the search again.

The change reaches into both professional workflows and small decisions in an ordinary day. It feels fast even from the middle of the industry. The tasks these systems handle well still surprise me, and so do the tasks that trip them up.

## From doing each step to setting up the work

In 2023 and 2024, the most familiar AI interaction was a conversation. You asked a question and received an answer, a draft, or some code. [ChatGPT](https://openai.com/index/chatgpt/) had arrived at the end of 2022; [Claude](https://www.anthropic.com/news/introducing-claude) followed in 2023. A good answer still left a lot of work between the chat window and the actual task.

An agent changes that distance. It can use tools, inspect files or pages, run code, and revise its work in response to what happens. The models have improved. The surrounding system matters too: access to the right data, connectors to the systems where work lives, reusable instructions or *skills*, and a way to check the result. Anthropic's [2026 Economic Index](https://www.anthropic.com/research/economic-index-june-2026-report) describes Claude sessions moving toward longer-running agent tasks. That matches the direction I see, although any one product's usage data tells only part of the story.

In analytics, this puts an old problem in a new place. An agent can write a technically correct query against the wrong definition of a customer. It can draw a clean chart from incomplete data. SQL fluency does not rescue the answer if the agent cannot reach the right source or does not understand what the business means by a metric. A useful semantic layer, data quality, and access rules become part of the agent's working environment. If those are weak, the agent may simply produce the wrong analysis faster. If they are strong, an agent can connect signals that used to require cross-department coordination.

I see a similar change in coding. In my workflow, agents do much of the implementation, and we no longer rely on routine manual, line-by-line pull request review. More of my effort goes into specifying the change, giving the agent the right repository context, deciding what should be tested, and checking whether the finished behavior solves the problem. A skill that captures a good workflow can matter more than performing the same steps by hand each time.

There is something odd about spending time writing a skill that teaches an agent how to do work I used to do myself. I like being able to get more done. I also notice what happens to the value of knowing the old steps.

<!-- AUTHOR DETAIL: Add one shareable coding example if available: the task, the agent's implementation and review steps, and the check that caught or confirmed the result. -->

I could tell myself that judgment and taste are the safe part of the job. They matter a great deal right now. I am less confident that they form a permanent boundary. Agents can already compare alternatives, criticize a design, and learn from feedback. I expect them to get better at work we currently describe as "the human part." Accountability may stay with a person even as more of the judgment behind a decision is assisted or automated.

That is why the future of analytics looks different to me from a future with faster SQL tools. When someone with a question can get a credible analysis without knowing SQL or building a dashboard, the technical barrier to routine analysis falls. I expect much everyday analysis to stop requiring a specialist, and demand for people hired mainly to perform those steps to shrink. Data access, messy definitions, and slow-moving organizations will delay that change. They may also create new work around governing the systems that answer questions. I cannot infer a job count from my own workflow. I can say that the workflow I was trained to do is already changing underneath me.

## The habit spreads beyond technical work

I think of "AI native" as a habit rather than a job title. Before opening an app or learning its controls, you ask whether an agent can handle the task, what it needs to know, and how you will check its work. Someone without a technical background can become very good at this. In some cases, they can imagine a use that a person building AI systems would never think to try.

One school teacher I know of first tried OpenClaw and now uses [Grok Bot](https://x.ai/news/introducing-grok-bot) to help with routine work around lesson planning, preparing teaching materials, and grading. I do not know how much time it saves, but the possibility is striking: a teacher can shape a tool around the way they teach. A niche workflow with little software budget might never justify a custom SaaS business. An agent can make a personal tool possible anyway.

<!-- AUTHOR DETAIL: Confirm how the teacher's agent assists with these tasks and whether this anonymized account may be published. -->

For everyday questions, the shift is easy to feel. I used to search, open several pages, and assemble an answer myself. Now I often begin by asking ChatGPT. If the answer is good enough for a low-stakes decision, I can move on. If I am spending serious money or relying on a fact that might have changed, I check the underlying sources. The first step has changed even though the need to verify has not disappeared.

Writing offers a stranger version of the same change. I see more text produced with AI, and I see people use AI to read or summarize someone else's text. A long AI-written document travels from one agent to another, with humans perhaps reading a condensed version in between. The result can be useful, but it should make a writer ask what the reader actually needed. I do not want this article to be another long piece that says little and then needs a model to shorten it.

Creative work has its own version of this. Image and video generation make it cheap to try a visual idea, revise it, and try another. That changes the pace of advertising and marketing even when the result is uneven. I notice generated video in more places before I am consistently impressed by its quality. A medium can become common before it becomes excellent. The interesting question for a creator is what becomes possible when the first version takes minutes, and what remains difficult after producing fifty versions becomes easy.

## Why the easy tasks can be hard

The frontier does not move in the order I would have guessed. In 2025, an advanced Gemini Deep Think system produced solutions that [official IMO graders scored at the gold-medal level](https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/). That is a remarkable result on exceptionally difficult competition problems. It does not mean research mathematics is solved, and it does make "AI will soon be able to do hard math" an outdated prediction.

Trip planning is the example I keep returning to. An AI can summarize thousands of posts about a city and produce a neat itinerary. How do I tell whether the itinerary is good? A booking is a weak measure: it does not say whether the restaurants were worth visiting, whether I enjoyed the pace, or whether I spent half the day crossing town. My budget, available time, energy, and preferences may be vague even to me until I start choosing among options.

When I plan a trip myself, I also learn what I want during the process. I may favor the result partly because I put effort into making it. A one-click itinerary can save that effort while losing some of the discovery and attachment that came with it. Perhaps a useful travel agent will need to help me make choices, then remember *why* I made them. A fluent list of attractions is a much easier output to produce.

<!-- AUTHOR DETAIL: If available, add an actual AI trip-planning attempt or another ordinary-looking failure. State what was wrong and how you noticed. -->

This gives me a better way to judge a possible AI task than asking whether it is "easy" or "hard." Can I state what a good result looks like? Can the agent reach the relevant information and tools? Can I check the answer without doing the entire task again? What happens if it is wrong? Advanced math problems can have a clear target and expert grading. A good holiday has no answer key.

I often feel that the possible uses are limited mainly by imagination and the cost of running models. In practice, the list is longer: data quality, permission to act, reliability, and the time it takes to check the output. Those constraints do not make the possibilities less exciting. They help explain which ideas become part of daily life.

## From experiments to things people use

The product cycle matters as much as the demo. [OpenClaw](https://docs.openclaw.ai/) gives people an agent they can run on their own infrastructure and reach through familiar messaging apps. Meta introduced [Muse](https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent/) in September 2026 as a personal agent that works across connected services from a dedicated virtual machine. They are different products, but they point toward the same kind of everyday request: tell an agent what you need done, then let it use software on your behalf. Muse looks to me like one sign that this idea is moving into a more polished consumer product.

I felt a similar shift on a visit to San Francisco. A colleague told me I had to try Waymo. I got into a car with no human driver, and it took me onto a freeway. It was not slow. The balance of speed and caution felt right to me, and I had the impression it noticed possible hazards I might have missed. That is an impression from one ride, not a safety finding or a claim about which sensor saw what. Waymo's [driverless service in San Francisco](https://waymo.com/blog/2024/06/waymo-one-is-now-open-to-everyone-in-san-francisco/) and [freeway rides in the Bay Area](https://waymo.com/blog/2025/12/2025-year-in-review/) are real products, and this one made physical AI feel much closer to me than a demo. The downside I noticed was the fare: an Uber driven by a person could be cheaper.

Physical AI makes me imagine a home agent that could handle household work that still requires a person. I would love to have one. Recent [robotics demonstrations](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) show progress in movement, manipulation, and multi-step tasks. A house is also full of unfamiliar objects, people, and situations where a small mistake matters. I expect useful physical applications, but I am less willing to put a date on a general household helper than on the next wave of software agents.

## What I expect to see next

By 2027 or 2028, I expect more analytical and coding work to begin with an agent. More teams will maintain definitions, tests, permissions, and reusable instructions that let agents work without repeatedly being taught the organization from scratch. Routine execution will be cheaper. My stronger prediction is that this will reduce the need for some specialist roles, including parts of data analytics. If those roles remain in high demand despite agents taking over the mechanical work, I will have underestimated the amount of interpretation, ownership, and organizational change the job requires.

I also expect personal agents to take on more connected, multi-step tasks: gathering information from several places, preparing an action, and asking a person to approve it. Their usefulness will depend on how well they handle access and mistakes. A dazzling demo will not make me hand over my calendar, email, or payment details if I cannot see what the agent is doing or stop it easily.

Generated media will become an even more ordinary part of what I see. Physical agents will probably arrive in narrower jobs before one can reliably do all the work in my home. Those are predictions, not a straight line from this year's demos.

What has already changed is my starting point. I still know how to open a SQL editor, search the web, and inspect a pull request. I increasingly ask what context an agent would need to take the first pass—and how I would know when it has done the job well. I am curious how long that will remain the part only I can do.
