# Writing roadmap for nanx.cc

Reference for the next writing phase. These are proposed articles, not published posts or finished drafts. This directory is excluded from the website build.

## Editorial direction

**Shawn Xiao — AI / ML Engineer & Data Scientist**

Building production AI systems, agents and intelligent products. The central question is how Shawn thinks about building intelligent systems: production decisions, evaluation, engineering trade-offs and product impact, grounded in firsthand experience.

Prefer concrete lessons from building systems over generic introductions to Transformers, RAG or agents. Anonymize proprietary details and verify any experience, metrics and examples before writing. Keep finance, travel and unrelated interests secondary; historical posts stay in Archives.

## Content pillars

| Pillar | Target share |
| --- | ---: |
| Production AI and agents | 60% |
| ML systems and experimentation | 25% |
| Career and engineering perspective | 15% |

## Proposed articles

### 1. From ML Models to AI Agents: How the Production Stack Is Changing

Potential cornerstone article establishing the new identity. Compare classical ML pipelines, LLM applications, workflow-based LLM systems and agents. Cover where agents help; state, memory and tools; evaluation and observability; failure recovery; human review; and deterministic versus probabilistic components. Include a Mermaid architecture diagram.

### 2. Building LLM Classification Systems: Prompting vs Fine-Tuning vs Small Models

Compare prompt APIs, few-shot prompting, frontier and smaller models, open-source alternatives and QLoRA fine-tuning. Ground the discussion in latency, throughput, cost and quality trade-offs. Use current model names at writing time and anonymize proprietary deployment details.

### 3. LLM Evaluation Is a System, Not a Metric

Connect offline golden sets, human evaluation, LLM-as-judge, pairwise comparisons, rubric design, slice analysis, regression tests, production feedback, online experiments and business metrics. Diagram the loop: offline evaluation → release gate → canary → A/B experiment → monitoring → new evaluation data.

### 4. How I Use Coding Agents for Real Engineering Work

A personal account of planning, repository exploration, implementation, testing, review, reusable instructions and skills, context management, and parallel delegation. Explain where agent-written code creates risk and how the workflow handles it. Avoid a generic prompt list.

### 5. Designing Reliable Agentic Systems: What Happens When the Agent Is Wrong?

Architecture-focused treatment of retries, validation, permission boundaries, tool contracts, idempotency, human confirmation, tracing, deterministic orchestration, fallback models and evaluation. Use a concrete failure-and-recovery example.

### 6. ML System Design Beyond the Model

Potential reference architecture: sources → ingestion → data quality → feature pipelines → training → registry → serving → monitoring → retraining. Discuss schema evolution, lineage, offline/online consistency, drift and governance. Useful for system-design preparation as well as production work.

### 7. Why Offline ML Metrics Are Not Enough

Explain why model accuracy, user impact and business impact differ. Connect offline evaluation to product metrics, A/B testing, guardrails, long-term and novelty effects, and heterogeneous treatment effects.

### 8. A Practical Guide to Experimentation for ML Products

Potential cornerstone article focused on operational problems rather than introductory statistics: randomization unit, exposure logging, sample-ratio mismatch, CUPED, power and MDE, sequential peeking, guardrails, holdouts and ML feedback loops.

### 9. Causal Inference vs Prediction vs Decisioning

Organize around three questions: “What happens next?” (prediction), “What caused this?” (causal inference), and “What should I do?” (decisioning). Extend to contextual bandits, reinforcement learning, optimization and agents.

### 10. What 12 Years in Data Science Changed My Mind About

Provisional title: confirm the tenure before publishing. Reflect on the relative importance of models, data quality, evaluation, engineering, product judgment and end-to-end building. Discuss how AI changes traditional role boundaries, using personal examples rather than résumé claims.

### 11. The Data Scientist Role Is Splitting

Explore product analytics, experimentation and causal inference, applied science, ML engineering, AI engineering and decision science. Discuss where agentic AI fits and what the changing boundaries mean for practitioners.

### 12. From Notebook Data Scientist to End-to-End AI Builder

A partly autobiographical technical essay about the changing bottleneck in building products. Cover APIs, backend, frontend, deployment, infrastructure, agents and coding assistants. Explain the transition through work and decisions, rather than a résumé narrative.

## First six articles

| Order | Article | Purpose |
| --- | --- | --- |
| 1 | From ML Models to AI Agents: How the Production Stack Is Changing | Establish the new identity |
| 2 | LLM Evaluation Is a System, Not a Metric | Demonstrate professional differentiation |
| 3 | How I Use Coding Agents for Real Engineering Work | Offer a current, personal workflow |
| 4 | Building LLM Classification Systems: Prompting vs Fine-Tuning vs Small Models | Explain real deployment trade-offs |
| 5 | What 12 Years in Data Science Changed My Mind About | Add a personal engineering perspective |
| 6 | Designing Reliable Agentic Systems: What Happens When the Agent Is Wrong? | Demonstrate architectural depth |

Suggested cadence: one substantive article every two weeks, approximately three months for this first sequence. Pin up to three strong cornerstone articles once they are actually written and published; do not create placeholder posts.

## Future taxonomy

Use a small category vocabulary: **AI Systems**, **Machine Learning**, **AI Engineering**, **Data Science**, **Career**. Choose one primary category per article; Chirpy treats multiple category values as a hierarchy.

Candidate tags: `agents`, `llm-evaluation`, `rag`, `experimentation`, `causal-inference`, `mlops`, `system-design`. Add Categories and Tags navigation when the new content warrants it.

## Deferred Projects page

Curate 4–6 verified, publishable projects rather than listing the entire GitHub history. Potential groups: Production AI (customer experience, classification/evaluation, agentic tooling), ML (NLP/CV, ranking/recommendation), and Experiments (open-source agents, the historical trading experiment).

Before adding a project, collect its actual name, public link, personal contribution, problem, technical approach and supported outcome. The examples above are suggestions, not claims about completed work.
