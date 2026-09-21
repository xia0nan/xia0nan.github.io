# Archive rewrite plan for nanx.cc

This plan guides subsequent edits to the six historical articles. Preserve the archive’s personal voice, correct misleading technical claims, and improve readability for practitioners and prospective OMSCS students reading historical experience. This planning change contains no article rewrites.

The [content roadmap](content-roadmap.md) remains the guide for new writing. For the sentiment archive, use a conceptual comparison with explicitly labeled pseudocode; defer a reproducible benchmark to a separate future project.

## Editorial judgments

- Prioritize technical credibility over stylistic uniformity. Sentiment Analysis and Trading Book need substantive corrections; the other four articles need restrained editing.
- The reported ROC formatting bug is absent from the current sentiment source: `{roc_auc:.2f}` is valid. Do not list it as a required fix.
- Replace the sentiment article’s runnable-guide promise instead of partially repairing scripts that still lack a complete evaluation pipeline.
- Treat Trading Book’s in-sample evaluation as its main limitation. The [historical notebook](https://github.com/xia0nan/trading-book/blob/e67a53ac9d26b576f62ac724e3ad04d8a839cf90/notebooks/05_ML_strategy.ipynb) evaluates on its training period; this matters more than terminology alone.
- Keep grades, tables, and candid opinions when they provide historical context. Move distracting details behind the main takeaway rather than deleting them categorically.
- Separate present-day factual corrections from original experience. Do not invent remembered lessons, motivations, experiments, or changed opinions.

## Article rewrite briefs

### 1. Sentiment Analysis from Word2Vec to BERT — substantial correction

Source: [sentiment article](../_posts/2020-04-25-sentiment-analysis-from-word2vec-to-bert.md).

**New title:** “Sentiment Analysis from Word2Vec to BERT: Comparing Approaches”

**Purpose:** explain differences between model families without claiming a reproduced benchmark.

- Open with a dated correction note explaining that the earlier code was incomplete and its performance figures were unsupported.
- Organize around IMDb → model-specific preprocessing → four approaches → evaluation requirements → practical considerations.
- Introduce TF-IDF + logistic regression, averaged Word2Vec + logistic regression, BiLSTM, and BERT. Compare representation, classifier, assumptions, and limitations without ranking unmeasured performance.
- Replace the current Python scripts with concise explanations and explicitly labeled pseudocode. Remove the broken HTML regex and dimensionally inconsistent weighting implementation.
- Explain that Word2Vec supplies features and still needs a classifier. Preserve negation during preprocessing; feed review text through BERT’s associated tokenizer rather than the current stopword-removal and lemmatization pipeline. Link to maintained [Hugging Face text-classification guidance](https://huggingface.co/docs/transformers/main/tasks/sequence_classification).
- Describe a validation split drawn from training data, training-only fitting of learned preprocessing, and a held-out test set. Explain that accuracy, F1, timing, and hardware would need recorded runs before comparison.
- Remove the 86%/89%/93% figures, unsupported review-length observations, unmeasured speed rankings, and “current state-of-the-art” claims.
- Describe the labeled train/test splits in the [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and identify the additional unlabeled data as unused here.
- Remove the hand-maintained table of contents in favor of the theme’s TOC, preserving existing fragment destinations.

**Done when:** readers can distinguish conceptual guidance from executed evidence, and no incomplete script or unsupported number is presented as a working experiment.

### 2. Trading Book — substantive methodological correction

Source: [Trading Book](../_posts/2020-02-26-trading-book.md).

**Keep the title and two-day project framing.**

- Organize around the experiment’s purpose → implemented pipeline → what the demonstration shows → limitations → historical setup and next steps.
- Describe the actual supervised learner: bagged randomized trees predicting short/cash/long positions from indicators. Explain how position changes become buy/sell orders. Identify Q-learning as an alternative rather than the active implementation. Use the [historical learner](https://github.com/xia0nan/trading-book/blob/e67a53ac9d26b576f62ac724e3ad04d8a839cf90/StrategyLearner.py) as the implementation reference.
- State that the [linked notebook](https://github.com/xia0nan/trading-book/blob/e67a53ac9d26b576f62ac724e3ad04d8a839cf90/notebooks/05_ML_strategy.ipynb) trains and evaluates on JPM data from 2008–2009. It defines a later period but does not use it in the displayed evaluation.
- Record the notebook’s settings accurately: zero commission and an impact parameter of `0.005`. Explain their limited realism without describing all trading costs as absent.
- Retain the original chart with a historical, unreproduced-result caption. Remove language implying demonstrated live profitability.
- Replace the broad rejection of EMH with a testable predictive-signal hypothesis. If discussing historical price information, connect it to weak-form efficiency. Refer to the [CFA market-efficiency reference](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/market-efficiency).
- Correct “algorithmic trading” and “orders file”; distinguish positions from orders and supervised learning from reinforcement learning.
- Add a short limitations paragraph covering execution timing, leakage, costs, survivorship bias, repeated experimentation, and chronological out-of-sample testing. Distinguish observed shortcomings from issues not audited.
- Label AWS/Colab recommendations and the old future-work list as historical. Do not upgrade the separate trading repository or rerun experiments under this plan.

**Done when:** the article accurately describes the implementation and limits its conclusions to an exploratory demonstration.

### 3. ISYE6420 Bayesian Statistics — focused revision

Source: [Bayesian Statistics](../_posts/2020-12-14-bayesian-statistics.md).

- Lead with the documented benefit: greater confidence reading mathematical notation and translating equations into code.
- Keep the textbook preference, dry-lecture judgment, MATLAB experience, WinBUGS workaround, and grade as personal context.
- Qualify the CPO threshold as specific to the exercise. Rephrase “MATLAB does not have CPO” as a limitation of the author’s workflow.
- Clarify that Bayesian optimization was the application the author connected to at the time, not the boundary of Bayesian inference.
- Add a short dated tooling note: the [Fall 2026 syllabus](https://omscs.gatech.edu/sites/default/files/documents/Syllabi/ISYE%206420%202026-3.pdf) specifies Python and requires PyMC for later assignments and the final. Do not repeat the suggested R/Stan claim.

### 4. CS6300 Software Development Process — light revision

Source: [Software Development Process](../_posts/2020-01-01-cs6300-software-development-process.md).

- Structure around personal difficulty, development tools, asynchronous teamwork, and preparation advice.
- Change universal difficulty claims to “I found…” and retain uncertainty around the originally estimated grade.
- Preserve the experience of working with three remote teammates without implying all teams should work asynchronously.
- Remove the TensorFlow.js aside and bookmark dump. Retain the official course link and two concise Git/Java preparation references.
- Clearly label current prerequisites as current guidance; the [official course page](https://omscs.gatech.edu/cs-6300-software-development-process) recommends Java and software-engineering familiarity.

### 5. CS6476 Computer Vision — light revision

Source: [Computer Vision](../_posts/2019-12-06-cs6476-computer-vision.md).

- Foreground the connection to the OCR project at work, followed by workload and project-selection lessons.
- Preserve “CNN project is a monster” and the grade, while shortening the grade discussion.
- Do not add specific OCR techniques or learning anecdotes without supporting recollection or artifacts.
- Remove unexplained EAR/MHI references and the certainty that unchosen projects would have been easier.
- Replace “study all materials beforehand” with a short preparation note grounded in Python, linear algebra, probability, and image-processing fundamentals. Refer to the [official course guidance](https://omscs.gatech.edu/cs-6476-computer-vision).

### 6. Almost Year End — copyedit only

Source: [Almost Year End](../_posts/2019-11-18-almost-year-end.md).

- Fix grammar and spelling, including “Anomaly Detection,” and rename the work table’s “Course” column to “Project.”
- Preserve the first-post framing, research dead ends, music and travel details, and exam quotation.
- Retain the compact tables and historical goals. Do not retrospectively turn plans into accomplishments.
- Add no new thesis or obligatory present-day reflection.

## Short writing guide

- Begin with the actual problem, experience, or question.
- State the useful takeaway early when the material supports one.
- Separate personal experience, source-backed facts, measured results, and proposed work.
- Use concrete details and explain only the theory needed.
- Keep candid judgments; qualify their scope.
- End with a specific lesson or unresolved question when appropriate. Personal journal posts need no prescribed conclusion.
- Borrow structural discipline from other writers without imitating their voice.

## Repository compatibility and validation

### Planning commit

Add this plan and a link from the existing content roadmap. Review the staged diff and run `git diff --check`. Stage only those two documentation changes; leave the existing untracked `.vscode/` files untouched.

Commit locally with `docs: add archive rewrite plan and editorial guidelines` and report the commit hash. Pushing or publishing is outside this task. No article rewrites, baseline changes, or README edits belong in this commit.

### Subsequent rewrites

- Work in three batches: sentiment, trading, then the four smaller revisions.
- Preserve publication dates, fixed permalinks, archive flags, feed entry IDs, comment mappings, and the archive banner.
- Use the existing `last_modified_at` metadata convention for substantive revisions, with the actual revision date. Add visible correction notes to the two technical articles; do not call substantial edits “light updates.”
- Preserve heading IDs or provide aliases for removed headings.
- Update only the intentionally changed body hashes and title expectations in [tools/archive-baseline.json](../tools/archive-baseline.json), after reviewing each article diff. Retain the archive checks and document the correction policy in the [README](../README.md).
- No new public API or content schema is needed.

After article changes, run the production build, archive verification, homepage tests, and internal-link checks documented in the README:

```sh
JEKYLL_ENV=production bundle exec jekyll build
bundle exec ruby tools/verify_site.rb
bundle exec ruby tools/test_home.rb
bundle exec htmlproofer _site --disable-external --no-enforce-https
```

Inspect the six rendered posts for TOC behavior, old fragments, tables, images, correction dates, and mobile readability. Verify archive/search/feed visibility and continued exclusion from the homepage and Recently Updated.

Completion requires supported claims, preserved personal voice, working historical URLs, and passing site checks. New benchmark experiments and expanded personal recollections are not prerequisites.
