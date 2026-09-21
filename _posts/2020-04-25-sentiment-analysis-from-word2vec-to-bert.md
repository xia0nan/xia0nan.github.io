---
layout: post
title: "Sentiment Analysis from Word2Vec to BERT: Comparing Approaches"
archived: true
hidden: true
permalink: /2020/04/25/sentiment-analysis-from-word2vec-to-bert.html
last_modified_at: 2026-09-22
---

> **Correction — 22 September 2026:** The earlier version presented incomplete code as a working guide and gave accuracy figures without recorded runs to support them. I've removed those figures and replaced the scripts with a conceptual comparison. The pseudocode below describes how an experiment could be built; it has not been run as a benchmark.

<span id="table-of-contents"></span>

## What are we comparing?
{: #overview }

A movie review can call the acting good and still dislike the movie. Turning that review into a single positive or negative label means deciding how much of the wording, order, and context the model gets to see.

Here are four ways to do it: TF-IDF with logistic regression, averaged Word2Vec with logistic regression, a bidirectional LSTM, and BERT. The useful comparison starts with what each model receives as input. A list of accuracy numbers would need a separate, reproducible experiment.

## The IMDb dataset
{: #dataset-and-exploratory-data-analysis }

The [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/) has 25,000 labeled training reviews and 25,000 labeled test reviews. Each split is balanced between positive and negative sentiment. There are also 50,000 unlabeled reviews, which are outside the setup described here.

For an experiment, draw a validation set from the labeled training split before fitting anything. Use the remaining training reviews to learn the model and any vocabulary or statistics needed to prepare its inputs. The validation set is for choosing settings; the test set stays aside until those choices are finished.

Review length is worth inspecting on the training data, especially before choosing how much text a sequence model will keep. I don't have a recorded length analysis to report here.

## Preprocessing depends on the model
{: #text-preprocessing }

Removing words can remove the answer. “Not good” should not become “good” because a stopword list happened to include “not.” Keep negation, and check what the tokenizer does with contractions before applying it to the whole dataset.

For TF-IDF and Word2Vec, decide how to handle case, punctuation, and HTML markup, then apply the same rules to every split. Learn vocabularies, document frequencies, and any locally trained word vectors from the training subset only. For a BiLSTM, fit its token vocabulary there too, with an unknown-token rule for words seen later.

For BERT, pass the review text to the tokenizer supplied with the chosen checkpoint. It handles the model's subword vocabulary and special tokens. The old stopword-removal and lemmatization pipeline does not belong in front of it. Padding and truncation still need explicit choices.

## Start with TF-IDF and logistic regression

TF-IDF represents a review through its terms and their weights. Logistic regression learns how those features relate to the sentiment label. Word bigrams can preserve short phrases such as “not good,” although this still loses most sentence structure. The [scikit-learn text-feature guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) explains the representation and its limits.

**Pseudocode — not executable Python:**

```text
fit TF-IDF vocabulary and document frequencies on training reviews
transform training and validation reviews with that fitted vectorizer
fit logistic regression on training vectors and sentiment labels
choose n-gram range and regularization using validation scores
```

This gives the other approaches a baseline to compare against. Moving to embeddings does not, by itself, establish an improvement.

## Averaged Word2Vec still needs a classifier
{: #word-embeddings-approach }

[Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) learns a vector for each word from its surrounding words. Those vectors are features, not positive or negative predictions. One way to represent a whole review is to average its known word vectors, then train logistic regression on the resulting review vectors.

**Pseudocode — not executable Python:**

```text
train Word2Vec on tokenized training reviews
for each review:
    collect vectors for tokens in the learned vocabulary
    average those vectors; use a zero vector if none are known
fit logistic regression on training review vectors and sentiment labels
transform validation reviews with the same Word2Vec model and averaging rule
choose settings using validation scores
```

The zero-vector fallback needs to be counted and inspected: a review with no known words gives the classifier no useful text features. Averaging also throws away word order. Each occurrence of a word contributes the same vector, whatever the sentence says around it.

The earlier code tried to multiply a vocabulary-sized TF-IDF array by a token-sized matrix of word vectors. Those dimensions do not generally match. Simple averaging makes the proposed representation clear without carrying that broken implementation forward.

## A bidirectional LSTM reads a sequence
{: #lstm-based-approach }

A BiLSTM receives an ordered sequence of token embeddings. Its forward and backward recurrent passes let the review representation depend on words before and after a token. A classification layer then maps that representation to sentiment. The [Keras bidirectional-layer documentation](https://keras.io/api/layers/recurrent_layers/bidirectional/) shows how the recurrent wrapper works.

**Pseudocode — not executable Python:**

```text
fit a token vocabulary on training reviews
encode reviews, keeping token order and marking unknown words
pad or truncate to a chosen length; mask padding
train embedding -> bidirectional LSTM -> sentiment classification layer
select sequence length and training checkpoint using validation scores
```

Keeping order gives this model information that averaging loses. It does not guarantee that the model will learn useful long-range relationships. Truncation can also cut out the part of a review that changes its meaning.

## Fine-tuning BERT
{: #bert-based-approach }

[BERT](https://aclanthology.org/N19-1423/) starts from a pretrained transformer whose token representations depend on the surrounding text. For sentiment classification, add a classification head and fine-tune on labeled reviews.

**Pseudocode — not executable Python:**

```text
load a pretrained BERT checkpoint and its matching tokenizer
tokenize review text with truncation and attention masks
pad batches as needed
fine-tune BERT and its classification head on training labels
select learning rate and checkpoint using the validation set
```

The [Hugging Face text-classification guide](https://huggingface.co/docs/transformers/main/tasks/sequence_classification) provides maintained implementation guidance using DistilBERT. Use the documentation for the library version you install. For the comparison proposed here, supply a validation split drawn from training data rather than using the held-out test set for checkpoint selection.

Pretraining gives BERT a different starting point from the locally trained models above. Record the exact checkpoint and any known pretraining-data limitations when reporting results. Long reviews also need a stated truncation or chunking policy; a model cannot use the text it never receives.

## What a fair evaluation would require
{: #model-evaluation-and-comparison }

Use the same training, validation, and test review IDs across the four approaches, while allowing each model its own preprocessing. Keep test reviews out of fitting and tuning, including unsupervised steps such as vocabulary building or Word2Vec training in this setup.

Choose hyperparameters and any decision threshold on validation data. Once those choices are fixed, evaluate on the held-out test set. Record accuracy and F1, state which class is positive and how F1 is averaged, and inspect the confusion matrix. Save predictions so the reported scores can be checked later.

A reproducible comparison would also need split IDs, seeds, package versions, model settings, and saved checkpoints. Timing needs its own measurement: training time and inference latency or throughput, with hardware, batch size, and input lengths recorded. Repeated runs would help show how much the result depends on initialization.

## What can be compared here
{: #results-and-performance-comparison }

There are no measured results in this revision. This table compares the proposed designs.

| Approach | Review representation and classifier | Main limitation to examine |
| --- | --- | --- |
| TF-IDF + logistic regression | Sparse term weights, linear classifier | Little word order beyond chosen n-grams |
| Word2Vec + logistic regression | Mean of word vectors, linear classifier | Word order is lost; unknown words need a policy |
| BiLSTM | Ordered embeddings, recurrent encoder and classification layer | Sequence length, padding, and training choices |
| BERT | Pretrained contextual representations and classification head | Checkpoint choice, input limit, and fine-tuning choices |

For a deployment decision, measure on the text and hardware the application will actually use. A movie-review score cannot tell us how well the same model will handle customer complaints, and a model's name cannot tell us its serving latency.

## Where this leaves the comparison
{: #conclusion }

The open question is whether the extra context available to a sequence model improves the result enough to justify its cost for a particular task. Answering that needs recorded runs. A reproducible benchmark remains a separate project; this article stops at explaining the approaches and the experiment they would need.
