---
layout: post
title: "Sentiment Analysis from Word2Vec to BERT: Comparing Approaches"
archived: true
hidden: true
permalink: /2020/04/25/sentiment-analysis-from-word2vec-to-bert.html
last_modified_at: 2026-09-22
---

> **Update — 22 September 2026:** The earlier version included incomplete code and unsupported accuracy figures. Those figures have been removed. This revision includes code excerpts and a completed Colab notebook with saved validation outputs from a new quick-mode run. These are not historical results or a full IMDb benchmark; the test set remains untouched, and BERT's checkpoint-loading warning is retained below.

<span id="table-of-contents"></span>

## What are we comparing?
{: #overview }

A movie review can call the acting good and still dislike the movie. Turning that review into a single positive or negative label means deciding how much of the wording, order, and context the model gets to see.

Here are four ways to do it: TF-IDF with logistic regression, averaged Word2Vec with logistic regression, a bidirectional LSTM, and BERT. The useful comparison starts with what each model receives as input. The companion notebook implements each approach on the same training and validation subsets; its saved results show what happened in one small run.

## Read the completed notebook
{: #follow-the-implementation-in-colab }

**[View the complete walkthrough on nbviewer](https://nbviewer.org/github/xia0nan/xia0nan.github.io/blob/master/notebooks/sentiment-analysis-from-word2vec-to-bert.ipynb).** It displays the code alongside the saved tables, review-length plot, confusion matrices, and training output. You can read it without installing packages or running a Colab session.

[Open the shared Colab notebook](https://colab.research.google.com/drive/1tFnKYPX37N7kmoqgKaEtL0qfo1E1DK33?usp=sharing) · [View source on GitHub](https://github.com/xia0nan/xia0nan.github.io/blob/master/notebooks/sentiment-analysis-from-word2vec-to-bert.ipynb) · [Download the notebook]({{ '/notebooks/sentiment-analysis-from-word2vec-to-bert.ipynb' | relative_url }})

The notebook covers dataset loading, model-specific preprocessing, all four training pipelines, validation-based selection, optional test evaluation, and artifact exports. The Python excerpts below come from that implementation and depend on the imports, data, and helpers defined in the notebook; they are not standalone scripts.

The saved run completed in Colab on 22 September 2026 (Singapore). It used `MODE = "quick"`, 2,000 training reviews, 500 validation reviews, seed 42, and one epoch for each neural model. Runtime output records Python 3.13.15, Transformers 5.16.1, and PyTorch 2.11.0+cu128, with a T4 GPU identified in notebook metadata. The executed setup installs unpinned packages, so future Colab environments may differ.

## The IMDb dataset
{: #dataset-and-exploratory-data-analysis }

The [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/) has 25,000 labeled training reviews and 25,000 labeled test reviews. Each split is balanced between positive and negative sentiment. There are also 50,000 unlabeled reviews, which are outside the setup described here.

The notebook first makes a stratified 80/20 split of the labeled training data, then draws the quick-mode subsets from those partitions. The saved output confirms 1,000 reviews per class in training and 250 per class in validation. Only the training subset is used to learn vocabularies, preprocessing statistics, and model parameters. Validation selects settings; `RUN_TEST = False` leaves the official test set unused.

The training-length analysis reports a median of 175 word tokens and a maximum of 1,099. With a 256-word limit, 28.6% of this training subset is truncated for the BiLSTM. BERT also uses a length limit of 256, but counts subwords and special tokens rather than words, so its coverage differs.

## Preprocessing depends on the model
{: #text-preprocessing }

Removing words can remove the answer. “Not good” should not become “good” because a stopword list happened to include “not.” Keep negation, and check what the tokenizer does with contractions before applying it to the whole dataset.

For TF-IDF and Word2Vec, decide how to handle case, punctuation, and HTML markup, then apply the same rules to every split. Learn vocabularies, document frequencies, and any locally trained word vectors from the training subset only. For a BiLSTM, fit its token vocabulary there too, with an unknown-token rule for words seen later.

For BERT, pass the review text to the tokenizer supplied with the chosen checkpoint. It handles the model's subword vocabulary and special tokens. The old stopword-removal and lemmatization pipeline does not belong in front of it. Padding and truncation still need explicit choices.

## Start with TF-IDF and logistic regression

TF-IDF represents a review through its terms and their weights. Logistic regression learns how those features relate to the sentiment label. Word bigrams can preserve short phrases such as “not good,” although this still loses most sentence structure. The [scikit-learn text-feature guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) explains the representation and its limits.

```python
vectorizer = TfidfVectorizer(
    tokenizer=word_tokens, token_pattern=None, lowercase=False,
    ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True,
)
x_train = vectorizer.fit_transform(train["text"])
x_val = vectorizer.transform(val["text"])

tfidf_clf, tfidf_c = choose_logistic(
    x_train, train["label"], x_val, val["label"]
)
```

The notebook's `word_tokens` helper preserves negation and contractions. Its `choose_logistic` helper tries `C = 0.1, 1.0, 10.0` and selects by validation F1; this run selected `C = 10.0`. The n-gram settings stay fixed. This gives the other approaches a baseline to compare against. Moving to embeddings does not, by itself, establish an improvement.

## Averaged Word2Vec still needs a classifier
{: #word-embeddings-approach }

[Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) learns a vector for each word from its surrounding words. Those vectors are features, not positive or negative predictions. One way to represent a whole review is to average its known word vectors, then train logistic regression on the resulting review vectors.

```python
def mean_vectors(token_lists):
    vectors = np.zeros((len(token_lists), w2v.vector_size), dtype=np.float32)
    empty = 0
    for i, tokens in enumerate(token_lists):
        known = [w2v.wv[token] for token in tokens if token in w2v.wv]
        if known:
            vectors[i] = np.mean(known, axis=0)
        else:
            empty += 1
    return vectors, empty
```

The notebook trains 100-dimensional skip-gram vectors for five epochs on training reviews only, with `window=5` and `min_count=2`. It then fits a scaler and logistic regression on the averaged training vectors. Validation selected `C = 0.1`.

The zero-vector fallback needs to be counted and inspected: a review with no known words gives the classifier no useful text features. This run found no such reviews in either subset. Averaging still throws away word order. Each occurrence of a word contributes the same vector, whatever the sentence says around it.

The earlier code tried to multiply a vocabulary-sized TF-IDF array by a token-sized matrix of word vectors. Those dimensions do not generally match. Simple averaging makes the proposed representation clear without carrying that broken implementation forward.

## A bidirectional LSTM reads a sequence
{: #lstm-based-approach }

A BiLSTM receives an ordered sequence of token embeddings. Its forward and backward recurrent passes let the review representation depend on words before and after a token. A classification layer then maps that representation to sentiment. The companion uses PyTorch for both this model and BERT.

```python
class BiLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100, padding_idx=0)
        self.lstm = nn.LSTM(100, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(
            self.embedding(ids), lengths.cpu(),
            batch_first=True, enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return self.head(torch.cat([hidden[-2], hidden[-1]], dim=1))
```

The vocabulary is fitted on training reviews, with separate padding and unknown-token IDs. Packed sequences keep padding out of the recurrent computation. Concatenating the final 64-unit forward and backward states gives the 128 inputs to the classifier. The embeddings start randomly; the notebook includes the batching, training loop, gradient clipping, validation, and checkpoint restoration.

Keeping order gives this model information that averaging loses. It does not guarantee that the model will learn useful long-range relationships. Truncation can also cut out the part of a review that changes its meaning.

## Fine-tuning BERT
{: #bert-based-approach }

[BERT](https://aclanthology.org/N19-1423/) starts from a pretrained transformer whose token representations depend on the surrounding text. For sentiment classification, add a classification head and fine-tune on labeled reviews.

```python
trainer = Trainer(
    model=bert_model,
    args=bert_args,
    train_dataset=bert_train,
    eval_dataset=bert_val,
    processing_class=bert_tokenizer,
    data_collator=DataCollatorWithPadding(bert_tokenizer),
    compute_metrics=bert_metrics,
)
trainer.train()
```

The notebook loads `google-bert/bert-base-uncased` and its matching tokenizer, uses dynamic padding, and truncates to 256 subwords including special tokens. Training uses a learning rate of `2e-5`, batch size 8, and validation F1 for checkpoint selection. Its Transformers 5.x configuration uses `warmup_steps=0.1`. The [Hugging Face text-classification guide](https://huggingface.co/docs/transformers/main/tasks/sequence_classification) gives further implementation guidance; use documentation matching your installed version.

The saved output includes a checkpoint-loading warning about missing LayerNorm `weight`/`bias` keys and unexpected `gamma`/`beta` keys. That is separate from the expected initialization of the new sentiment head. The run produced predictions, but correct restoration of all fine-tuned parameters has not been verified. Its score below is provisional, and the notebook retains the warning for inspection.

Pretraining gives BERT a different starting point from the locally trained models above. Record the exact checkpoint and any known pretraining-data limitations when reporting results. Long reviews also need a stated truncation or chunking policy; a model cannot use the text it never receives.

## What a fair evaluation would require
{: #model-evaluation-and-comparison }

Use the same training, validation, and test review IDs across the four approaches, while allowing each model its own preprocessing. Keep test reviews out of fitting and tuning, including unsupervised steps such as vocabulary building or Word2Vec training in this setup.

Choose hyperparameters and any decision threshold on validation data. Once those choices are fixed, evaluate on the held-out test set. Record accuracy and F1, state which class is positive and how F1 is averaged, and inspect the confusion matrix. Save predictions so the reported scores can be checked later.

A reproducible comparison would also need split IDs, seeds, package versions, model settings, and saved checkpoints. Timing needs its own measurement: training time and inference latency or throughput, with hardware, batch size, and input lengths recorded. Repeated runs would help show how much the result depends on initialization.

## What the saved run shows
{: #results-and-performance-comparison }

These values come from the notebook's final saved prediction outputs on the same **500 validation reviews**, using threshold 0.5 and binary F1 with positive label 1. They are not test-set scores. Validation F1 was also used to choose settings, so this is a record of model selection rather than an independent evaluation after tuning.

| Approach | Validation accuracy | Validation F1 |
| --- | ---: | ---: |
| TF-IDF + logistic regression | 85.4% | 0.8543 |
| Mean Word2Vec + logistic regression | 78.2% | 0.7883 |
| BiLSTM | 54.8% | 0.5150 |
| BERT (provisional: checkpoint warning) | 88.2% | 0.8808 |

TF-IDF was a useful baseline in this run: its sparse features outperformed the averaged Word2Vec representation. The BiLSTM's one-epoch result was close to chance on this balanced subset; this run does not establish whether more training or different settings would close the gap. BERT recorded the highest score, but its loading warning needs investigation before relying on that comparison.

All four used one seed and a small training subset. The linear classifiers searched three regularization strengths, while the neural models ran for one epoch at a fixed learning rate. BERT also brings external pretraining. Those differences prevent this quick run from establishing a general ranking of model families.

The notebook preserves the displayed outputs, including confusion matrices and the length plot. The separate run manifest, split-ID files, prediction CSVs, and model checkpoints were not included in the supplied export and are not bundled with the article. The code writes them in Colab, but the exact resolved dataset and model revision SHAs are not printed in the saved notebook.

For a deployment decision, measure on the text and hardware the application will actually use. A movie-review score cannot tell us how well the same model will handle customer complaints, and a model's name cannot tell us its serving latency.

## Where this leaves the comparison
{: #conclusion }

This walkthrough now has inspectable code and a recorded run. It shows why a TF-IDF baseline belongs in the experiment, and how much the interpretation depends on training budget, preprocessing, and checkpoint handling. The next useful experiment would resolve the BERT loading warning, retain the full run artifacts, and compare repeated runs before a final held-out test evaluation. Until then, the saved validation scores describe this run alone.
