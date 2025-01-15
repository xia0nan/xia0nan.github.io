---
layout: post
title: Sentiment Analysis from Word2Vec to BERT
---

This is a comprehensive walkthrough about sentiment analysis task, using techniques from shallow word embeddings like word2vec to the state of art BERT and its descendants.

EDA
---

Exploratory Data Analysis (EDA) is a crucial first step in sentiment analysis. Begin by understanding the dataset. Key steps include:

```
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Analyze dataset
print("Training samples:", len(x_train))
print("Test samples:", len(x_test))
print("Classes:", set(y_train))

# Plot review lengths
review_lengths = [len(x) for x in x_train]
plt.hist(review_lengths, bins=50)
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.title('Distribution of Review Lengths')
plt.show()
```

Preprocessing
-------------

Text preprocessing ensures the data is clean and consistent. Common steps include:

```
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Pad sequences to ensure uniform length
maxlen = 200
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print("Padded sequence shape:", x_train.shape)
```

Convert Word to Vector
----------------------

### Word2Vec

-   Use Word2Vec to create word embeddings.

```
from gensim.models import Word2Vec

# Example sentences from IMDB dataset
sentences = [[str(word) for word in x] for x in x_train[:1000]]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Access word vectors
print(word2vec_model.wv['1'])  # Vector for word '1'
```

Convert Sentence to Vector
--------------------------

### Sentence Embeddings

-   Use average word embeddings to represent a sentence.

```
import numpy as np

def sentence_to_vector(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Example
sentence_vector = sentence_to_vector(sentences[0], word2vec_model)
print("Sentence vector shape:", sentence_vector.shape)
```

Combine with NLP Features
-------------------------

-   Augment embeddings with traditional NLP features, such as:

```
from sklearn.feature_extraction.text import TfidfVectorizer

# Example with raw text
vectorizer = TfidfVectorizer(max_features=5000)
x_tfidf = vectorizer.fit_transform([' '.join(map(str, x)) for x in x_train])
print("TF-IDF feature matrix shape:", x_tfidf.shape)
```

Feed in Classifier
------------------

-   Use classifiers to predict sentiment.

```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Train a Logistic Regression model
classifier = LogisticRegression()
classifier.fit(x_tfidf, y_train[:len(x_tfidf)])

# Evaluate
x_test_tfidf = vectorizer.transform([' '.join(map(str, x)) for x in x_test])
predictions = classifier.predict(x_test_tfidf)
print(classification_report(y_test[:len(predictions)], predictions))
```

LSTM Based Approach
-------------------

### Recurrent Neural Networks (RNN)

-   Implement an LSTM model for sentiment analysis.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=maxlen),
    LSTM(128, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
```

End-to-End Approach
-------------------

### Transformer Models

-   Fine-tune BERT for sentiment analysis using Hugging Face Transformers.

```
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize input
encoded_train = tokenizer([' '.join(map(str, x)) for x in x_train[:5000]], truncation=True, padding=True, max_length=128, return_tensors='tf')

# Compile and train
bert_model.compile(optimizer=Adam(learning_rate=5e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
bert_model.fit(encoded_train['input_ids'], y_train[:5000], epochs=2, batch_size=16)
```

Results Analysis
----------------

-   Evaluate and visualize results.

```
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion Matrix
cm = confusion_matrix(y_test[:len(predictions)], predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

Conclusion
----------

Sentiment analysis has evolved from basic embeddings to complex transformer models. Word2Vec and LSTM approaches are foundational but effective. BERT and its descendants have revolutionized the task with superior contextual understanding. The choice of approach depends on the dataset, computational resources, and required accuracy. Experimentation and analysis remain key to achieving optimal results.
