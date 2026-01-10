
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

import pandas as pd

reddit_df = pd.read_csv("Reddit_Data.csv")
twitter_df = pd.read_csv("Twitter_Data.csv")
youtube_df = pd.read_csv("YoutubeCommentsDataSet.csv")
sentiment_df = pd.read_csv("sentimentdataset.csv")

print("Reddit columns:", reddit_df.columns)
print("Twitter columns:", twitter_df.columns)
print("YouTube columns:", youtube_df.columns)
print("Sentiment dataset columns:", sentiment_df.columns)

import pandas as pd

reddit_df = pd.read_csv("Reddit_Data.csv")
twitter_df = pd.read_csv("Twitter_Data.csv")
youtube_df = pd.read_csv("YoutubeCommentsDataSet.csv")
sentiment_df = pd.read_csv("sentimentdataset.csv")

reddit_df = reddit_df.rename(columns={
    "clean_comment": "text",
    "category": "sentiment"
})

twitter_df = twitter_df.rename(columns={
    "clean_text": "text",
    "category": "sentiment"
})

youtube_df = youtube_df.rename(columns={
    "Comment": "text",
    "Sentiment": "sentiment"
})

sentiment_df = sentiment_df.rename(columns={
    "Text": "text",
    "Sentiment": "sentiment"
})

def normalize_sentiment(val):
    if val == 2 or val == "2" or val == "positive" or val == "Positive":
        return "positive"
    elif val == 1 or val == "1" or val == "neutral" or val == "Neutral":
        return "neutral"
    else:
        return "negative"

reddit_df['sentiment'] = reddit_df['sentiment'].apply(normalize_sentiment)
twitter_df['sentiment'] = twitter_df['sentiment'].apply(normalize_sentiment)
youtube_df['sentiment'] = youtube_df['sentiment'].apply(normalize_sentiment)
sentiment_df['sentiment'] = sentiment_df['sentiment'].apply(normalize_sentiment)

reddit_df['source'] = 'Reddit'
twitter_df['source'] = 'Twitter'
youtube_df['source'] = 'YouTube'
sentiment_df['source'] = 'Mixed'

reddit_df = reddit_df[['text', 'sentiment', 'source']]
twitter_df = twitter_df[['text', 'sentiment', 'source']]
youtube_df = youtube_df[['text', 'sentiment', 'source']]
sentiment_df = sentiment_df[['text', 'sentiment', 'source']]

data = pd.concat(
    [reddit_df, twitter_df, youtube_df, sentiment_df],
    ignore_index=True
)

data.head()
data.shape

data = data.dropna(subset=['text', 'sentiment'])
data = data.drop_duplicates(subset=['text'])

data['clean_text'] = data['text'].apply(clean_text)

data['sentiment'].value_counts()

X = data['clean_text']     # input text
y = data['sentiment']      # labels

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,   # keeps model efficient
    ngram_range=(1,2)    # unigrams + bigrams (better accuracy)
)

X_tfidf = tfidf.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred, labels=['negative','neutral','positive'])

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['negative','neutral','positive'],
            yticklabels=['negative','neutral','positive'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

def predict_sentiment_ml(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    return nb_model.predict(vec)[0]

predict_sentiment_ml("I really loved this video")
predict_sentiment_ml("This was a terrible experience")

!pip install transformers torch --quiet

from transformers import pipeline

bert_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

bert_pipeline("I really love this product!")

def bert_predict(text):
    # Truncate text to fit within BERT's max sequence length (512 tokens)
    truncated_text = text[:512]
    result = bert_pipeline(truncated_text)[0]
    return result['label'], result['score']

bert_predict("This is the worst experience ever")

bert_data = data.sample(1000, random_state=42)  # safe size

bert_data['bert_sentiment'] = bert_data['text'].apply(
    lambda x: bert_predict(x)[0]
)

comparison_df = bert_data[['text', 'sentiment', 'bert_sentiment']]
comparison_df.head()

bert_data['bert_sentiment'] = bert_data['bert_sentiment'].str.lower()

import matplotlib.pyplot as plt

bert_data['bert_sentiment'].value_counts().plot(kind='bar')
plt.title("BERT Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

sample_text = "The video was good but the audio quality was bad"

print("ML Prediction:", predict_sentiment_ml(sample_text))
print("BERT Prediction:", bert_predict(sample_text))
