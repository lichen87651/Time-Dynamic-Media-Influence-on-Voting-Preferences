import pandas as pd
from functools import reduce
from transformers import pipeline
from tqdm import tqdm

news_sentiment = 'nela_preprocessed_sample.csv'
news_sentiment = pd.read_csv(news_sentiment, low_memory=False)


# Load a multi-class classification pipeline - if the model runs on CPU, comment out "device"
classifier = pipeline("text-classification", model="classla/multilingual-IPTC-news-topic-classifier", device=0, max_length=512, truncation=True)

# Parameters
batch_size = 100
n = len(news_sentiment)
iptc_topics = []
iptc_scores = []

# Batched processing
for i in tqdm(range(0, n, batch_size)):
    batch = news_sentiment['content'].iloc[i:i + batch_size].tolist()
    results = classifier(batch)
    iptc_topics.extend([r['label'] for r in results])
    iptc_scores.extend([r['score'] for r in results])

# Assign results back to DataFrame
news_sentiment['IPTC_topic'] = iptc_topics
news_sentiment['IPTC_topic_score'] = iptc_scores

# Save to CSV
news_sentiment.to_csv("nela_preprocessed_iptc.csv", index=False)