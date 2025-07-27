from nela_preprocessing import extract_candidate_sentences, candidate_keywords
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

news_df = 'nela_preprocessed_sample.csv'
news_df = pd.read_csv(news_df, low_memory=False)

# Load the sentiment model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Map sentiment labels to numerical values
# "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"
label_to_score = {"LABEL_0": -1,  "LABEL_1": 0,  "LABEL_2": 1}

# Function to compute classifier-based sentiment score for candidate-specific sentences
def huggingface_candidate_sentiment_score(text, keywords):
    candidate_text = extract_candidate_sentences(text, keywords)
    if not candidate_text.strip():
        return None
    result = classifier(candidate_text[:512])[0]  # Truncate to max token length
    label = result['label']
    score = result['score']
    return label_to_score.get(label, 0) * score  # e.g., +0.85, -0.92, 0.0

# Add new columns based on Hugging Face sentiment
for candidate, keyword_list in candidate_keywords.items():
    col_name = f"{candidate}_hf_sentiment_score"
    news_df[col_name] = news_df["content"].apply(lambda x: huggingface_candidate_sentiment_score(x, keyword_list))

# Save results to a CSV
news_df.to_csv("nela_preprocessed_sentiment.csv", index=False)