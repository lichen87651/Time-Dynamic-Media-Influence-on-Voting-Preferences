import json
import pandas as pd
from functools import reduce
import os
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Download stopwords and punkt tokenizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

news = [
    "nela-gt-2020/newsdata/yahoonews.json",
    "nela-gt-2020/newsdata/cnn.json",
    "nela-gt-2020/newsdata/thenewyorktimes.json",
    "nela-gt-2020/newsdata/breitbart.json",
    "nela-gt-2020/newsdata/foxnews.json",
    "nela-gt-2020/newsdata/washingtonpost.json",
    "nela-gt-2020/newsdata/theguardian.json",
    "nela-gt-2020/newsdata/usatoday.json",
    "nela-gt-2020/newsdata/bbc.json",
    "nela-gt-2020/newsdata/npr.json",
    "nela-gt-2020/newsdata/buzzfeed.json"]

# List to store DataFrames for each specified resource
dataframes = []

# Load each specified JSON file into a DataFrame
for i, file in enumerate(news):
    # Load JSON file into a DataFrame
    df = pd.read_json(file)
    
    # Filter for the period from July to November 2020
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[(df['date'] >= '2020-08-01') & (df['date'] <= '2020-11-02')]
    df = df.filter(items=['id', 'date', 'source', 'content'])
    
    # Append the DataFrame to the list
    dataframes.append(df)

# concatenate all DataFrames into a single DataFrame
news_df = pd.concat(dataframes, ignore_index=True)

# Simplified topics and keywords
keywords = {
    "healthcare": ["medicare", "insurance", "covid", "vaccine", "healthcare", "pandemic", "public health"],
    "economy": ["unemployment", "stimulus", "jobs", "tax", "inflation", "wages", "economic recovery"],
    "social_issues": ["racial justice", "black lives matter", "abortion", "gun control", "police reform", "protests"]}

# Function to check if text matches keywords for a topic
def keyword_match(text, keyword_list):
    if pd.isnull(text):  # Handle missing values
        return 0  # Not relevant if text is missing
    text = text.lower()  # Convert to lowercase for case-insensitive matching
    for keyword in keyword_list:
        if re.search(r"\b" + re.escape(keyword) + r"\b", text):  # Match whole words
            return 1  # Relevant if any keyword matches
    return 0  # Not relevant if no keywords match

# Add columns for each topic, with 1 or 0 indicating relevance
for topic, kw_list in keywords.items():
    news_df[topic] = news_df["content"].apply(lambda x: keyword_match(x, kw_list))

# Define candidate keywords
candidate_keywords = {
    "biden": ["biden", "joe biden", "president biden", "joe"],
    "trump": ["trump", "donald trump", "president trump", "donald"]}

# Function to extract sentences mentioning a candidate
def extract_candidate_sentences(text, keywords):
    if pd.isnull(text):  # Handle missing values
        return ""
    text = text.lower()
    sentences = text.split(".")  # Split article into sentences
    relevant_sentences = [sentence for sentence in sentences if any(re.search(r"\b" + re.escape(keyword) + r"\b", sentence) for keyword in keywords)]
    return ". ".join(relevant_sentences)  # Join back into a string

# Candidate-specific sentiment scoring
def candidate_sentiment_score(text, keywords):
    if not text:  # Handle missing or empty text
        return None
    candidate_text = extract_candidate_sentences(text, keywords)
    if not candidate_text:  # If no relevant sentences are found
        return None
    sentiment_score = TextBlob(candidate_text).sentiment.polarity  # Get polarity score (-1 to 1)
    return sentiment_score

# Apply candidate-specific sentiment scoring
for candidate, keyword_list in candidate_keywords.items():
    news_df[candidate + "_sentiment_score"] = news_df["content"].apply(lambda x: candidate_sentiment_score(x, keyword_list))

# Drop rows where neither candidate is mentioned
news_df = news_df.dropna(subset=["biden_sentiment_score", "trump_sentiment_score"], how="all")

# Save results to a CSV
news_df.to_csv("nela_preprocessed_sample.csv", index=False)