"""
Vote
● V201029: who have already voted in the general election who they voted for.
● V201033: for respondents who had not yet voted and asks who they intend to vote for.
○ Joe Biden(1)
○ Donald Trump(2)

News mentioned
● V201634a: Yahoo News (www.yahoo.com/news)
● V201634b: CNN.com
X ● V201634c: Huffington Post (www.huffingtonpost.com)
● V201634d: New York Times (nytimes.com)
● V201634e: Breitbart News Network (breitbart.com)
● V201634f: Fox News (www.foxnews.com)
● V201634g: Washington Post (washingtonpost.com)
● V201634h: The Guardian (theguardian.com)
● V201634i: USA Today (usatoday.com)
● V201634j: BBC News (news.bbc.co.uk)
● V201634k: NPR News (npr.org)
X ● V201634m: Daily Caller (dailycaller.com)
X ● V201634n: Bloomberg (bloomberg.com)
● V201634p: Buzzfeed (buzzfeed.com)
X ● V201634q: NBC News (www.nbcnews.com)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# Load the ANES 2020 survey data
anes_survey = 'anes_timeseries_2020_csv_20220210/anes_timeseries_2020_csv_20220210.csv'
anes_df = pd.read_csv(anes_survey, low_memory=False)

# Define the columns of interest
news_mentioned = ['V201634a', 'V201634b', 'V201634d', 'V201634e', 'V201634f', 'V201634g', 'V201634h', 'V201634i', 'V201634j', 'V201634k', 'V201634p']

# states, age, gender, race, edu, income, vote_already, vote_preferred, begin_date, end_date, news_mentioned
anes_df = anes_df[['V201014b', 'V201507x', 'V201600', 'V201549x', 'V201510', 'V201617x', 'V201029', 'V201033', 'V203053', 'V203055', 
                'V201634a', 'V201634b', 'V201634d', 'V201634e', 'V201634f', 'V201634g', 'V201634h', 'V201634i', 'V201634j', 'V201634k', 'V201634p']] 
                
# Convert the columns to datetime
anes_df ['V203053'] = pd.to_datetime(anes_df['V203053'], format='%Y%m%d', errors='coerce')
anes_df ['V203055'] = pd.to_datetime(anes_df['V203055'], format='%Y%m%d', errors='coerce')

# Calculate the difference in days, filter out rows where the difference is more than 1 day
anes_df['date_diff'] = (anes_df['V203055'] - anes_df['V203053']).abs().dt.days
anes_df = anes_df[anes_df['date_diff'] <= 1].drop(columns=['date_diff', 'V203055'])

# Convert news into a binary indicator 
anes_df[news_mentioned] = anes_df[news_mentioned].map(lambda x: x if x == 1 else 0)

# Clean up vote
anes_df = anes_df[~((anes_df['V201029'] == -1) & (anes_df['V201033'] == -1))]
anes_df['vote'] = anes_df.apply(lambda row: row['V201033'] if row['V201033'] != -1 else row['V201029'], axis=1)
anes_df = anes_df.drop(['V201029', 'V201033'], axis=1)
anes_df = anes_df.loc[anes_df.vote.isin([1,2])]

# Rename columns
anes_df.rename(columns={'V201014b': 'state', 'V201507x': 'age', 'V201600': 'gender',
                        'V201549x': 'race', 'V201510': 'edu', 'V201617x': 'income', 'V203053': 'date'}, inplace=True)
anes_df.rename(columns={'V201634a': 'Yahoo', 'V201634b': 'CNN', 'V201634d': 'New York Times', 
                        'V201634e': 'Breitbart',  'V201634f':'Fox','V201634g': 'Washington Post', 'V201634h':'The Guardian', 
                        'V201634i': 'USA Today', 'V201634j': 'BBC', 'V201634k': 'NPR', 'V201634p': 'Buzzfeed'}, inplace=True)

# Convert categorical variables
anes_df  =  anes_df.replace({'race': {1: 'White (Non-Hisp.)',
                                      2: 'Black (Non-Hisp.)', 
                                      3: 'Hispanic', 
                                      4: 'Asian/PI (Non-Hisp.)',
                                      5: 'Other (Non-Hisp.) ',
                                      6: 'Multiple (Non-Hisp.)',
                                      0: 'Race_Unknown/Refused'},
                             'gender': {1: 'Female', 2: 'Male', -9: 'Sex_Refused'},
                             'state': {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California', 
                                       8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'Washington DC', 12: 'Florida',
                                       13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois', 18: 'Indiana',
                                       19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana', 23: 'Maine',
                                       24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota', 28: 'Mississippi',
                                       29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire',
                                       34: 'New Jersey', 35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota',
                                       39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 44: 'Rhode Island',
                                       45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee', 48: 'Texas', 49: 'Utah',
                                       50: 'Vermont', 51: 'Virginia', 53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin',
                                       56: 'Wyoming', -1: 'State_Refused'}})

# Remove invalid state values
anes_df = anes_df[~anes_df.state.isin([86, -9, -8])]
# Remove invalid race values
anes_df = anes_df[~anes_df.race.isin([-9, -8])]

# Replace missing education values (-9, -8, 95) with the median education level
anes_df['edu'] = anes_df['edu'].replace({-9: 0, -8: 0, 95: 0})
median_edu = anes_df.loc[anes_df['edu'] != 0, 'edu'].median() # Median Education: 5.0
anes_df['edu'] = anes_df['edu'].replace(0, median_edu)

# Remove income bad lines
anes_df = anes_df.loc[anes_df['income'] != -5]
# Replace missing income values (-9) with the median income
median_income = anes_df.loc[~anes_df['income'].isin([-9, -5]), 'income'].median() # Median Income: 13.0
anes_df['income'] = anes_df['income'].replace({-9: median_income, -5: median_income})

# Convert age values
def categorize_age(age):
    if age == -9:  
        return -9
    elif age == 80: 
        return 8.0
    elif 18 <= age < 20:
        return 1.0
    elif 20 <= age < 30:
        return 2.0
    elif 30 <= age < 40:
        return 3.0
    elif 40 <= age < 50:
        return 4.0
    elif 50 <= age < 60:
        return 5.0
    elif 60 <= age < 70:
        return 6.0
    elif 70 <= age < 80:
        return 7.0

# Apply the categorization to the 'age' column
anes_df['age_group'] = anes_df['age'].apply(categorize_age)
median_value = anes_df.loc[anes_df['age_group'] != -9, 'age_group'].median() # Median age: 5.0
anes_df['age_group'] = anes_df['age_group'].replace({-9: median_value})
anes_df = anes_df.drop(['age'], axis=1)

# Save results to a CSV
anes_df.to_csv("anes_preprocessed_data.csv", index=False)