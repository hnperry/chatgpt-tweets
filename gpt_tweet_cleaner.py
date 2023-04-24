#Import relevant libraries
import pandas as pd

#Open the file and review
df=pd.read_csv("gpt_raw_tweets.csv")
print(df.head())

print(df['labels'].value_counts())

#plot
import matplotlib.pyplot as plt
df['labels'].value_counts().plot.bar(title="Sentiment Counts")
plt.xticks(rotation=0)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

#Drop the unnamed:0 column
df=df.drop(df.columns[0], axis=1)

#Data has positive, negative, and neutral responses
#Let's call good = 1, neutral = 0, bad = -1

#A method to replace sentiment text data with numerical data
def encoder(x):
    if x=='good':
        return 1
    elif x=='bad':
        return -1
    else:
        return 0

df['labels'] = df['labels'].apply(encoder)

#Check out the first 10 tweets
print("Dirty tweets:")
for i in df.tweets.head(10):
    print(i)

#Time to clean this dirty tweet data
#In the text data we observe need for cleaning:
#Links, hashtags, emojis, @s, newlines, extra spaces, etc.
import re

def tweetcleaner(df, column):
    #convert all text to lowercase
    df[column] = df[column].str.lower()
    #remove newlines
    df[column] = df[column].str.replace(r"\S+\\nhttp\S+", "", regex=True)
    df[column] = df[column].str.replace(r"(\\n*)\S+", " ", regex=True)
    #remove urls
    df[column] = df[column].str.replace(r"\S+http\S+", "", regex=True)

    df[column] = df[column].str.replace(r"http\S+", "", regex=True)
    df[column] = df[column].str.replace(r"http", "", regex=True)
    #remove @ signs
    df[column] = df[column].str.replace(r"@\S+", "", regex=True)
    #remove words with numbers inside words
    df[column] = df[column].str.replace(r'\w*[0-9]\w*', "", regex=True)
    #remove all other weird characters
    df[column] = df[column].str.replace(r"[^A-Za-z0-9()!?@\'\`\"\_]", " ", regex=True)
    #change multiple spaces to single spaces
    df[column] = df[column].str.replace(r'\s+', " ", regex=True)
    #remove all single characters
    df[column] = df[column].str.replace(r'\s+[a-zA-Z]\s+', ' ', regex=True)

    return df

print("Clean Tweets:")
df = tweetcleaner(df, 'tweets')
for i in df.tweets.head(10):
    print(i)

df.to_csv("gpt_clean_tweets.csv")