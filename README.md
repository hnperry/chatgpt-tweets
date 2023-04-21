# ChatGPT Tweets Sentiment Analysis 

![ChatGPT-Header](https://user-images.githubusercontent.com/116209783/233449495-96b3dfb1-3885-4612-bfef-cc44b14df5a1.jpg)

## 0.1 Intent
I am performing sentiment analysis on a dataset of tweets about chatGPT. The dataset can be found on Kaggle [here](https://www.kaggle.com/datasets/charunisa/chatgpt-sentiment-analysis).

## 0.2 Dataset Features
This dataset contains a month's worth of tweets about chatGPT sampled in early 2023. NLP sentiment analysis was performed on the tweets, assigning them 'good', 'neutral', or 'bad' sentiments.

**Labels (Sentiment)**
- Good = 1
- Neutral = 0
- Bad = -1

**Tweets**
- Textual data of the tweets

Let's check out the data

```
import pandas as pd

df=pd.read_csv("gpt_raw_tweets.csv")
print(df.head())
```

```
   Unnamed: 0                                             tweets   labels
0           0  ChatGPT: Optimizing Language Models for Dialog...  neutral
1           1  Try talking with ChatGPT, our new AI system wh...     good
2           2  ChatGPT: Optimizing Language Models for Dialog...  neutral
3           3  THRILLED to share that ChatGPT, our new model ...     good
4           4  As of 2 minutes ago, @OpenAI released their ne...      bad
```

## 1. Preprocessing
### 1.1 Clean the data

We don't need the "Unnamed:0" column, so let's drop that.

```
df=df.drop(df.columns[0], axis=1)
```

First, we will replace the sentiment labels with numeric representations. 1 for good, 0 for neutral, -1 for bad.

```
#A method to replace sentiment text data with numerical data
def encoder(x):
    if x=='good':
        return 1
    elif x=='bad':
        return -1
    else:
        return 0
        
df['labels'] = df['labels'].apply(encoder)
```

Now, we can check out the text data of the first 10 tweets to see what steps we should take in cleaning

```
print("Dirty tweets:")
for i in df.tweets.head(10):
    print(i)
```

```
Dirty tweets:
ChatGPT: Optimizing Language Models for Dialogue https://t.co/K9rKRygYyn @OpenAI
Try talking with ChatGPT, our new AI system which is optimized for dialogue. Your feedback will help us improve it. https://t.co/sHDm57g3Kr
ChatGPT: Optimizing Language Models for Dialogue https://t.co/GLEbMoKN6w #AI #MachineLearning #DataScience #ArtificialIntelligence\n\nTrending AI/ML Article Identified &amp; Digested via Granola; a Machine-Driven RSS Bot by Ramsey Elbasheer https://t.co/RprmAXUp34
THRILLED to share that ChatGPT, our new model optimized for dialog, is now public, free, and accessible to everyone. https://t.co/dyvtHecYbd https://t.co/DdhzhqhCBX https://t.co/l8qTLure71
As of 2 minutes ago, @OpenAI released their new ChatGPT. \n\nAnd you can use it right now 👇 https://t.co/VyPGPNw988 https://t.co/cSn5h6h1M1
Just launched ChatGPT, our new AI system which is optimized for dialogue: https://t.co/ArX6m0FfLE.\n\nTry it out here: https://t.co/YM1gp5bA64
As of 2 minutes ago, @OpenAI released their new ChatGPT. \n\nAnd you can use it right now \n \nhttps://t.co/kUcnWYhQ1b\n\n🤯 https://t.co/kCE59Xs0YG https://t.co/cSn5h6h1M1
ChatGPT coming out strong refusing to help me stalk someone but agreeing providing that someone is Waldo. https://t.co/CVIJERbW38
#0penAl just deployed a thing I've been helping build the last couple of months, it's a chatbot based on GPT 3. I'm really excited to share this vl\nhttps://t.co/zp7HniUxBu\nhttps://t.co/NISJLWhOMw
Research preview of our newest model: ChatGPT\n\nWe're trying something new with this preview: Free and immediately available for everyone (no waitlist!) https://t.co/0RDT7QNZRD
```

This text data is dirty needs to be cleaned! In the text, we can observe what needs to be removed: links, hashtags, emojis, @ symbols, newlines, extra spaces, etc... The below method uses regex to remove specific unwanted aspects I observed in the data.

```
import re

#A method to clean text data of the tweets
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

df = tweetcleaner(df, 'tweets')
df.to_csv("gpt_clean_tweets.csv")

#Print the first 10 cleaned tweets
print("Clean Tweets:")
for i in df.tweets.head(10):
    print(i)
```

After applying the tweetcleaner method, our tweets look like this:

```
Clean Tweets:
chatgpt optimizing language models for dialogue
try talking with chatgpt our new ai system which is optimized for dialogue your feedback will help us improve it
chatgpt optimizing language models for dialogue ai machinelearning datascience artificialintelligence ai ml article identified amp digested via granola machine driven rss bot by ramsey elbasheer
thrilled to share that chatgpt our new model optimized for dialog is now public free and accessible to everyone
as of minutes ago released their new chatgpt you can use it right now
just launched chatgpt our new ai system which is optimized for dialogue it out here
as of minutes ago released their new chatgpt you can use it right now
chatgpt coming out strong refusing to help me stalk someone but agreeing providing that someone is waldo
just deployed thing i've been helping build the last couple of months it's chatbot based on gpt i'm really excited to share this
research preview of our newest model chatgpt trying something new with this preview free and immediately available for everyone (no waitlist!)
```

These tweets are now much more suitable for working with. Now we are ready to continue with the analysis.

### 1.2 Wordcloud

For some EDA, I decided to create a wordcloud of the most frequent words in the tweet data.

```
df=pd.read_csv("gpt_clean_tweets.csv")

#Import stop words
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Create a variable that is a long string with all the words in the text column
df[['tweets']] = df[['tweets']].astype(str)
text = ' '.join(df['tweets'])

#Create wordcloud and print
cloud = WordCloud(collocations = False, width=1600, height=800, max_words=100, colormap='tab20b', background_color = 'white', random_state = 1, stopwords=stop_words).generate(text)
plt.imshow(cloud,interpolation='bilinear')
plt.axis("off")
plt.show()
```

![wordcloud](https://user-images.githubusercontent.com/116209783/233725463-6050eeb9-002b-49c8-8022-c92c6dc25da3.png)

### 1.3 Data Tokenization

Tokenization splits our twitter strings into individual tokens.

```
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
df['tokens'] = df['tweets'].apply(tokenizer.tokenize)
print(df.head())
```
```
                                             tweets  labels                                             tokens
0  chatgpt optimizing language models for dialogue         0  [chatgpt, optimizing, language, models, for, d...
1  try talking with chatgpt our new ai system whi...       1  [try, talking, with, chatgpt, our, new, ai, sy...
2  chatgpt optimizing language models for dialogu...       0  [chatgpt, optimizing, language, models, for, d...
3  thrilled to share that chatgpt our new model o...       1  [thrilled, to, share, that, chatgpt, our, new,...
4  as of minutes ago released their new chatgpt y...      -1  [as, of, minutes, ago, released, their, new, c...
```

### 1.3.1 Sentence Length

#### 1.3.2 Frequency Distribution

Another way of visualizing this data is to calculate the frequency distribution of each word and plot it as a bar chart. 

```
#Create list of tokens, removing stop words
import itertools
words = [word for word in df['tokens']]
words = list(itertools.chain.from_iterable(words))
words = [i for i in words if i not in stop_words]

#Calculate frequency distribution
from nltk.probability import FreqDist
top_50 = FreqDist(words).most_common(50)

#Plot
fig, ax = plt.subplots()
plt.figure(figsize = (10,10))
ax.bar(range(len(top_50)), [t[1] for t in top_50]  , align="center")
ax.set_xticks(range(len(top_50)))
ax.set_xticklabels([t[0] for t in top_50], rotation=45, ha="right")
plt.show()
```
![wordcounts](https://user-images.githubusercontent.com/116209783/233728834-64140916-190e-4a79-8826-cfbf5fc427c8.png)

### 1.4 Sentence Length

Our sentences have an average length of 10-12 words.

```
from nltk.tokenize import RegexpTokenizer

#NLTK tokenizer
tokenizer = RegexpTokenizer(r'\w+')
df['tokens'] = df['tweets'].apply(tokenizer.tokenize)
print(df.head())

#Check sentence lengths
length = [len(tokens) for tokens in df['tokens']]
print(type(length))

#plot
plt.figure(figsize = (10,10))
plt.title('Sentence Length')
plt.xlabel('Count of words')
plt.ylabel('Number of Tweets')
plt.hist(length)
plt.savefig("sentencelength.jpg")
plt.show()
```

![sentencelength](https://user-images.githubusercontent.com/116209783/233729778-94c8d373-d1fe-4298-8f88-c36b1392918d.jpg)

