import pandas as pd

df=pd.read_csv("gpt_clean_tweets.csv")

#make word cloud
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Create a variable that is a long string with all the words in the text column
df['tweets'] = df['tweets'].astype(str)
text = ' '.join(df['tweets'])

#create wordcloud
cloud = WordCloud(collocations = False, width=1600, height=800, max_words=100, colormap='tab20b', background_color = 'white', random_state = 1, stopwords=stop_words).generate(text)

#Print wordcloud
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(cloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("wordcloud.jpg")
plt.show()

#Tokenzie the data
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
df['tokens'] = df['tweets'].apply(tokenizer.tokenize)
print(df.head())

#Check sentence lengths
length = [len(tokens) for tokens in df['tokens']]

#plot
plt.figure(figsize = (10,10))
plt.title('Sentence Length')
plt.xlabel('Count of words')
plt.ylabel('Number of Tweets')
plt.hist(length)
plt.savefig("sentencelength.jpg")
plt.show()

#Make list of tokens and remove stop words
import itertools
words = [word for word in df['tokens']]
words =list(itertools.chain.from_iterable(words))
words = [i for i in words if i not in stop_words]

#Calculate Frequency Distribution
from nltk.probability import FreqDist
top_30 = FreqDist(words).most_common(30)

#Plot
fig, ax = plt.subplots(figsize=(10,10))
fig.suptitle('Word Frequency', x=0.5, y=.9, size='x-large')
ax.set_xlabel('Word')
ax.set_ylabel('Count of Word')
ax.bar(range(len(top_30)), [t[1] for t in top_30]  , align="center")
ax.set_xticks(range(len(top_30)))
ax.set_xticklabels([t[0] for t in top_30], rotation=45, ha="right")
plt.show()

#bag of words vectorization
from sklearn.model_selection import train_test_split

#split features and targets
x = df['tweets']
y= df['labels']

#split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#to csv
x_train.to_csv('x_train_clean.csv')
x_test.to_csv('x_test_clean.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')


