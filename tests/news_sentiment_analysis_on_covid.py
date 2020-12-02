import tweepy
import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import re
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-bright')

keyword = 'covid'

#Enter your tweeter keys here

APIKey = 'YOUR_APIKEY'
APISecretKey = 'YOUR_SECRET_KEY'
Access_Token = 'YOUR_ACCESS_TOKEN'
Access_Token_Secret = 'YOUR_ACCESS_TOKEN_SECRET_KEY'

authenticate = tweepy.OAuthHandler(consumer_key=APIKey,consumer_secret=APISecretKey)
authenticate.set_access_token(Access_Token,Access_Token_Secret)
api = tweepy.API(authenticate,wait_on_rate_limit=True)


posts = api.search(keyword, count=200, lang='en', exclude= 'retweets',tweet_mode='extended')

lst_tweet=[]

for tweet_ET in posts:
    lst_tweet.append(tweet_ET.full_text )

df_tweet = pd.DataFrame(lst_tweet,columns=['tweets'])

def clnText(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    return text

df_tweet['tweets'] = df_tweet['tweets'].apply(clnText)

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

df_tweet['subjectivity'] = df_tweet['tweets'].apply(getSubjectivity)
df_tweet['polarity'] = df_tweet['tweets'].apply(getPolarity)

df_tweet['Analysis'] = ['Positive' if x > 0 else 'Negative' for x in df_tweet['polarity']]
df_tweet['Analysis'] = np.select([df_tweet['polarity']>0,df_tweet['polarity']<0],['Positive','Negative'],'Neutral')

# df_tweet['Analysis'].value_counts().plot(kind='bar', color='Orange')
# plt.xlabel('Sentiments')
# plt.ylabel('Percentage')
# plt.title('Sentiments Analysis')
# plt.gcf().set_facecolor("white")
# plt.show()


plt.scatter(df_tweet['polarity'],df_tweet['subjectivity'], color='Blue')
plt.xlabel('Sentiments')
plt.ylabel('Strength')
plt.title('Twitter Sentiment Analysis')
plt.grid()
plt.gcf().set_facecolor("white")
plt.show()

my_stopwords = ['will','take',keyword]+list(STOPWORDS)
allwords =  ' '.join([tweets for tweets in df_tweet['tweets']])
wordcloud = WordCloud(stopwords=my_stopwords,width=600,height=400,random_state=21,max_font_size=100, background_color='white').generate(allwords)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()
