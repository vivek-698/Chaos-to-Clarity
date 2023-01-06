import pandas as pd
df=pd.read_csv('./vivek_1000_post_sentiment_gathering.csv')
pcount=0
ncount=0
ntrcount=0
t=0
for i in df['Sentiment']:
    t+=1
    if (i=='Positive'):
        pcount+=1
    elif (i=='Negative'):
        ncount+=1
    else  :
        ntrcount+=1
         