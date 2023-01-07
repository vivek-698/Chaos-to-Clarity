from transformers import pipeline
import pandas as pd
import torch
import torch.nn.functional as F

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
classifier = pipeline("sentiment-analysis", model=model_name)

df = pd.read_csv("tweets_5000.csv.xls")
df = df.dropna()
tweets = df.iloc[:, 4]


tweet_analysis = []
for tweet in tweets:
    results = classifier(tweet)
    print(tweet, "\n", results)
    tweet_analysis.append(results[0]["label"])

print(tweet_analysis)

df["Sentiment"] = tweet_analysis
df.to_csv("vivek_1000_post_sentiment_gathering.csv")
print("Finished exporting")
print(df.head())
