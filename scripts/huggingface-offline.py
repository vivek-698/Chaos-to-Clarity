from transformers import pipeline
import pandas as pd
import torch
import torch.nn.functional as F

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
classifier = pipeline("sentiment-analysis", model=model_name)

df = pd.read_csv("headings.csv")

headings = df["title"]


tweet_analysis = []
for text in headings:
    results = classifier(text)
    print(text, "\n", results)
    tweet_analysis.append(results[0]["label"])

print(tweet_analysis)

df["Sentiment"] = tweet_analysis
df.to_csv("headings-with-sentiment", index=False, encoding="utf-8")
print("Finished exporting")
print(df.head())
