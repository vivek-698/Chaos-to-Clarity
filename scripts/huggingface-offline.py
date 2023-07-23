from transformers import pipeline
import pandas as pd
import torch
import torch.nn.functional as F

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
classifier = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

df = pd.read_csv("./data/headings.csv")
headings = df["title"]


heading_analysis = []
for i,text in enumerate(headings):
    results = classifier(text)
    if results[0]["label"] == "POS":
        res = "Positive"
    elif results[0]["label"] == "NEG":
        res = "Negative"
    else:
        res = "Neutral"
    heading_analysis.append(res)
    print(i)


df["Sentiment"] = heading_analysis
df.to_csv("headings-with-sentiment.csv", index=False, encoding="utf-8")
print("Finished exporting")