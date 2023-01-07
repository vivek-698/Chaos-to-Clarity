import pandas as pd
import requests

model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
hf_token = "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"

API_URL = "https://api-inference.huggingface.co/models/" + model
headers = {"Authorization": "Bearer %s" % (hf_token)}

# df = pd.read_csv("training.1600000.processed.noemoticon.csv")
# tweets = df.iloc[:, -1]
df = pd.read_csv("tweets_5000.csv.xls")
tweets = df.iloc[:, 4]
print(tweets)


def analysis(data):
    payload = dict(inputs=data, options=dict(wait_for_model=True))
    response = requests.post(API_URL, headers=headers, json=payload)
    print(response)
    return response.json()


tweets_analysis = []
# gets rate limited sometimes
for tweet in tweets[:20]:
    try:
        sentiment_result = analysis(tweet)[0]
        top_sentiment = max(
            sentiment_result, key=lambda x: x["score"]
        )  # Get the sentiment with the higher score
        tweets_analysis.append({"tweet": tweet, "sentiment": top_sentiment["label"]})
        print("Tweet: ", tweet, "\nSentiment", top_sentiment["label"], "\n")

    except Exception as e:
        print(e)

sentiment_df = pd.DataFrame(tweets_analysis)
# print(sentiment_df.to_string())
