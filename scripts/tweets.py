import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "elon musk"
tweets = []
limit = 1000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit:
        break
    else:
        print(tweet.date, tweet.username, tweet.content)
        tweetscontent = tweet.renderedContent.split(" ")
        i = 0
        while i != len(tweetscontent):
            if "@" in tweetscontent[i]:
                tweetscontent.remove(tweetscontent[i])

            else:
                i += 1
        tweetscontent = ",".join(tweetscontent)
        tweetscontent = tweetscontent.replace(",", " ")
        tweets.append(
            [tweet.date, tweet.id, tweet.username, tweetscontent, tweet.retweetCount]
        )
df = pd.DataFrame(tweets, columns=["Date", "Id", "User", "Tweet", "Retweet"])

# df['Tweet']=datacleaningarray
df.to_csv("tweets.csv")
