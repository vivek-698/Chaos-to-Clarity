import pandas as pd
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt

df = pd.read_csv("vivek_1000_post_sentiment_gathering.csv")

# Wordcloud with positive tweets
positive_tweets = df["Tweet"][df["Sentiment"] == "Positive"]
stop_words = ["https", "co", "RT"] + list(STOPWORDS)
positive_wordcloud = WordCloud(
    max_font_size=50, max_words=50, background_color="white", stopwords=stop_words
).generate(str(positive_tweets))
plt.figure()
plt.title("Positive Tweets - Wordcloud")
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Wordcloud with negative tweets
negative_tweets = df["Tweet"][df["Sentiment"] == "Negative"]
stop_words = ["https", "co", "RT"] + list(STOPWORDS)
negative_wordcloud = WordCloud(
    max_font_size=50, max_words=50, background_color="white", stopwords=stop_words
).generate(str(negative_tweets))
plt.figure()
plt.title("Negative Tweets - Wordcloud")
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
