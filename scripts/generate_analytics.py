import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from transformers import pipeline



class Analytics:
    # initialize dataframes of headings and comments to be used later
    # in production let this fetch from the mongoDB database instead of csv
    def __init__(self, name) -> None:
        print(f"..Fetching analytics for {name}")
        self.headings = pd.read_csv("./data/headings.csv")
        self.comments = pd.read_csv("./data/comments.csv")
        # Run some preinitiazing of the data to make it in the correct format for later analysis

    def mentionsOverTime(self, start_date="2023-01-18", end_date="2023-07-20", weekly=False):
        combined_df = pd.concat([self.headings, self.comments])

        combined_df['date'] = pd.to_datetime(combined_df['date'], format="%d %B %Y, %I:%M:%S %p %Z")

        # Filter the data based on the start and end dates
        mask = (combined_df['date'] >= start_date) & (combined_df['date'] <= end_date)
        combined_df = combined_df[mask]

        # Aggregate the number of posts per day or week based on the 'weekly' parameter
        if weekly:
            combined_df['date_agg'] = combined_df['date'].dt.to_period('W')
            posts_per_time = combined_df.groupby('date_agg').size().reset_index(name='num_posts')
            posts_per_time['date_agg'] = posts_per_time['date_agg'].dt.strftime("%Y-%m-%d")  # Convert to string
            time_label = 'Week'
        else:
            combined_df['date_agg'] = combined_df['date'].dt.date
            posts_per_time = combined_df.groupby('date_agg').size().reset_index(name='num_posts')
            time_label = 'Day'


        # Convert date_agg back to pandas datetime object
        posts_per_time['date_agg'] = pd.to_datetime(posts_per_time['date_agg'])

        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot the data as LINE GRAPH
        ax.plot(posts_per_time['date_agg'], posts_per_time['num_posts'], color='#ff6314')

        # Format the X-axis date labels
        if weekly:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        else:
            plt.xticks(rotation=45)

        plt.xlabel(f'{time_label}')
        plt.ylabel('Number of Posts')
        plt.title('Number of Mentions Over Time')
        plt.tight_layout()
        plt.show()


    # Read from the self.headings and self.comments and generate sentiment and append it to the datafram NOT the csv
    # def generateSentiment(self):
    #     MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    #     HF_TOKEN = "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"
    #     API_URL = "https://api-inference.huggingface.co/models/" + MODEL
    #     headers = {"Authorization": "Bearer %s" % (HF_TOKEN)}


    #     def analyze_sentiment(data):
    #         payload = dict(inputs=data, options=dict(wait_for_model=True))
    #         response = requests.post(API_URL, headers=headers, json=payload)
            
    #         sentiment_result = response.json()[0]
    #         top_sentiment = max(sentiment_result, key=lambda x: x["score"])
    #         print(top_sentiment["label"])
    #         return top_sentiment["label"]
        
    #     print("Started sentiment analysis for headings...")
    #     self.headings["sentiment"] = self.headings["title"].apply(analyze_sentiment)
    #     print("...Finished Sentiment analysis")

    #     # Just in case it gets rate limited in the future let me save this to another file
    #     self.headings.to_csv("headings-with-sentiment", index=False, encoding="utf-8")
    #     print("Saved to local file just in case")


    def commentSentiment(self):
        # download classifier
        classifier = pipeline(model="bhadresh-savani/distilbert-base-uncased-emotion")

        comment_body = self.comments["body"]
        comment_analysis = []
        for i,comment in enumerate(comment_body):
            try:
                result = classifier(comment)
                res = result[0]["label"]
            except:
                res = "Neutral"
            # print(result[0]["label"],"-----",comment)
            
            print(comment)
            comment_analysis.append(res)

        self.comments["Sentiment"] = comment_analysis
        self.comments.to_csv("comments-with-sentiment.csv", index=False, encoding="utf-8")
        print("exported self.comments")




    def drawPieChart(self,data):
        sentiment_counts = data.value_counts()
        # print(sentiment_counts)
        colors = [ '#FFAE42', '#ED2939','#3BB143'] # neutral, negative, positive
        # Set up the figure and axes
        fig, ax = plt.subplots()
        # Plot the pie chart
        ax.pie(sentiment_counts, labels=sentiment_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Sentiment Distribution')
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        plt.show()


    # not working for some reason
    def wordCloud(self):
        data = self.headings["title"].to_string(index=False,header=False)
        # print(len(data))
        stop_words = ["https", "co", "RT"] + list(STOPWORDS)
        wordcloud = WordCloud().generate(data)
        plt.figure()
        plt.title("Positive Tweets - Wordcloud")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()





A = Analytics("Elon Musk")
# A.generateSentiment()
# A.mentionsOverTime(start_date="2023-05-01", end_date="2023-08-01") 
df = pd.read_csv("./data/headings-with-sentiment.csv")
# A.mentionsOverTime()
# A.drawPieChart(df["Sentiment"])
# A.wordCloud()