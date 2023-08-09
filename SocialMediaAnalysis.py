### Generating data from REddit
import csv
import praw
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from transformers import pipeline
from datetime import datetime
import pandas as pd

## Libraries for search
import nltk
import numpy as np
import pandas as pd
import string
from nltk import ngrams
import re

import re
import nltk
import time
import string
import numpy as np
import pandas as pd
from nltk import ngrams
from tqdm.auto import tqdm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#nltk.download("punkt")
#nltk.download("stopwords")
#nltk.download("wordnet")
#nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize

# libraries for word cloud
import pandas as pd
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image



class generateData:
    def __init__(self) -> None:
        pass
    def RedditData(self,name):
        print(name)
        r = praw.Reddit(
            client_id="5nV8Wwj6VxKrQ9o-lo5qkg",
            client_secret="tNIL7JG0IWh2pbC60bFkOPe5evVZAQ",
            user_agent="Doomsday",
            username="test_and_doom",
            password="test@123",
        )

        # fetches the top 1000 posts from the subreddit
        top1k_posts = r.subreddit(name).hot(limit=10)

        attributes = [
            "title",
            "name",
            "score",
            "visited",
            "id",
            "author",
            "created_utc",
            "url",
            "upvote_ratio"
        ]

        comment_attributes = ["body", "ups", "created_utc", "score"]
        # ['selftext', 'saved',  'title', 'name', 'score','likes', 'view_count','visited', 'id', 'author', 'num_comments']
        with open("headings.csv", "w", newline="", encoding="utf-8") as headings_file, open(
            "comments.csv", "w", newline="", encoding="utf-8"
        ) as comments_file:
            writer1 = csv.writer(headings_file)
            writer1.writerow(attributes)

            writer2 = csv.writer(comments_file)
            writer2.writerow(["parent_post_id"] + comment_attributes)
            for post in top1k_posts:
                values1 = [getattr(post, attr) for attr in attributes]
                writer1.writerow(values1)

                post.comments.replace_more(limit=50)
                # print("Fetching ", min(50, len(post.comments.list()))," comments for post titled",post.title)
                # print(getattr(post,"id"))
                for comment in post.comments.list():
                    values2 = [getattr(post,"id")]
                    for attr in comment_attributes:
                        values2.append(getattr(comment,attr))

                    # values2 = [getattr(comment, attr) for attr in comment_attributes]
                    writer2.writerow(values2)




## Sentiment Analysis


class Analytics:
    # initialize dataframes of headings and comments to be used later
    # in production let this fetch from the mongoDB database instead of csv
    def __init__(self, name) -> None:
        print(f"..Fetching analytics for {name}")
        self.headings = pd.read_csv("./headings.csv")
        self.comments = pd.read_csv("./comments.csv")
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
            
            # print(comment)Elon Musk
            comment_analysis.append(res)

        self.comments["Sentiment"] = comment_analysis
        self.comments.to_csv("comments-with-sentiment.csv", index=False, encoding="utf-8")
        print("exported self.comments")




    def draw3PieChart(self,data):
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
        plt.savefig("piechart.png")

    def draw5PieChart(self,data):
        sentiment_counts = data.value_counts()[:-1] # we cut off the last Neutral part as that was generated due to exception while running the sentiment analysis
        # need to specify sepcific color for specific emotion
        colors = ['#ED2939' , '#FF9B9B','#567B89', "#311212",  "#FF819F","#00e8be"] # anger, joy, saddness, suprise, love, fear
        fig,ax = plt.subplots()
        wedges,texts,autotexts = ax.pie(sentiment_counts, colors=colors, autopct=lambda p: f'{p:.1f}%' if p >= 3 else '', startangle=90)
        ax.set_title('Comment Sentiment Distribution')
        ax.axis('equal')
        plt.legend(wedges, sentiment_counts.index, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout()
        plt.show()
        plt.savefig("piechart.png")


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

class Search:
    def _init_(slef):
        pass
    def search(self):
        stopwords = nltk.corpus.stopwords.words('english')
        stemmer = nltk.stem.SnowballStemmer('english')
        lemmatizer = nltk.stem.WordNetLemmatizer()
        df=pd.read_csv("headings.csv")
        text=df["title"]
        def tokenize_corpus(texts):
            return list(map(word_tokenize, texts))
        tokenized_corpus = tokenize_corpus(text)
        def remove_stopwords(tokenized_texts):
            _function = lambda text: [word for word in text if word not in stopwords]
            return list(map(_function, tokenized_texts))
        tokenized_nostopw_corpus = remove_stopwords(tokenized_corpus)
        def create_inverted_index(tokenized_texts):
            inverted_index = dict()
            for document_idx,text in enumerate(tokenized_texts):
                for word_pos,word in enumerate(text):
                    if word not in inverted_index:
                        inverted_index[word] =dict()
                    if document_idx not in inverted_index[word]:
                        inverted_index[word][document_idx]=list()
                    inverted_index[word][document_idx].append(word_pos)
            for word in inverted_index:
                for document_idx in inverted_index[word]:
                    inverted_index[word][document_idx]=sorted(inverted_index[word][document_idx])
                items = list(inverted_index[word].items())
                items.sort(key=lambda x: x[0])
                for k, v in items:
                    inverted_index[word][k] = v
            return inverted_index
        inverted_index = create_inverted_index(tokenized_nostopw_corpus)
        def case_folding(tokenized_text):
            _function = lambda text: [word.lower() for word in text]
            return list(map(_function, tokenized_text))
        tokenized_nostopw_case_corpus = case_folding(tokenized_nostopw_corpus)
        inverted_index = create_inverted_index(tokenized_nostopw_case_corpus)
        #Lemmatization
        def lemmatize_words(tokenized_text):
            _function = lambda text: [lemmatizer.lemmatize(word) for word in text]
            return list(map(_function, tokenized_text))
        tokenized_nostopw_case_lemm_corpus = lemmatize_words(tokenized_nostopw_case_corpus)
        inverted_index = create_inverted_index(tokenized_nostopw_case_lemm_corpus)
        #Remove punctutaions
        def remove_punct(tokenized_text):
            _function = lambda text: [word for word in text if (word not in string.punctuation and word.isalnum())]
            return list(map(_function, tokenized_text))
        No_punct_corpus = remove_punct(tokenized_nostopw_case_lemm_corpus)
        inverted_index = create_inverted_index(No_punct_corpus)
        #Remove numbers and numbers with characters'
#removing number
        sorted_dict_inverted_index={}
        for x in inverted_index.keys():
            if (x.isnumeric()==False):
                sorted_dict_inverted_index[x]=inverted_index[x]
        #Sort dictionary
        myKeys = list(sorted_dict_inverted_index.keys())
        myKeys.sort()
        sorted_dict = {i: sorted_dict_inverted_index[i] for i in myKeys}
        def parse_query(infix_tokens):
            precedence = {}
            precedence['NOT'] = 3
            precedence['AND'] = 2
            precedence['OR'] = 1
            precedence['and'] = 2
            precedence['or'] = 1
            precedence['('] = 0
            precedence[')'] = 0    

            output = []
            operator_stack = []

            for token in infix_tokens:
                if (token == '('):
                    operator_stack.append(token)

                elif (token == ')'):
                    operator = operator_stack.pop()
                    while operator != '(':
                        output.append(operator)
                        operator = operator_stack.pop()

                elif (token in precedence):
                    if (operator_stack):
                        current_operator = operator_stack[-1]
                        while (operator_stack and precedence[current_operator] > precedence[token]):
                            output.append(operator_stack.pop())
                            if (operator_stack):
                                current_operator = operator_stack[-1]
                    operator_stack.append(token) # add token to stack
                else:
                    output.append(token.lower())

            while (operator_stack):
                output.append(operator_stack.pop())

            return output
        def boolean_query(query, inverted_index):
                query = query.strip()
                query_tokens = query.split()
                boolean_query = parse_query(query_tokens)
                    
                result_stack = list()
                for idx, token in enumerate(boolean_query):
                    if token not in ["AND", "NOT", "OR","and","or"]:
                        result = set(inverted_index[token])
                    else:
                        if token in ['AND', 'OR',"and","or"]:
                            right_operand = result_stack.pop()
                            left_operand = result_stack.pop()
                            
                            if token == 'AND' or token == 'and':
                                operation = set.intersection
                            else:
                                operation = set.union
                            
                            result = operation(left_operand, right_operand)
                            
                        else:
                            operand = result_stack.pop()
                            complement_document_ids = inverted_index[boolean_query[idx-1]]
                            result = list()
                            for word in inverted_index:
                                result.extend([_id for _id in inverted_index[word] if _id not in complement_document_ids])
                            result = set(result)
                            
                    result_stack.append(result)
                
                return result_stack.pop()
        K=3
        
        def get_kgrams(word, k):
            word = f'${word.strip()}$'
            kgrams = ngrams(word, k)
            kgrams = list(map(lambda x: ''.join(x), kgrams))
            return kgrams


        def create_kgram_index(articles, k=3):
            kgram_index = dict()
            for article in articles:
                for token in article:
                    kgrams = get_kgrams(token, k)
                    for kgram in kgrams:
                        if kgram not in kgram_index:
                            kgram_index[kgram] = set()
                        kgram_index[kgram].add(token)
                        
            for kgram in kgram_index:
                kgram_index[kgram] = sorted(list(kgram_index[kgram]))
            
            return kgram_index
        No_number_corpus=[]


        for x in No_punct_corpus:
            for i in x:
                if (i.isnumeric()==True):
                    x.remove(i)
        kgram_index = create_kgram_index(No_punct_corpus, K)
        def search_kgram(query):
            if '*' in query:
                query_regex = query.replace('*', '.*')
                query_kgrams = get_kgrams(query, K)
                query_kgrams = list(filter(lambda x: not('*' in x), query_kgrams))
                search_words = list()
                for query_kgram in query_kgrams:
                    search_words.append(set(kgram_index[query_kgram]))
                
                search_words = list(set.intersection(*search_words))
                search_words = [w for w in search_words if re.match(query_regex, w).span()[1] == len(w)]
                
            else:
                search_words = [query]
            
            #print(search_words)
            return (list(search_words))
        results = search_kgram("el*")
        print(results)
        #function to retrieve all the kgram documents
        
class Wordcloud:
    def __init__(self) -> None:
        
    
        df = pd.read_csv("D:\\HALLOTHON\comments-with-sentiment.csv")
        type_of_sentiments=df["Sentiment"].unique()
        # print(type_of_sentiments)
        text=[]
        for i in type_of_sentiments:
            text.append(str(df["body"][df["Sentiment"]==str(i)]))
            print(text)
        
        postive_text=df["body"][df["Sentiment"]=="joy"]
        text=str(text[0])
        python_mask=np.array(PIL.Image.open("Old_Nike_logo.jpg"))
        # plt.imshow(py)
        wc= WordCloud(stopwords=STOPWORDS,mask=python_mask,contour_color="black",background_color="white",contour_width=3,min_font_size=3).generate(text)


        plt.imshow(wc)
        plt.axis("off")
        plt.show()
        plt.savefig("wordcloud.png")
        # print(text[0])

   
   
            
        
        
    

        

 

    

        
            
           
    
   



# Main

# GD=generateData()
# GD.RedditData("ElonMusk")
A = Analytics("Elon Musk")
print("done generating data")

A.commentSentiment()
# Below code was used to clean up the date column


# Function was used to convert the utc format from 1689878167 to Human Readable
def convertUTCHumanReadable(utc_initial):
    try:
        # Convert the UTC timestamp to a datetime object
        utc_datetime = datetime.utcfromtimestamp(utc_initial)

        # Format the datetime object to a human-readable date format
        formatted_date = utc_datetime.strftime("%d %B %Y, %I:%M:%S %p UTC")
        # if you need to mention the day, then add a %A flag before the %d. Put a comma between them ,
        return formatted_date
    except Exception as e:
        return str(e)

# Below code was used to clean up the date column
headings = pd.read_csv("./headings.csv")
comments = pd.read_csv("./comments.csv")

headings["date"] = headings["created_utc"].apply(convertUTCHumanReadable)
comments["date"] = comments["created_utc"].apply(convertUTCHumanReadable)

headings.to_csv("headings.csv", index=False, encoding="utf-8")
comments.to_csv("comments.csv", index=False, encoding="utf-8")


A.mentionsOverTime(start_date="2023-05-01", end_date="2023-08-01")

B=Search()
B.search()

# df = pd.read_csv("./data/headings-with-sentiment.csv")
df2 = pd.read_csv("comments-with-sentiment.csv")
# A.mentionsOverTime()
# A.draw3PieChart(df["Sentiment"])
A.draw5PieChart(df2["Sentiment"])
C=Wordcloud()
# A.wordCloud()