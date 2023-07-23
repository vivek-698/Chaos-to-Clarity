## Goals
1. Run social media analysis on a celebrity of your choice. (Demo: Elon Musk)
2. View the number of "mentions" over time
3. View sentiment over time (Define a window say 5 days and average out the sentiment in each window and display in a graph)
4. View the social media posts (tweets, reddit articles) that contributed most to the sentiment.
5. General overview of some analytics

Analytics being offered:
* Sentiment distribution pie chart (positive/negative/neutral)
* Trending sentiment (window size = daily,weekly,monthly)
* Top Positive/Negative posts
* Word cloud
* Topic modeling (View the topics)
* Hater analysis (Are there any specific users that hold a grudge and are driving your sentiment scores down dramatically)
* Sentiment change after events 

User will also be given an option to filter out the analyitcal data based on the social media (Reddit, Instagram, Youtube, Twiiter, Facebook etc.)

### Working

kalal_reddit.py contains the reddit api call, it sends the output to 2 csv files located in the data directory (comments,headings)
