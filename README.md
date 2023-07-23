## File structure

## Workflow
To run this project, type in the terminal `python3 flask-app.py`. 
What this program does first is, it initializes all the things necessary to draw the graphs. So it imports from `scripts/generate_analytics.py`. It creates an object of `Analysis` class and then calls the respective methods based on what information is requested by the client.

There are comments over every line explaining its purpose.

## Goals
1. Run social media analysis on a celebrity of your choice. (Demo: Elon Musk)
2. View the number of "mentions" over time
3. View sentiment over time (Define a window say a day/week and average out the sentiment in each window and display in a graph)
4. View the social media posts (tweets, reddit articles) that contributed most to the sentiment. (Sort by sentiment score, break the tie by most number of upvotes and display the post, embed the link into the website.)
5. General overview of some analytics
6. Which posts are contributing the most to our analysis. (use upvotes of post and number of comments. show +ve/-ve outcome)

Analytics being offered:
* Sentiment distribution pie chart (positive/negative/neutral)
* Trending sentiment (window size = daily,weekly,monthly)
* Top Positive/Negative posts
* Word cloud
* Topic modeling (View the topics)
* Hater analysis (Are there any specific users that hold a grudge and are driving your sentiment scores down dramatically)
* Sentiment change after events 

User will also be given an option to filter out the analyitcal data based on the social media (Reddit, Instagram, Youtube, Twiiter, Facebook etc.)

## Working
kalal_reddit.py contains the reddit api call, it sends the output to 2 csv files located in the data directory (comments,headings) 
huggingface-offline.py was used to generate sentiment for headings. "headings-with-sentiment.csv"

## Sample Screenshots (23/07/2023)
![image](https://github.com/vivek-698/Chaos-to-Clarity/assets/77607172/9a83e55f-d0b1-46e2-874a-fed15832c784)
![image](https://github.com/vivek-698/Chaos-to-Clarity/assets/77607172/a85dfec3-1562-4edc-8fba-da5bd99ff9f0)
![image](https://github.com/vivek-698/Chaos-to-Clarity/assets/77607172/1260c680-0599-4d19-bc90-6d933fd590f3)


## Future Work
* Factor in "upvotes" into the sentiment score
