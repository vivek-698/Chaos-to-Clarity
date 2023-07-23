from flask import Flask, render_template
import pandas as pd
from scripts.generate_analytics import Analytics

# from generate_analytics import Analytics

A = Analytics(name="Elon Musk") # name is just for reference

app = Flask(__name__)
@app.route('/')
def home():
    # Here you can provide the necessary data for the home page
    return render_template('index.html')

@app.route('/about')
def about():
    # Here you can provide the necessary data for the about page
    return render_template('about.html')

@app.route('/contact')
def contact():
    # Here you can provide the necessary data for the contact page
    return render_template('contact.html')

@app.route('/mention-count')
def mentionCount():
    A.mentionsOverTime()
    return render_template('mention_count.html', graph_path="static/mentions-over-time.png")

@app.route('/sentiment')
def sentiment():
    df = pd.read_csv("data/headings-with-sentiment.csv")
    A.draw3PieChart(df["Sentiment"])
    return render_template('sentiment.html',graph_path="static/3piechart.png")


@app.route('/emotion')
def emotion():
    df = pd.read_csv("data/comments-with-sentiment.csv")
    A.draw5PieChart(df["Sentiment"])
    return render_template('emotion.html',  graph_path="static/5piechart.png")

@app.route('/wordcloud')
def word_cloud():
    return render_template('word_cloud.html')



if __name__ == '__main__':
    app.run()
