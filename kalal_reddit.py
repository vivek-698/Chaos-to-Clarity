import csv
import praw

r = praw.Reddit(
    client_id="5nV8Wwj6VxKrQ9o-lo5qkg",
    client_secret="tNIL7JG0IWh2pbC60bFkOPe5evVZAQ",
    user_agent="Doomsday",
    username="test_and_doom",
    password="test@123",
)

submissions = r.subreddit("elonmusk").hot(limit=1000)

attributes = [
    "selftext",
    "saved",
    "title",
    "name",
    "score",
    "likes",
    "view_count",
    "visited",
    "id",
    "author",
    "num_comments",
]
# ['selftext', 'saved',  'title', 'name', 'score','likes', 'view_count','visited', 'id', 'author', 'num_comments']
with open("submissions.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(attributes)
    for i in submissions:
        values = [getattr(i, attr) for attr in attributes]
        writer.writerow(values)
