import csv
import praw

r = praw.Reddit(
    client_id="5nV8Wwj6VxKrQ9o-lo5qkg",
    client_secret="tNIL7JG0IWh2pbC60bFkOPe5evVZAQ",
    user_agent="Doomsday",
    username="test_and_doom",
    password="test@123",
)

# fetches the top 1000 posts from the subreddit
top1k_posts = r.subreddit("elonmusk").hot(limit=1000)

attributes = [
    "title",
    "name",
    "score",
    "likes",
    "view_count",
    "visited",
    "id",
    "author",
    "ups",
    "downs",
    "created_utc",
]

comment_attributes = ["body", "ups", "downs", "created_utc"]
# ['selftext', 'saved',  'title', 'name', 'score','likes', 'view_count','visited', 'id', 'author', 'num_comments']
with open("headings.csv", "w", newline="", encoding="utf-8") as headings_file, open(
    "comments.csv", "w", newline="", encoding="utf-8"
) as comments_file:
    writer1 = csv.writer(headings_file)
    writer1.writerow(attributes)

    writer2 = csv.writer(comments_file)
    writer2.writerow(comment_attributes)
    for post in top1k_posts:
        values = [getattr(post, attr) for attr in attributes]
        writer1.writerow(values)

        post.comments.replace_more(limit=50)
        print(
            "Fetching ",
            min(50, len(post.comments.list())),
            " comments for post titled",
            post.title,
        )
        for comment in post.comments.list():
            values = [getattr(comment, attr) for attr in comment_attributes]
            writer2.writerow(values)
