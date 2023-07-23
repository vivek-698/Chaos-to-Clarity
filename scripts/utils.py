from datetime import datetime
import pandas as pd

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
headings = pd.read_csv("./data/headings.csv")
comments = pd.read_csv("./data/comments.csv")

headings["date"] = headings["created_utc"].apply(convertUTCHumanReadable)
comments["date"] = comments["created_utc"].apply(convertUTCHumanReadable)

headings.to_csv("headings.csv", index=False, encoding="utf-8")
comments.to_csv("comments.csv", index=False, encoding="utf-8")
