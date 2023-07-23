from datetime import datetime


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


print(convertUTCHumanReadable(1689878167))
