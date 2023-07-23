import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import convertUTCHumanReadable


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

        # Convert the 'date' column to a pandas datetime object
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
        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the data as a line graph
        ax.plot(posts_per_time['date_agg'], posts_per_time['num_posts'], color='#ff6314')
        # Format the X-axis date labels
        if weekly:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        else:
            plt.xticks(rotation=45)

        # Set labels and title
        plt.xlabel(f'{time_label}')
        plt.ylabel('Number of Posts')
        plt.title('Number of Mentions Over Time')

        # Show the plot
        plt.tight_layout()
        plt.show()



A = Analytics("Elon Musk")
A.mentionsOverTime(start_date="2023-05-01", end_date="2023-08-01") 