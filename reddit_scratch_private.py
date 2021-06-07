"""
ADVANCED DATA ANALYSIS 2021

EVENT BASED TRADING ALGORITHM: A REDDIT CASE

Authors: Sebastien Gorgoni & Liam Svoboba

File Name: reddit_scratch_private.py

This is an external file for main.py that collect the post from reddit.

Source: https://github.com/throwaway0101/reddot/blob/master/reddot.py
"""

# Import the libraries
import praw
from psaw import PushshiftAPI
import datetime as dt
import pandas as pd
from tqdm import tqdm

# Set the personal application
reddit = praw.Reddit(client_id = 'enter client id',
                     client_secret = 'enter client secret',
                     username = 'enter username',
                     password = 'enter password',
                     user_agent = 'enter any name')

api = PushshiftAPI(reddit)

def top(start_date, number_top, sub_reddit):
    end_date = start_date + dt.timedelta(days=1)
    top_list = list(
        api.search_submissions(
            after=int(start_date.timestamp()),
            before=int(end_date.timestamp()),
            subreddit = sub_reddit, 
            filter=["url", "author", "title", "subreddit"],
            sort_type="num_comments",
            sort="desc",
            limit=40, #You add more in the limit than in the top_list bracket in case some post are removed or deleted.
        )
    )
    top_list = [
        post for post in top_list if post.selftext not in ["[removed]", "[deleted]"]
    ]
    return top_list[:number_top]

def output(submissions):
    posts = []
    for post in tqdm(submissions):
        posts.append(post.title)
        #print("* {}: [{}](<{}>)".format(post.score, post.title, post.url))
    return posts

def download_reddit(index, nbtop, sub_reddit_theme, name):
    """
    This function will download all post from a subreddit.

    Parameters
    ----------
    index : datetime
        Determine the time frame in which you want to collect the reddit posts.
    nbtop : int
        Determine the number of post you want to collect each days.
    sub_reddit_theme : string
        Determine which subreddit to collect the post.
    name : string
        Name of CSV file to save.

    Returns
    -------
    df : Dataframe
        It returns a dataframe with each top daily posts.

    """

    df = pd.DataFrame()
    
    for i in index:
        print("## {} {} {}".format(i.year, i.month, i.day))
        data = output(top(i, number_top=nbtop, sub_reddit=sub_reddit_theme))
        df_temp =  pd.DataFrame([data])
        df = df.append(df_temp)
        print("\n___\n")
        
    df['Date'] = index
    
    df.set_index('Date', inplace=True)

    for i in range(1, df.shape[1]+1):
        df.rename(columns={i-1: f'Top{i}'}, inplace=True)
        
    df.to_csv(name, index=True)
    
    df = df.dropna(how='any')
    
    return df
