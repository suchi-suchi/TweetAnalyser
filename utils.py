## Utility Methods used to extract useful information for statistical analysis ##

from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd

extract = URLExtract()


def monthly_timeline(selected_user,df):

    if selected_user != 'Group':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Group':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def week_activity_map(selected_user,df):

    if selected_user != 'Group':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Group':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()



def get_most_words_by_user(df):
    user_list = list(df["user"].unique())
    return 0

def get_messages_by_user(df):
    user_list = list(df["user"].unique())
    d = dict()
    for i in user_list:
        d[i] = len(df[df["user"] == i])
    return d

def return_stats(df,selected_user):
    if selected_user != 'Group':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    return num_messages,len(words),num_media_messages,len(links)


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df
