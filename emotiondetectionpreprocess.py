## Preprocessing the tweet dataset for neat text ##
import neattext.functions as nfx
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from textblob import TextBlob
import nltk
from nltk.stem import PorterStemmer
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

df = pd.read_csv("/content/sample_data/tweet_emotions.csv")
df.rename(columns={"sentiment": "emotion"}, inplace=True)

plt.figure(figsize=(12, 8))
sns.countplot(x="emotion", data=df)
plt.show()

df.emotion.replace("empty", "neutral", inplace=True)
df.emotion.replace("anger", "hate", inplace=True)
df.emotion.replace("boredom", "neutral", inplace=True)
df.emotion.value_counts()

df.drop(["tweet_id"], axis=1, inplace=True)

cols_ = list(df.columns)
df = df.sample(frac=1).reset_index()
df = df[cols_]
emotion_list = {"hate", "happiness", "sadness", "neutral"}
new_df = pd.DataFrame(columns=cols_)
for emotion in emotion_list:
  emotion_df = df[df["emotion"] == emotion]
  new_df = pd.concat([new_df, emotion_df.head(1000)])

#Preproessing with different techinques (Stop words, Punctuations, UserHandles, URLs, Numbers, accents, Correcting Text, removing numbered words)
new_df["updated_content"] = new_df.content.apply(lambda x: x.lower())
new_df["updated_content"] = new_df.updated_content.apply(nfx.remove_punctuations)
new_df["updated_content"] = new_df.updated_content.apply(nfx.remove_userhandles)
new_df["updated_content"] = new_df.updated_content.apply(nfx.remove_stopwords)
new_df["updated_content"] = new_df.updated_content.apply(nfx.remove_urls)
new_df["updated_content"] = new_df.updated_content.apply(nfx.remove_numbers)
new_df["updated_content"] = new_df.updated_content.apply(nfx.remove_accents)
new_df["updated_content"] = new_df.updated_content.apply(lambda x: str(TextBlob(x).correct()))

def containsNumber(value):
    for character in value:
        if character.isdigit():
            return True
    return False

def remove_numbered_words(text):
  return ' '.join([token for token in text.split() if not containsNumber(token)])

new_df["updated_content"] = new_df.updated_content.apply(remove_numbered_words)
new_df[["content", "updated_content"]].head()


wordnet_lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
def stem_text(text):
  return ' '.join([porter.stem(token) for token in text.split()]) 
def lemmatize_text(text):
  return ' '.join([wordnet_lemmatizer.lemmatize(token) for token in text.split()])

new_df["updated_content"] = new_df.updated_content.apply(stem_text)
new_df["updated_content"] = new_df.updated_content.apply(lemmatize_text)
new_df[["content", "updated_content"]].head()

new_df.to_csv("/content/drive/MyDrive/Docs/balanced_filtered_tweet_emotions.csv")

