#12/23: get the data and preprocess it.
#use gzip to get the data. Examine and decide what you want to research.
# write a replace method to get rid of punctuation, short words, @user handle
# Consider using word2vec to get rid of words without sentiment. Tokenize and lemmatize
#  Download any libraries needed
#12/24:
import pandas as pd
import numpy
import re

df = pd.read_csv("train.csv", encoding = 'latin-1' )
print("here")
train_text = df.iloc[0:10, 5] # dataframe of tweets for train set
train_polarity = df.iloc[0:10, 0] # dataframe of polarities for train set

def main():
    preprocess_text(train_text)


def preprocess_text(text_sample):
    from bs4 import BeautifulSoup
    for text in text_sample:
        print(text)
        soup = BeautifulSoup(text, features="html.parser") #parse html tags into words
        text = soup.getText()
        text = re.sub("@\w", "", text) # get rid of @ notation
        text = re.sub('https?://[A-Za-z0-9./]+', '', text) # get rid of html tags
        text = text.decode("utf-8-sig")
        text = text.replace(u"\ufffd", "?")
        print(text)
    print(text_sample)
if __name__ == "__main__":
    main()
