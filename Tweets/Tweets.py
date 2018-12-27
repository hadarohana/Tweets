
import pandas as pd
import numpy
import re
import string
import pickle
import contractions

df = pd.read_csv("train.csv", encoding = 'latin-1' )
train_text = df.iloc[:, 5] # dataframe of tweets for train set
train_polarity = df.iloc[:, 0] # dataframe of polarities for train set

def main():
    preprocess_text(train_text)
    train_w2v_model()

def preprocess_text(text_sample):
    from bs4 import BeautifulSoup
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words("english"))
    processed_text_sample = []
    for text in text_sample:
        processed_text = ""
        soup = BeautifulSoup(text, features="html.parser") #parse html tags into words
        text = soup.getText()
        text = re.sub(r"@[A-Za-z0-9]+", '', text) # get rid of @ notation
        text = re.sub('http?://[A-Za-z0-9./]+ | https?://[A-Za-z0-9./]+', '', text)  # get rid of html tags
        text = contractions.fix(text) # expand contractions
        try: # replace utf-8 BOM
            text = text.decode("utf-8-sig")
            text = text.replace(u"\ufffd", "?")
        except:
            text = text
        # remove punctuation and stop words
        for word in text.split():
            processed_word = ''
            for char in word:
                if(char not in string.punctuation):
                    processed_word += char
            if(processed_word not in stop):
                processed_text += (processed_word.lower() + ' ')
        processed_text_sample.append(processed_text)
    processed_text_sample = pd.DataFrame(processed_text_sample)
    # tokenization and pos tagging
    processed_text_sample = processed_text_sample.apply(lambda x: pos_tag(word_tokenize(x.to_string())), axis=1)

    #lemmatization:
    for i, tweet in enumerate(processed_text_sample):
        for j, tagged_word in enumerate(tweet):
            pos = tagged_word[1]
            converted_pos = 'n'
            #convert nltk pos tags to lemmatizer-recognized tags
            if(pos.startswith('J')):
                converted_pos = 'a'
            elif(pos.startswith('V')):
                converted_pos = 'v'
            elif(pos.startswith('R')):
                converted_pos = 'r'
            #to_list = list(tagged_word)
            #to_list[0] = ((lemmatizer.lemmatize(word = tagged_word[0], pos = converted_pos)))
            processed_text_sample[i][j] = lemmatizer.lemmatize(word = tagged_word[0], pos = converted_pos)
    processed_text_sample = processed_text_sample.apply(lambda  x: x[1:])
    #saves dataframe containing lists of processed words
    pickle_out = open("processed_text_sample.pickle", "wb")
    pickle.dump(processed_text_sample, pickle_out)
    print("done preprocessing")
    print(processed_text_sample)

def train_w2v_model():
    #run once to convert dataframe to list
    pickle_in = open("processed_text_sample.pickle", "rb")
    processed_text_sample = pickle.load(pickle_in)
    preprocessed_text_list = processed_text_sample.tolist();
    pickle_out = open("processed_text_list.pickle", "wb")
    print(preprocessed_text_list)
    pickle.dump(preprocessed_text_list, pickle_out)


    from gensim.models import Phrases
    bigrams = Phrases(preprocessed_text_list, min_count = 2) # bigram model
    from gensim.models import Word2Vec
    w2v_model = Word2Vec(bigrams[preprocessed_text_list], size=300,  min_count=3, window=5, sg=1)
    w2v_model.train()
    pickle_out = open()
    pickle_out = open("w2v_model.pickle", "wb")
    pickle.dump(w2v_model, pickle_out)

    #for tweet in processed_text_sample:
     #   print(tweet)
     #   preprocessed_text_list.append(tweet)
   # pickle_out = open("processed_text_sample.pickle", "wb")
    #pickle.dump(processed_text_list, pickle_out)

#def process_features_and_labels:



if __name__ == "__main__":
    main()