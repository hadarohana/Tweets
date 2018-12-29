
import pandas as pd
import numpy
import re
import string
import pickle
import contractions
import gensim
from gensim.models import Word2Vec
import sklearn
import NaiveBayes
from NaiveBayes import NaiveBayes

df = pd.read_csv("train.csv", encoding = 'latin-1' )
text = df.iloc[:, 5] # dataframe of tweets
polarity = df.iloc[:, 0] # dataframe of polarities

def main():
    pickle_in = open("processed_text_list.pickle", "rb")
    processed_text_list = pickle.load(pickle_in)
    #preprocess_text(train_text)

    #train_w2v_model(processed_text_list)
    #w2v_exploration()

    #shuffle and partition dataset
    from sklearn.utils import shuffle
    data = pd.DataFrame({'text': processed_text_list, 'labels': polarity})
    shuffle(data)
    split_ratio = int(len(processed_text_list) * .8)
    train_list = data['text'].loc[0:split_ratio].tolist()
    test_list =  data['text'].loc[split_ratio:len(processed_text_list)].tolist()
    labels_list = data['labels'].tolist()

    naive_bayes = NaiveBayes(train_list, test_list, labels_list)
    accuracy = naive_bayes.evaluate()
    print("Naive Bayes accuracy: " + str(accuracy))


#cleans, tokenizes, and lemmatizes tweets
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

# word2vec model trained on entire sentiment 140 corpus (including neutral text)
def train_w2v_model(processed_text_list):

    from gensim.models import Phrases
    bigrams = Phrases(processed_text_list, min_count = 2) # bigram model
    w2v_model = Word2Vec(bigrams[processed_text_list], size=300,  min_count=3, window=5, sg=1)
    print("model created")
    w2v_model.train(bigrams[processed_text_list], total_examples= len(bigrams[processed_text_list]), epochs=10)
    print("model training complete")
    w2v_model.save("w2v_model")


def w2v_exploration():
   w2v_model = Word2Vec.load("w2v_model")
   print(list(w2v_model.wv.vocab))



#def process_features_and_labels:



if __name__ == "__main__":
    main()
