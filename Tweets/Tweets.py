
import pandas as pd
import numpy
import re
import string
import pickle
import contractions
import gensim
from gensim.models import Word2Vec
import sklearn
from NaiveBayes import NaiveBayes
from SVM import SVM
from RandomForest import RandomForest

df = pd.read_csv("train.csv", encoding = 'latin-1' )
text = df.iloc[:, 5] # dataframe of tweets
polarity = df.iloc[:, 0] # dataframe of polarities

def main():
    pickle_in = open("processed_text_list.pickle", "rb")
    processed_text_list = pickle.load(pickle_in)
    #preprocess_text(train_text)
    #train_w2v_model(processed_text_list)

    #shuffle and partition dataset
    from sklearn.utils import shuffle
    data = pd.DataFrame({'text': processed_text_list, 'labels': polarity})
    data = shuffle(data)
    #get_w2v_array(data[:300000])
    w2v_array = pickle.load(open('w2v_features.pickle', 'rb'))
    num_tweets = 40000 # number of tweets to consider
    w2v_array = shuffle(w2v_array)[:num_tweets]
    split_ratio = int(num_tweets * .8)

    w2v_train = w2v_array[:split_ratio] # w2v averages for each tweet
    w2v_test = w2v_array[split_ratio:]

    data = shuffle(data)
    simple_train = data['text'][:split_ratio] # preprocessed text
    simple_test =  data['text'][split_ratio:]

    labels_list = data['labels'].tolist()
    train_labels = labels_list[:split_ratio] # list of labels
    test_labels = labels_list[split_ratio:]

    # get_w2v_array(data=data)
    # pickle_in = open("w2v_features.pickle", "rb")
    # w2v_features = pickle.load(pickle_in)

    # naive_bayes = NaiveBayes(simple_train.tolist(), simple_test.tolist(), labels_list)
    # accuracy = naive_bayes.evaluate()
    # print("Naive Bayes accuracy: " + str(accuracy)) #.499

    # svm = SVM(simple_train, train_labels, simple_test, test_labels)
    # accuracy = svm.predict()
    # print("SVM accuracy: " + str(accuracy)) #.744 with a=.0000001 and 3000 epochs

    random_forest = RandomForest(w2v_train, w2v_test, train_labels, test_labels, 'log2', max_depth=5, min_leaf=2, n_trees=5, model_type='my_model')
    accuracy = random_forest.evaluate()
    print("Random Forest accuracy: " + str(accuracy))

# get array of words converted into their average w2v vectors
def get_w2v_array(data):
    w2v_model = Word2Vec.load('w2v_model')
    w2v_features = numpy.zeros(shape=(len(data), 300), dtype=object)
    for i, tweet in enumerate(data['text']):
        w2v_features[i] = numpy.mean([w2v_model.wv[word] for word in tweet if word in w2v_model.wv.vocab], axis=0)
    print('done making w2v array')
    pickle_out = open("w2v_features.pickle", "wb")
    pickle.dump(w2v_features, pickle_out)

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

if __name__ == "__main__":
    main()
