
#ideas: get rid of words that don't appear frequently in naive bayes. try the scikit learn algorithm and see
#check your word vectors, makes sure they make sense. test the model on pretrained vectors
#try random forest on small dataset and trace it
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
    plot_zipf()
    word2vec_explore()
    pickle_in = open("processed_text_list.pickle", "rb")
    processed_text_list = pickle.load(pickle_in)
    #preprocess_text(train_text)
    train_w2v_model(processed_text_list)

    #shuffle and partition dataset
    from sklearn.utils import shuffle
    data = pd.DataFrame({'text': processed_text_list, 'labels': polarity})
    data = shuffle(data)
    get_w2v_array(data[:400000])
    w2v_array = pickle.load(open('w2v_features.pickle', 'rb'))
    num_tweets = 400000 # number of tweets to consider
    w2v_array = w2v_array[:num_tweets]
    split_ratio = int(num_tweets * .8)

    w2v_train = w2v_array[:split_ratio] # w2v averages for each tweet
    w2v_test = w2v_array[split_ratio:]

    data = shuffle(data)
    simple_train = data['text'][:split_ratio] # preprocessed text
    simple_test =  data['text'][split_ratio:]

    labels_list = data['labels'].tolist()[:num_tweets]
    train_labels = labels_list[:split_ratio] # list of labels
    test_labels = labels_list[split_ratio:]


    # get_w2v_array(data=data)
    # pickle_in = open("w2v_features.pickle", "rb")
    # w2v_features = pickle.load(pickle_in)

    # naive_bayes = NaiveBayes(simple_train.tolist(), simple_test.tolist(), labels_list)
    # accuracy = naive_bayes.evaluate()
    # print("Naive Bayes accuracy: " + str(accuracy)) #.499

    # svm = SVM(simple_train, train_labels, simple_test, test_labels, 3000, .0000001)
    # accuracy = svm.predict()
    # print("SVM accuracy: " + str(accuracy)) #.744 with a=.0000001 and 3000 epochs

    random_forest = RandomForest(w2v_train, w2v_test, train_labels, test_labels, 'sqrt', max_depth=25, min_leaf=2, n_trees=500, model_type='scikit')
    accuracy = random_forest.evaluate()
    print("Random Forest accuracy: " + str(accuracy))
    #poor performance: next step is to try with tfidf vectors and only use words that appear between 5 and 500 times in the corpus. Consider bagging and pruning.
def plot_zipf():
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
    plotly.tools.set_credentials_file(username='hadarohana', api_key='DvO469BCrrthG3ZxyI9L')
    w2v_model = Word2Vec.load('w2v_model')
    word_frequency = {}
    for word in w2v_model.wv.vocab:
        word_frequency[word] = w2v_model.wv.vocab[word].count
    word_frequency = dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))
    rank = [x for x in word_frequency.keys()][:250]
    frequency = list(x for x in word_frequency.values())[:250]
    trace = go.Bar(x=rank, y = frequency)
    layout = go.Layout(title="Twitter data for 250 most common words in 1 million tweets", xaxis={'title': 'words ordered by rank'}, yaxis={'title': 'frequency'})
    data = [trace]
    figure = go.Figure(data=data, layout=layout)
    py.iplot(figure, filename='zipf')


def word2vec_explore():
    w2v_model = Word2Vec.load('w2v_model')
    i = 0
    # for word in w2v_model.wv.vocab:
    #     print(str(i) + ' ' + word)
    #     i += 1
    print(len(w2v_model.wv.vocab))
    print(w2v_model.wv.most_similar(positive=['good']))
# get array of words converted into their average w2v vectors
def get_w2v_array(data):
    import math
    w2v_model = Word2Vec.load('w2v_model')
    w2v_features = numpy.zeros(shape=(len(data), 300), dtype='float32')
    for i, tweet in enumerate(data['text']):
        w2v_features[i] = numpy.mean([w2v_model.wv[word] for word in tweet if word in w2v_model.wv.vocab], axis=0)
        if(math.isnan(w2v_features[i][0])):
            w2v_features[i] = numpy.zeros(300)
    print('done making w2v array')
    pickle_out = open("w2v_features.pickle", "wb")
    pickle.dump(w2v_features, pickle_out)

# Cleans, tokenizes, and lemmatizes tweets
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
        soup = BeautifulSoup(text, features="html.parser") # parse html tags into words
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
    w2v_model = Word2Vec(bigrams[processed_text_list], size=300,  min_count=30, window=5, sg=1)
    print("model created")
    w2v_model.train(bigrams[processed_text_list], total_examples= len(bigrams[processed_text_list]), epochs=10)
    print("model training complete")
    w2v_model.save("w2v_model")

def generate_embedding_matrix():
    model = gensim.models.Word2Vec.load("w2v_model")
    embedding_dim = len(model.wv[model.wv.index2word[0]])
    embedding_matrix = numpy.zeros((len(model.wv.vocab), embedding_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

if __name__ == "__main__":
    main()
