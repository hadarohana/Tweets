import math
class NaiveBayes:
    train_set = []
    test_set = []
    labels_list = []
    word_counts = {}
    log_priors = {}
    vocab = set()
    def __init__(self, train_list, test_list, labels_list):
        self.train_set = train_list
        self.test_set = test_list
        self.labels_list = labels_list

    # return word count dictionary for each word in text_list. Adds words to global vocab
    def get_word_counts(self, text_list):
        word_counts = {}
        for tweet in text_list:
            for word in tweet:
                word_counts[word] = 1 if (word_counts.get(word) == None) else word_counts.get(word) + 1
                self.vocab.add(word)
        return word_counts

    # initialize log priors and get word counts for each category
    def initialize(self):
        neg_list = []
        pos_list = []
        for i, tweet in enumerate(self.train_set):
            if self.labels_list[i] == 0:
                neg_list.append(tweet)
            elif self.labels_list[i] == 4:
                pos_list.append(tweet)
        self.word_counts['pos'] = self.get_word_counts(pos_list)
        self.word_counts['neg'] = self.get_word_counts(neg_list)
        self.log_priors['pos'] = math.log(float(len(pos_list)) / len(self.train_set))
        self.log_priors['neg'] = math.log(float(len(neg_list)) / len(self.train_set))
        print("initialization complete")

    def predict(self):
        self.initialize()
        result = []
        for tweet in self.test_set:
            pos_score = self.log_priors['pos']
            neg_score = self.log_priors['neg']
            for word in tweet:
                if(word in self.vocab):
                    #log of probability of each word given its category
                    neg_count = self.word_counts['neg'].get(word)
                    pos_count = self.word_counts['pos'].get(word)
                    # if word not in the dictionary
                    if pos_count == None:
                        pos_count = 0
                    if neg_count == None:
                        neg_count = 0
                    log_w_given_pos = math.log((pos_count + 1.0)/ (len(self.word_counts['pos']) + len(self.vocab)))
                    log_w_given_neg = math.log((neg_count + 1.0)/ (len(self.word_counts['neg']) + len(self.vocab)))
                    pos_score += log_w_given_pos
                    neg_score += log_w_given_neg
            if(pos_score > neg_score):
                result.append(4)
            else:
                result.append(0)
        print("prediction complete")
        return result

    def evaluate(self):
        result = self.predict()
        count = 0.0
        for i, label in enumerate(result):
            if(result[i] == self.labels_list[i]):
                count += 1.0
        accuracy = count/float(len(result))
        return accuracy







