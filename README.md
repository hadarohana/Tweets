# Tweets
Sentiment analysis of Twitter data using Sentiment 140 Dataset
## Data Preprocessing
* Cleans tweets 
  * Removes HTML tags and hashtags
  * Expands contractions
  * Decodes utf-8 BOM
* Removes punctuation and stop words
* Uses POS tagging to accurately lemmatize text
## Word2Vec Embeddings
* Creates bigram model and translates words into 300 dimensional feature vectors
* Averages feature vectors for all words in a tweet
## Naive Bayes (Multinomial)
* Prediction based on probability of given word appearing in either category
* Uses laplace smoothing to account for zero probabilities
## SVM
  * Major parameters: alpha (learning rate) and number of iterations 
  * Converts tweets into tf-idf vectors
  * Uses sklearn SGD Classifier with linear kernel which is much faster than SVC
  * Grid search to find best alpha parameter conducted
    * alpha = .0000001 and num_iter = 3000 yields 74 percent accuracy
## Random Forest
  * Uses the word2vec averaged vectors as features
  * Implemented model
    * use model_type = 'my_model'
    * very slow for large set
    * uses gini index for impurity measure
  * sklearn model

