import pymysql
import settings
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import Binarizer
from sklearn import linear_model
from sklearn import tree

from sshtunnel import SSHTunnelForwarder

import numpy


class TwitterDataset:
    def __init__(self, documents, targets):
        self.documents = numpy.array(documents)
        self.targets = numpy.array(targets)

    def return_targets(self):
        return self.targets

    def return_documents(self):
        return self.documents

# the classifier object stores vectorizer, binarizer, training tweets and target tweets.

class ClassifierObject:
    def __init__(self, vectorizer, training, targets):
        self.training = training
        self.targets = targets
        self.vectorizer = vectorizer
        self.binarizer = Binarizer()

    # here we initialize the classifiers and fit them to the data.
    # 'fitting to the data' is the same as 'training' the classifier.

        b = BernoulliNB()
        r = RandomForestClassifier()

        self.BernoulliClassifier = b.fit(training, targets)
        self.RandomForestClassifier = r.fit(training, targets)

    # we can define different functions to test different algorithms. Here I have
    # the Bernoulli (read: binary) Naive Bayes and the RandomForest Classifier




    def predict_Ber(self, tweet):
        text = tweet['tweettext']
        text = self.vectorizer.transform([text])
        text = self.binarizer.transform(text)
        pred = self.BernoulliClassifier.predict(text)
        return pred




    def predict_RandomForest(self, tweet):
        text = tweet['tweettext']
        text = self.vectorizer.transform([text])
        return self.RandomForestClassifier.predict(text)


# support vector machine / stemming



def main_loop():

    recruiting_ids = []
    nonrecruiting_ids = []

    # each tweet at some index in `tweets` corresponds to the number at the same index
    # in targets: 1 if it is a recruiting tweet, and 0 if not.

    training_tweets = []
    training_targets = []



    # read each line from file, query database for that tweet
    with SSHTunnelForwarder(
            'ucla.seanbeaton.com',
            ssh_port=7822,
            ssh_username="db_readonly_user",
            local_bind_address=settings.local_address,
            remote_bind_address=('127.0.0.1', 3306)) as server:

        cnx = pymysql.connect(**settings.db_config, cursorclass=pymysql.cursors.DictCursor)
        with cnx.cursor() as cur:
            with open(settings.recruiting_ids, 'r') as file:
                for line in file:
                    id = int(line)
                    if id not in recruiting_ids or nonrecruiting_ids:
                        recruiting_ids.append(id)
                        cur.execute("SELECT tweettext FROM twitter_collect WHERE tweetid = %s", (id))
                        tweet = cur.fetchone()
                        print(tweet)
                        training_tweets.append(tweet['tweettext'])

                        training_targets.append(1)

            with open(settings.nonrecruiting_ids, 'r') as file:
                for line in file:
                    id = int(line)
                    if id not in recruiting_ids or nonrecruiting_ids:
                        nonrecruiting_ids.append(id)
                        cur.execute("SELECT tweettext FROM twitter_collect WHERE tweetid = %s", (id))
                        tweet = cur.fetchone()
                        training_tweets.append(tweet['tweettext'])
                        training_targets.append(0)
    print(training_tweets[0])
    # extract unigram and bigram frequency in training data

    # the vectorizer transforms a list of tweets to a distribution of `features`: the appearance of any given word

    vectorizerCount = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1,2))
    vectorizerTFIDF = TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1,2))

    # binarizer = Binarizer().fit_transform(training_tweets)

    tfidfizer = vectorizerTFIDF.fit_transform(training_tweets)

    # classobjCount = ClassifierObject(vectorizerCount, binarizer, training_targets)
    classobjDFIDF = ClassifierObject(vectorizerTFIDF, tfidfizer, training_targets)

    predicted_recruiting = 0
    predicted_nonrecruiting = 0

    sql = "SELECT * FROM twitter_collect ORDER BY tweetid limit 20000"

    with cnx.cursor() as cur:
        cur.execute(sql)
        results = cur.fetchall()
        for result in results:
            text = result['tweettext']
            if classobjDFIDF.predict_RandomForest(result):
                predicted_recruiting += 1
                if result['tweetid'] not in recruiting_ids:
                    print(str(result['tweetid']) + ' ' + text)
                    # try:
                    #     cur.execute("INSERT IGNORE INTO training_collect(tweetid, tweettext, category) VALUES (%s, %s, %s)",
                    #                 (str(result['tweetid']), str(result['tweettext']), str(1)))
                    # except Exception as e:
                    #     print(e)
                    #     exit(1)
                else:
                    predicted_nonrecruiting += 1

    print(str(predicted_recruiting) + " recruitment tweets found")
    print(str(predicted_nonrecruiting) + " other tweets")
    # cnx.commit()
    cnx.close

if __name__ == '__main__':
    main_loop()
