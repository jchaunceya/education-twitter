#!/usr/local/bin/python3
# This Python file uses the following encoding: utf-8
import pymysql
import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Binarizer, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sshtunnel import SSHTunnelForwarder
from time import time

# import nltk
# from nltk import SnowballStemmer
# from nltk.tag import StanfordNERTagger

# import ner

import re

from optparse import OptionParser

op = OptionParser()

op.add_option("--b", "--begin",
              action="store", dest="begin", type="int",
              help="Begins at first argument, classifies tweets in `twitter_collect` until it has collected n = second arg recruiting tweets returns ending id",)

op.add_option("--n", "--num",
              action="store", dest="num", type="int", default=25,
              help="")

op.add_option("--r", "--range",
              action="store", dest="range", type="int", nargs=2,
              help="Classify all tweets between the given tweetids")

op.add_option('--c', action="store_true", dest='do_classify', default=False,
                    help='Classify tweets from twitter_collect. If not specified, will test on subset of labeled training tweets from `twitter_learning.tweets`')

(opts, args) = op.parse_args()

# stemmer = SnowballStemmer('english', ignore_stopwords=True)
#
# jar = "downloads/stanford-ner/stanford-ner.jar"
# st = StanfordNERTagger('downloads/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', jar)
#
# def stem(text):
#     global stemmer
#     words = nltk.word_tokenize(text)
#     new = []
#     for word in words:
#         new.append(stemmer.stem(word))
#     return " ".join(new)

def clean(text, low):

    text = re.sub("(^rt )|(^RT )|\,|\\?|(\\.\\.\\.)|\\)|\(|@|!|…", " ", text)
    text = re.sub("(&amp;)", " and ", text)
    text = re.sub("(https?://t\.co/\w+)", " ", text)
    text = re.sub("(#\w+)", " ", text)
    text = re.sub("(\s+)|(\n+)", " ", text)
    if low:
        return str.lower(text)
    return text

def return_tweets(tar=False,low=False):
    tweetvec = []
    tweets = []
    targets = []
    rec = []
    nonrec = []
    with SSHTunnelForwarder(
            'ucla.seanbeaton.com',
            ssh_port=7822,
            ssh_username="db_readonly_user",
            local_bind_address=settings.local_address,
            remote_bind_address=('127.0.0.1', 3306)) as server:

        cnx = pymysql.connect(cursorclass=pymysql.cursors.DictCursor,**settings.db_config)
        with cnx.cursor() as cur:

            sql = "SELECT * FROM tweets ORDER BY RAND()"
            cur.execute(sql)
            tweets = cur.fetchall()

            tweetids = []

            for tweet in tweets:
                tweetids.append(int(tweet['tweetid']))
                tweetvec.append(clean(tweet['tweettext'], low))
                if tweet['category'] == 'r' or tweet['category'] == 's':
                    targets.append(1)
                    rec.append(clean(tweet['tweettext'], low))
                else:
                    targets.append(0)
                    nonrec.append(clean(tweet['tweettext'], low))

        cnx.close()
    if tar:
        return (tweetvec,targets)
    else:
        return (rec, nonrec)

def main_loop():

    # each tweet at some index in `tweets` corresponds to the number at the same index
    # in targets: 1 if it is a recruiting tweet, and 0 if not.

    (alltweets,alltargets) = return_tweets(tar=True)

    # read each line from file, query database for that tweet
    with SSHTunnelForwarder(
            'ucla.seanbeaton.com',
            ssh_port=7822,
            ssh_username="db_readonly_user",
            local_bind_address=settings.local_address,
            remote_bind_address=('127.0.0.1', 3306)) as server:

        cnx = pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **settings.db_config)

        with cnx.cursor() as cur:

            svm = Pipeline([
                ("svmvec", TfidfVectorizer(stop_words='english', ngram_range=(2, 3), )), #max_features=1000
                ("binarize1", Binarizer()),
                ('normalizer1', Normalizer()),
                ("clf_svc", SVC(C=10000, gamma=1e-3, kernel='rbf', class_weight={1:100}))
            ])

            rf = Pipeline([
                ("rfvec", TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(2, 3))),
                ("binarizer2", Binarizer()),
                ("normalize2", Normalizer()),
                ("clf_rf", RandomForestClassifier())
            ])

            ber = Pipeline([
                ("bervec", TfidfVectorizer(stop_words='english', ngram_range=(2, 3))),
                ('binarizer3', Binarizer()),
                ('normalizer3', Normalizer()),
                ("clf_ber", BernoulliNB())
            ])

            tree = Pipeline([
                ("treevec", TfidfVectorizer(stop_words='english', ngram_range=(2, 3))),
                ('binarizer4', Binarizer()),
                ('normalizer4', Normalizer()),
                ("clf_ber", DecisionTreeClassifier())
            ])

            neural = Pipeline([
                ("neuralvec", TfidfVectorizer(stop_words='english', ngram_range=(2, 3))),
                ('binarizer5', Binarizer()),
                ('normalizer5', Normalizer()),
                ("clf_neural", MLPClassifier())
            ])

            if not opts.do_classify:

                proportion = int(0.9 * len(alltweets))

                svm.fit(alltweets[:proportion], alltargets[:proportion])
                rf.fit(alltweets[:proportion], alltargets[:proportion])
                ber.fit(alltweets[:proportion], alltargets[:proportion])
                tree.fit(alltweets[:proportion], alltargets[:proportion])
                neural.fit(alltweets[:proportion], alltargets[:proportion])

                predictionsSVM = svm.predict(alltweets[proportion:])
                predictionsRF = rf.predict(alltweets[proportion:])
                predictionsBer = ber.predict(alltweets[proportion:])
                predictionsTree = tree.predict(alltweets[proportion:])
                predictionsNeural = neural.predict(alltweets[proportion:])


                csvm = confusion_matrix(alltargets[proportion:], predictionsSVM)
                print("Support vector machine:")
                print(csvm)

                crf = confusion_matrix(alltargets[proportion:], predictionsRF)
                print("Random forest classifier:")
                print(crf)

                cber = confusion_matrix(alltargets[proportion:], predictionsBer)
                print("Bernoulli Naive Bayes:")
                print(cber)

                ctree = confusion_matrix(alltargets[proportion:], predictionsTree)
                print("Decision tree:")
                print(ctree)

                cneural = confusion_matrix(alltargets[proportion:], predictionsNeural)
                print("Neural network:")
                print(cneural)

                exit(0)

            svm.fit(alltweets, alltargets)
            rf.fit(alltweets, alltargets)
            ber.fit(alltweets, alltargets)
            tree.fit(alltweets, alltargets)
            neural.fit(alltweets, alltargets)

            inc = 54852739182592                            # `inc` is totally arbitrary

            if opts.begin:
                begin_id = opts.begin
                end_id = begin_id + inc

            elif len(opts.range) == 1:
                print("Input two integers as tweetids!")
                exit(1)

            elif len(opts.range) == 2:
                begin_id = opts.range[0]
                end_id = opts.range[1]

            tweet_ct = 0

            sql = "USE university_twitter_data;"
            cur.execute(sql)

            last_id = 0

            while(opts.num > tweet_ct):
                try:
                    cur.execute("SELECT * FROM twitter_collect WHERE tweetid > %s and tweetid < %s", (begin_id, end_id))
                except Exception as e:
                    print(e)
                    exit(1)

                results = cur.fetchall()
                prediction_tweets = []
                prediction_ids = []

                if results:
                    for result in results:
                        prediction_tweets.append(clean(result['tweettext'], low=False))
                        prediction_ids.append(int(result['tweetid']))

                    predictionsSVM = svm.predict(prediction_tweets)
                    predictionsDT = tree.predict(prediction_tweets)
                    predictionsNeural= neural.predict(prediction_tweets)
                    predictionsRF = rf.predict(prediction_tweets)

                    for i in range(0, len(prediction_tweets)):
                        last_id = prediction_ids[i]
                        if predictionsSVM[i] + predictionsDT[i] + predictionsNeural[i] + predictionsRF[i] >= 2:
                            tweet_ct += 1
                            print(str(prediction_ids[i]) + ' ' + prediction_tweets[i])

                        if (opts.num <= tweet_ct and opts.range == None):
                            print("Last tweet classified = " + str(last_id))
                            return last_id

                end_id = begin_id + 2 * inc
                begin_id = begin_id + inc
                if opts.range != None:
                    break

        cnx.close()

if __name__ == '__main__':
    main_loop()