import pymysql
import settings
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sshtunnel import SSHTunnelForwarder

import nltk
# from nltk import SnowballStemmer
#
# from nltk.tag import StanfordNERTagger
import re

from optparse import OptionParser

import ner

from argparse import ArgumentParser

parser = ArgumentParser(description='The range of tweetids to classify.')

parser.add_argument('--c', action="store_true", dest='do_classify', default=False,
                    help='Classify tweets from database. If not specified, will test on tweets from `twitter_learning.tweets`')

parser.add_argument('integers', metavar='N', type=int, nargs='*',
                    help='give the beginning id and ending id.')

parser.add_argument('trainingprop', metavar='N', type=float, nargs='?', default=0.9,
                    help='The proportion of training documents to test documents. If not specified, will be 0.9.')

args1 = parser.parse_args()

op = OptionParser()

op.add_option("--range",
              action="store_true", dest="do_range", default=False,
              help="Insert all ids into 'tweets' in twitter_learn.")

(opts, args2) = op.parse_args()

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

def clean(text):

    text = re.sub("@|!|…", " ", text)
    text = re.sub("(&amp;)", " and ", text)
    text = re.sub("(https?://t\.co/\w+)", " ", text)
    text = re.sub("(#\w+)", " ", text)
    text = re.sub("(\s+)|(\n+)", " ", text)
    return text

def main_loop():

    if args1.do_classify and len(args1.integers) < 2:
        print('You need to give two tweetids to classify a range!')
        exit(1)

    # global st

    # each tweet at some index in `tweets` corresponds to the number at the same index
    # in targets: 1 if it is a recruiting tweet, and 0 if not.


    alltweets = []
    alltargets = []

    # read each line from file, query database for that tweet
    with SSHTunnelForwarder(
            'ucla.seanbeaton.com',
            ssh_port=7822,
            ssh_username="db_readonly_user",
            local_bind_address=settings.local_address,
            remote_bind_address=('127.0.0.1', 3306)) as server:

        cnx = pymysql.connect(**settings.db_config, cursorclass=pymysql.cursors.DictCursor)
        with cnx.cursor() as cur:

            sql = "SELECT * FROM tweets ORDER BY RAND()"
            cur.execute(sql)
            training = cur.fetchall()

            counter = 0
            tweetids = []
            for tweet in training:
                tweetids.append(int(tweet['tweetid']))
                alltweets.append(clean(tweet['tweettext']))
                if tweet['category'] == 'r' or tweet['category'] == 's':
                    alltargets.append(1)
                else:
                    alltargets.append(0)

            svm = Pipeline([
                ("svmvec", TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 3))),
                ("binarize1", Binarizer()),
                ("clf_svc", LinearSVC())
            ])

            rf = Pipeline([
                ("rfvec", TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 3))),
                ("binarize2", Binarizer()),
                ("clf_rf", RandomForestClassifier())
            ])

            ber = Pipeline([
                ("bervec", TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 3))),
                ("binarize3", Binarizer()),
                ("clf_ber", BernoulliNB())
            ])

            if not args1.do_classify:

                proportion = int(args1.trainingprop * len(alltweets))

                svm.fit(alltweets[:proportion], alltargets[:proportion])
                rf.fit(alltweets[:proportion], alltargets[:proportion])
                ber.fit(alltweets[:proportion], alltargets[:proportion])


                predictionsSVM = svm.predict(alltweets[proportion:])
                predictionsRF = rf.predict(alltweets[proportion:])
                predictionsBer = ber.predict(alltweets[proportion:])

                csvm = confusion_matrix(alltargets[proportion:], predictionsSVM)
                print("Linear support vector classifier:")
                print(csvm)

                crf = confusion_matrix(alltargets[proportion:], predictionsRF)
                print("Random forest classifier:")
                print(crf)

                cber = confusion_matrix(alltargets[proportion:], predictionsBer)
                print("Bernoulli Naive Bayes:")
                print(cber)

                exit(0)

            svm.fit(alltweets, alltargets)
            rf.fit(alltweets, alltargets)
            ber.fit(alltweets, alltargets)

            sql = "USE university_twitter_data;"
            cur.execute(sql)

            begin_id = args1.integers[0]
            end_id = args1.integers[1]


            if begin_id > end_id:
                temp = begin_id
                begin_id = end_id
                end_id = temp

            try:
                cur.execute("SELECT * FROM twitter_collect WHERE tweetid > %s and tweetid < %s", (begin_id, end_id))
            except Exception as e:
                print(e)
                exit(1)

            results = cur.fetchall()
            prediction_tweets = []
            prediction_ids = []

            for result in results:
                prediction_tweets.append(clean(result['tweettext']))
                prediction_ids.append(int(result['tweetid']))

            predictionsSVM = svm.predict(prediction_tweets)
            predictionsRF = rf.predict(prediction_tweets)
            predictionsBer = ber.predict(prediction_tweets)

            for i in range(0, len(prediction_tweets)):
                if prediction_ids[i] not in tweetids:
                    if predictionsSVM[i]:
                        print(str(prediction_ids[i]) + ' ' + prediction_tweets[i])

        cnx.close()

if __name__ == '__main__':
    main_loop()
