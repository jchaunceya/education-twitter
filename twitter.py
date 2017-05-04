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
from sklearn import linear_model
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sshtunnel import SSHTunnelForwarder

import numpy

import nltk
from nltk import SnowballStemmer

from nltk.tag import StanfordNERTagger
import re

from optparse import OptionParser

op = OptionParser()
op.add_option("--range",
              action="store_true", dest="do_range",
              help="Insert all ids into 'tweets' in twitter_learn.")

(opts, args) = op.parse_args()

stemmer = SnowballStemmer('english', ignore_stopwords=True)

jar = "downloads/stanford-ner/stanford-ner.jar"
st = StanfordNERTagger('downloads/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', jar)


def stem(text):
    global stemmer
    words = nltk.word_tokenize(text)
    new = []
    for word in words:
        new.append(stemmer.stem(word))
    return " ".join(new)

def clean(text):

    text = re.sub("@|!|…", " ", text)
    text = re.sub("(&amp;)", " and ", text)
    text = re.sub("(https?://t\.co/\w+)", " ", text)
    text = re.sub("(#\w+)", " ", text)
    text = re.sub("(\s+)|(\n+)", " ", text)
    print(text)
    return text


class TwitterDataset:
    def __init__(self, documents, targets):
        self.documents = numpy.array(documents)
        self.targets = numpy.array(targets)

    def return_targets(self):
        return self.targets

    def return_documents(self):
        return self.documents

# the classifier object stores vectorizer, binarizer, training tweets and target tweets.

def main_loop():

    global st

    recruiting_ids = []
    nonrecruiting_ids = []

    # each tweet at some index in `tweets` corresponds to the number at the same index
    # in targets: 1 if it is a recruiting tweet, and 0 if not.

    training_tweets = []
    training_targets = []

    test_tweets = []
    test_targets = []
    proportion = .99

    training = []
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
            percent = proportion * len(training)
            counter = 0
            tweetids = []
            for tweet in training:
                tweetids.append(int(tweet['tweetid']))
                if (counter < percent):
                    training_tweets.append(clean(tweet['tweettext']))

                    counter += 1
                    if tweet['category'] == 'r' or tweet['category'] == 's':
                        training_targets.append(1)
                    else:
                        training_targets.append(0)
                else:
                    test_tweets.append(clean(tweet['tweettext']))

                    if tweet['category'] == 'r' or tweet['category'] == 's':
                        test_targets.append(1)
                    else:
                        test_targets.append(0)



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

            svm.fit(training_tweets, training_targets)
            rf.fit(training_tweets, training_targets)
            ber.fit(training_tweets, training_targets)


            predictionsSVM = svm.predict(test_tweets)
            predictionsRF = rf.predict(test_tweets)
            predictionsBer = ber.predict(test_tweets)

            csvm = confusion_matrix(test_targets, predictionsSVM)
            print(csvm)

            crf = confusion_matrix(test_targets, predictionsRF)
            print(crf)

            cber = confusion_matrix(test_targets, predictionsBer)
            print(cber)


            correct_r = 0  # correctly categorized as recruiting
            ignorable = 0 # false positive (ignore for now)

            bad = 0 # BAD
            correct_n = 0 # correctly categorized as nonrecruiting


            badlist = []
            goodlist = []
            ignorelist = []
            fplist = []

            for i in range(0, len(test_tweets)):

                # Threshold of 2/3:

                if predictionsRF[i] + predictionsSVM[i] + predictionsBer[i] >= 1:
                    pred = 1
                else:
                    pred = 0

                if pred == test_targets[i]:
                    if pred == 0:
                        correct_n += 1
                    if pred == 1:
                        correct_r += 1

                else:
                    if test_targets[i]:
                        bad += 1
                    else:
                        ignorable += 1


            results = []
            results.append(str((correct_r + correct_n)/len(test_tweets)))
            results.append(str((ignorable)/len(test_tweets)))
            results.append(str(bad/len(test_tweets)))
            results.append(str(proportion))
            results.append(str(sum(training_targets)/len(training)))

            with open("results.csv", "a") as csvfile:
                csvfile.write(",".join(results) + '\n')
            # print("Correct recruiting =\t" + str((correct_r + correct_n)/len(test_tweets)))
            # print("Correct nonrecruiting =\t" + str(correct_n))
            # print("False positives =\t" + str(correct_r))
            # print("Bad =\t\t\t" + str(bad))
            # print(correct_r * 100/(bad+correct_r))

            sql = "USE university_twitter_data"

            cur.execute(sql)
            if opts.do_range:
                try:
                    begin_id = int(input("Input beginning id: ")) - 1
                    end_id = int(input("Input ending id: ")) + 1

                except:
                    print("invalid id")
                    exit(1)

                if begin_id > end_id:
                    temp = begin_id
                    begin_id = end_id
                    end_id = temp

                try:
                    cur.execute("SELECT * FROM twitter_collect WHERE tweetid > %s and tweetid < %s", (begin_id, end_id))
                except Exception as e:
                    print(e)
                    exit(1)
            else:
                num = int(input("Input number of tweets:")) - 1
                try:
                    cur.execute("SELECT * FROM twitter_collect ORDER BY tweetid DESC LIMIT %s", (num,))
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

            tagged_tweets = []
            tagged_tweet_ids = []


            for i in range(0, len(prediction_tweets)):
                if prediction_ids[i] not in tweetids:
                    if predictionsRF[i] or predictionsSVM[i]:
                        tagged_tweet_ids.append(prediction_ids[i])


                        t = st.tag(nltk.word_tokenize(prediction_tweets[i]))
                        tagged_tweets.append(t)

                        print(str(prediction_ids[i]) + '\t\t' + prediction_tweets[i])
                        print(t)
            nes = []
            currentchunk = []
            currentcat = ""
            last = ""

            for t in tagged_tweets:
                currentcat = t[0][1]
                for word, cat in t:
                    if cat != 'O' and cat == currentcat:
                        currentchunk.append(word)
                    else:
                        if currentchunk:
                            nes.append(" ".join(currentchunk))
                            currentchunk = []
                            currentcat = cat
                        if (cat != 0):
                            currentchunk.append(word)
                print(nes[len(nes)-1])
            for ne in nes:
                print(ne)

        cnx.close()

if __name__ == '__main__':
    main_loop()
