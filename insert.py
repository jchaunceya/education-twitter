import pymysql
import settings
from optparse import OptionParser
from sshtunnel import SSHTunnelForwarder

op = OptionParser()
op.add_option("--insert",
              action="store_true", dest="do_insert",
              help="Insert all ids into 'tweets' in twitter_learn.")
op.add_option("--sep",
              action="store_true", dest="do_sep",
              help="Extract nontraining tweets from range of tweetids given extracted training tweets.")
(opts, args) = op.parse_args()

def main_loop():
    with SSHTunnelForwarder(
            'ucla.seanbeaton.com',
            ssh_port=7822,
            ssh_username="db_readonly_user",
            local_bind_address=settings.local_address,
            remote_bind_address=('127.0.0.1', 3306)) as server:

        cnx = pymysql.connect(**settings.db_config, cursorclass=pymysql.cursors.DictCursor)

        recruiting_ids = []
        nonrecruiting_ids = []

        to_insert = []
        num_nonrecruiting = 0
        recruiting_in_range = 0
        with cnx.cursor() as cur:
            with open(settings.recruiting_ids, "r") as recruiting_file:
                with open(settings.nonrecruiting_ids, "a") as nonrecruiting_file:
                    cur.execute("USE university_twitter_data")
                    for line in recruiting_file:
                        cur.execute("SELECT * FROM twitter_collect WHERE tweetid = %s", (int(line),))
                        t = cur.fetchone()
                        to_insert.append((int(line), t['tweettext'], t['tweetdate'], 'r'))
                        print((int(line), t['tweettext'], t['tweetdate'], 'r'))
                        recruiting_ids.append(int(line))

                    if opts.do_sep:
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

                        tweets = cur.fetchall()
                        for tweet in tweets:
                            id = int(tweet['tweetid'])
                            if (id not in recruiting_ids):
                                nonrecruiting_file.write('\n' + str(id))
                                nonrecruiting_ids.append(id)
                                print(str(id) + " " + tweet['tweettext'])
                                num_nonrecruiting += 1
                            else:
                                recruiting_in_range += 1

            if opts.do_insert:
                with open(settings.nonrecruiting_ids, 'r') as file:
                    for line in file:
                        id = int(line)
                        cur.execute("SELECT * FROM twitter_collect WHERE tweetid = %s", (id,))
                        tweet = cur.fetchone()
                        to_insert.append((id, tweet['tweettext'], tweet['tweetdate'], 'n'))
                        print((id, tweet['tweettext'], tweet['tweetdate'], 'n'))

                cur.execute("USE twitter_learning")
                for info in to_insert:
                    print(info)
                    cur.execute("INSERT IGNORE INTO tweets (tweetid, tweettext, time_posted, category) VALUES (%s, %s, %s, %s)", info)

        cnx.commit()
        cnx.close()

if __name__ == "__main__":
    main_loop()
