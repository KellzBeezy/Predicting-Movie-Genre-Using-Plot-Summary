import json
import pymysql
import requests
import errno
from datetime import datetime


def convert_str(res):
    print(res)
    res = res.replace('-', ' ')
    return res


def strip_tuple(obj):
    result = ''.join(map(str, obj))
    result = result.strip('()')
    result = result.strip(',')
    print(result)
    return result


def get_movieid():
    conn = pymysql.connect(
        "localhost",
        "root",
        "1234",
        "plots"
    )
    mycursor = conn.cursor()

    movie = []

    mycursor.execute('select tmdb_Id from movie_genre where tmdb_Id>70590')
    name = mycursor.fetchall()
    print('the results found are: %s' % len(name))

    for t in name:
        movie_id = int(strip_tuple(t))
        get_cast(movie_id)

    print(name)


def get_cast(mid):
    # num = 4108
    try:
        response = requests.get(
                                'https://api.themoviedb.org/3/movie/%d/credits?api_key=bc96a52a6cb35498e2fe6f3e6dffeaec'
                                % mid)
        data = json.loads(response.text)
        print(data['cast'])
        castdb(data, int(mid))
    except:
        data = []
        castdb(data, int(mid))


def moviedb():
    conn = pymysql.connect(
        "localhost",
        "root",
        "1234",
        "plots"
    )
    mycursor = conn.cursor()

    with open('MOVE12345.json', 'rb') as f:
        data = json.load(f)
        # print(data["movies"])

    genre1 = genre2 = genre3 = genre4 = ""

    for number in range(1, 5):

        for t in data['movies']:

            if len(t['genre_ids']) == number:

                count = 0

                for r in t['genre_ids']:

                    if count == 0:
                        mycursor.execute("select  Genre_Name from genre where Imdb_Genre_Id=%s" % r)
                        name = mycursor.fetchone()
                        genre1 = name
                        # print(name)
                        count = count + 1
                    elif count == 1:
                        mycursor.execute("select  Genre_Name from genre where Imdb_Genre_Id=%s" % r)
                        name = mycursor.fetchone()
                        genre2 = name
                        # print(name)
                        count = count + 1
                    elif count == 2:
                        mycursor.execute("select  Genre_Name from genre where Imdb_Genre_Id=%s" % r)
                        name = mycursor.fetchone()
                        genre3 = name
                        # print(name)
                        count = count + 1
                    elif count == 3:
                        mycursor.execute("select  Genre_Name from genre where Imdb_Genre_Id=%s" % r)
                        name = mycursor.fetchone()
                        genre4 = name
                        # print(name)
                        count = count + 1

                print(t['id'], "  ", t['vote_count'], "   ", t['title'], "  ", genre4, genre3, genre2, genre1,
                      t['overview'], "  ", t['release_date'])
                try:
                    sql = "INSERT INTO movie_genre(Imdb_Id,Vote_count,Title,Genre_ids0,Overview,Release_date) VALUES(%s,%s,%s,%s,%s,%s)"
                    val = (t['id'], t['vote_count'], t['title'], genre1, t['overview'], t['release_date'])
                    mycursor.execute(sql, val)
                    conn.commit()
                    print(t['title'] + " has been commited")
                except:
                    conn.rollback()
                    print(t['title'] + " has been rolled back")
                    # print("Error: {}".format(e))
                    # return {"message": e.message}


def tagsdb():
    conn = pymysql.connect(
        "localhost",
        "root",
        "1234",
        "plots"
    )
    mycursor = conn.cursor()

    with open('tags.json', 'rb') as f:
        data = json.load(f)
        # print(data["movies"])

    tag = ''

    for item in data['result']:
        # print(item['tag'])
        time = datetime.fromtimestamp(int(item['timestamp']))
        print('\n')
        print(time)

        user = int(item['userId'])
        movie = int(item['movieId'])
        tag = convert_str(item['tag'])
        tag = tag.lower()
        print(tag)

        try:
            sql = "INSERT INTO Tags(userId,movieId,tag,dateT) VALUES(%s,%s,%s,%s)"
            val = (user, movie, tag, time)
            mycursor.execute(sql, val)
            conn.commit()
            print("%d and %d has been commited" % (user, movie))

        except:
            print("%d and %d has been rolled back" % (user, movie))
            conn.rollback()

    conn.close()


def castdb(data, mid):

    conn = pymysql.connect(
        "localhost",
        "root",
        "1234",
        "plots"
    )
    mycursor = conn.cursor()

    # print('printin %d ' % len(data['cast']))

    try:
        for t in data['cast']:

            cast = t['cast_id']
            pid = t['id']
            character = t['character']
            name = t['name']
            gender = t['gender']

            if gender == 1:
                gender = 'female'
            elif gender == 2:
                gender = 'male'

            print(name, cast, gender, character, mid, pid)

            try:
                sql = "INSERT INTO movie_cast( tmdb_id, cast_id, actor_id, movie_name, real_name, gender ) VALUES(%s, %s,%s,%s,%s,%s)"
                val = (mid, cast, pid, character, name, gender)
                mycursor.execute(sql, val)
                conn.commit()
                print("%d and %s has been commited\n" % (mid, name))


            except:
                print("%d and %s has been rolled back\n" % (mid, name))
                conn.rollback()
    except:
        pass

    conn.close()


def tag_counter(movie, tag):
    conn = pymysql.connect(
        "localhost",
        "root",
        "1234",
        "plots"
    )
    tcursor = conn.cursor()

    number = 0

    print(tag, movie)

    try:
        tcursor.execute('select count(*) from tags where movieid =' + str(movie) + ' and tag =\'' + tag + '\'')
        number = tcursor.fetchall()
        number = strip_tuple(number)
        print(number)
    except:
        print('An Error has occurred!!')
    conn.close()
    return number


def tag_count_db(s):
    conn = pymysql.connect(
        "localhost",
        "root",
        "1234",
        "plots"
    )
    mycursor = conn.cursor()

    mycursor.execute('select tag from tags where movieid = %d' % s)
    tags = mycursor.fetchall()
    print('the results found are: %s' % len(tags))

    for t in tags:
        tag = strip_tuple(t)
        num = tag_counter(s, tag)
        try:
            sql = "INSERT INTO Tag_count(movieId,tag,num_of_tags) VALUES(%s,%s,%s)"
            val = (s, tag, num)
            mycursor.execute(sql, val)
            conn.commit()
            print("%d and %s has been commited" % (s, tag))

        except:
            print("%d and %s has been rolled back" % (s, tag))
            conn.rollback()

    conn.close()


def tag_movie_id():
    conn = pymysql.connect(
        "localhost",
        "root",
        "1234",
        "plots"
    )
    mycursor = conn.cursor()

    mycursor.execute('select movie_id from movie_linkages')
    name = mycursor.fetchall()
    print('the results found are: %s' % len(name))

    for t in name:
        movie_id = int(strip_tuple(t))
        tag_count_db(movie_id)

    print(name)
    conn.close()


def main():
    print("THIS IS MAIN\n\n")
    moviedb()
    tagsdb()
    get_movieid()
    tag_movie_id()


if __name__ == '__main__':
    main()


