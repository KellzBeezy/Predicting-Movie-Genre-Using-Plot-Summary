import pymysql
import json
import csv
import sys
import pandas as pd
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold


def normalize(xtrain_tfidf, xtrain):
    xtrain_tfidf_new = []
    count = 0
    new_xtrain_tfidf = xtrain_tfidf.toarray()
    for i in xtrain:
        if count < len(xtrain):
            n = word_tokenize(i)
            print(i)
            print(n)
            a1 = additional_features(n)
            print(len(new_xtrain_tfidf[count]))
            new_xtrain_tfidf1 = np.append(new_xtrain_tfidf[count], a1)
            xtrain_tfidf_new.append(new_xtrain_tfidf1)
            print(count)
            print("\n")
            count = count + 1

    # next work with more than 1000 feature names to see the maximum number of feaures that it can accommodate !!!

    # xtrain_tfidf_new = np.asmatrix(xtrain_tfidf_new)
    xtrain_tfidf_new = sparse.csr_matrix(xtrain_tfidf_new)

    return xtrain_tfidf_new


def additional_features(array):
    feat = []
    add = []

    anger = disgust = fear = joy = sad = surprise = 0.00
    for i in array:
        conn = pymysql.connect(
            "localhost",
            "root",
            "1234",
            "plots"
        )
        curr = conn.cursor()
        curr.execute(
            "select anger, disgust, fear, joy, sad, surprise from emotions where word = '%s'" % i
        )
        data = curr.fetchone()

        count = 1

        try:
            if data:
                for item in list(data):
                    if count == 1:
                        anger = item + anger
                        count = count + 1
                    elif count == 2:
                        disgust = item + disgust
                        count = count + 1
                    elif count == 3:
                        fear = fear + item
                        count = count + 1
                    elif count == 4:
                        joy = joy + item
                        count = count + 1
                    elif count == 5:
                        sad = sad + item
                        count = count + 1
                    elif count == 6:
                        surprise = surprise + item
                        count = count + 1

                # a = [anger, disgust, fear, joy, sad, surprise]
                # feat.append(a)
        except:
            continue
    add.append(round(anger, 3))
    add.append(round(disgust, 3))
    add.append(round(fear, 3))
    add.append(round(joy, 3))
    add.append(round(sad, 3))
    add.append(round(surprise, 3))
    print(add)

    for item in add:
        if len(array) > 0:
            item = round(item, 3) / len(array)
            feat.append(round(item, 3))
        else:
            item = round(item, 3)
            feat.append(item)

    print(feat)

    return feat


def stop_words_fn(word):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(word)
    # result = [r for r in tokens if not r in stop_words]
    # print(result)
    result = []

    for w in tokens:
        if w not in stop_words:
            result.append(w)

    word = ' '.join([str(t) for t in result])
    # print(tokens)
    return word


def c_plot(text):
    text = re.sub("\'", "", text)
    text = text.strip(".,")
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    text = text.lower()

    return text


def get_genres():

    conn = pymysql.connect(
        "localhost",
        "root",
        "1234",
        "plots"
    )
    curr = conn.cursor()
    curr.execute("select count(*) from movie_genre")
    # curr.execute("SELECT  tmdb_id,  Overview from movie_genre")
    curr.execute(
        # "SELECT JSON_OBJECT('id', tmdb_Id, 'genre1', Genre_ids0, 'genre2' , Genre_ids1, 'genre3', Genre_ids2, 'genre4', Genre_ids3 ) from movie_genre")
        "select Genre_ids0, Genre_ids1, Genre_ids2, Genre_ids3 from movie_genre")

    data = [{"genre1": col1, "genre2": col2, "genre3": col3, "genre4": col4} for (col1, col2, col3, col4) in
            curr.fetchall()]

    genre = []

    genres = []
    # data = json.loads(data)

    for sample in data:
        sample = json.dumps(sample)
        genres.append(list(json.loads(sample).values()))

    print(genres[1])

    gent = []

    for item in genres:
        # eliminating null values, that is the genre columns that are null
        gen = [n for n in item if n]
        gent.append(gen)

    print(gent[1])
    # sys.exit(0)
    return gent


pd.set_option('display.max_colwidth', 900)

meta = pd.read_csv("C:\\Users\\MiraComnputers\\Desktop\\datamovies.csv", sep=';', header=None)
# print(meta.head())
# sys.exit(0)

meta.columns = ["tmdb_id", "title"]
# print(meta.head())
# print(meta['title'])
# sys.exit(0)

file = "C:\\Users\\MiraComnputers\\Desktop\\PycharmProjects\\test_pred\\summary1.txt"
plots = []

with open(file, 'r', encoding='utf8') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        plots.append(row)
        # print(row)
        # sys.exit(0)
# print(plots[0][0])
# sys.exit(0)
movie_id = []
plot = []

for i in tqdm(plots):
    try:
        plot.append(i[1])
        # print(i[1])
        movie_id.append(i[0])
        # print(i[0])

    except:
        continue

gent = get_genres()

# movies = pd.DataFrame({'tmdb_id': movie_id, 'plot': plot})
movies = pd.DataFrame({'tmdb_id': movie_id, 'plot': plot, 'genre': gent})
print(movies['plot'])

# sys.exit(0)

meta['tmdb_id'] = meta['tmdb_id'].astype(str)

# print(meta['movie_id'])

# print()

# merge meta with movies

movies = pd.merge(movies, meta[['tmdb_id', 'title']], on='tmdb_id')

print(movies.head())

print(movies.shape)
# sys.exit(0)

print(movies['genre'])

# get all genre tags in a list

all_genres = sum(gent, [])
print(all_genres)

movies['new_plot'] = movies['plot'].apply(lambda x: c_plot(x))
# print(movies['new_plot'])

movies['new_plot'] = movies['new_plot'].apply(lambda x: stop_words_fn(x))
# print(movies['new_plot'])
# sys.exit(0)


multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies['genre'])

# transform target variable
y = multilabel_binarizer.transform(movies['genre'])
print(y)
# sys.exit(0)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=5000)

# print(tfidf_vectorizer)
# sys.exit(0)
# split dataset into training and validation set
# xtrain, xval, ytrain, yval = train_test_split(movies['new_plot'], y, test_size=0.2, random_state=12)

# using k-fold cross validation with 5 splits

kf = KFold(n_splits=5, random_state=12, shuffle=True)

accuracies = []
f1scores = []
recall = []
precision = []

# 5 splits = 80:20 split
for tr_index, ts_index in kf.split(movies['new_plot']):

    xtrain = movies['new_plot'][tr_index]
    xval = movies['new_plot'][ts_index]
    ytrain = y[tr_index]
    yval = y[ts_index]

    # print(xtrain['tmdb_id'])
    # sys.exit(0)

    # print(y)
    # sys.exit(0)
    # for i in xval:
        # print(i)
        # sys.exit(0)

    # sys.exit(0)
    # g = ['5', '6', '5', '0', '3', '2']
    # lb = LabelBinarizer()
    # g1 = lb.fit_transform(g)
    # gtdf = tfidf_vectorizer.fit_transform(g)
    # create TF-IDF features

    # FOR THE TRAIN DATA SET

    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    print(xtrain_tfidf.shape)
    # sys.exit(0)
    # feature_names = tfidf_vectorizer.get_feature_names()
    # dense = xtrain_tfidf.todense()
    # df = pd.DataFrame(dense, columns=feature_names)
    # df.to_csv(r'tfidf_all.csv', sep=',', encoding='utf-8', header='true')
    # sys.exit(0)

    # sys.exit(0)
    # for i in dense:
        # print(i)
        # sys.exit(0)

    # FOR THE VALIDATION DATA SET

    feature_names = tfidf_vectorizer.get_feature_names()
    feature_names.append("anger")
    feature_names.append("disgust")
    feature_names.append("fear")
    feature_names.append("joy")
    feature_names.append("sad")
    feature_names.append("surprise")

    # include emotions
    xtrain_tfidf_new = normalize(xtrain_tfidf, xtrain)
    # print(xtrain_tfidf_new.shape)

    # FOR THE VALIDATION DATA SET
    xval_tfidf = tfidf_vectorizer.transform(xval)

    # include emotions
    xval_tfidf_new = normalize(xval_tfidf, xval)

    # df = pd.DataFrame(dense, columns=feature_names)
    # df.to_csv(r'validation_data.csv', sep=',', encoding='utf-8', header='true')


    # sys.exit(0)
    lr = LogisticRegression()

    clf = OneVsRestClassifier(lr)

    # SVC.fit(xtrain, ytrain)
    # print(SVC.score(xval, yval))

    # sys.exit(0)
    # fit model on train data

    # clf.fit(xtrain_tfidf, ytrain)
    clf.fit(xtrain_tfidf_new, ytrain)

    # make predictions for validation set
    # y_pred = clf.predict(xval_tfidf)

    y_pred = clf.predict(xval_tfidf_new)

    print("this is y_pred", y_pred)
    # sys.exit(0)
    multilabel_binarizer.inverse_transform(y_pred)

    # evaluate performance
    print("First fscore: ", f1_score(yval, y_pred, average="micro"))
    print("Precision score: ", precision_score(yval, y_pred, average="micro"))
    print("Accuracy score: ", accuracy_score(yval, y_pred))
    print("Recall score: ", recall_score(yval, y_pred, average="micro"))

    # predict probabilities

    # y_pred_prob = clf.predict_proba(xval_tfidf)
    y_pred_prob = clf.predict_proba(xval_tfidf_new)
    print(y_pred_prob)

    threshold = 0.3

    # threshold value 0.4_fscore = 0.533 ### 0.2_fscore = 0.583 ### 0.3_fscore = 0.574
    new_pred = (y_pred_prob >= threshold).astype(int)
    print("yval", yval)
    print("pred", new_pred)

    accuracies.append(accuracy_score(yval, new_pred))
    recall.append(recall_score(yval, new_pred, average="micro"))
    precision.append(precision_score(yval, new_pred, average="micro"))
    f1scores.append(f1_score(yval, new_pred, average="micro"))
    ft = len(tfidf_vectorizer.get_feature_names())


accuracy = np.mean(accuracies)
recalls = np.mean(recall)
f1score = np.mean(f1scores)
precisions = np.mean(precision)

print("ACCURACY: ", accuracy)
print("RECALL: ", recalls)
print("F-MEASURE: ", f1score)
print("PRECISION: ", precisions)


def predict(plot):
    plot_t = c_plot(plot)
    plot_t = stop_words_fn(plot_t)
    plot_vec = tfidf_vectorizer.transform([plot_t])
    clf.predict(plot_vec)
    y_pred_prob = clf.predict_proba(plot_vec)
    qw_pred = (y_pred_prob >= threshold).astype(int)
    print('\nin the inference\n')
    return multilabel_binarizer.inverse_transform(qw_pred)


t = "Inspired by the viral New York Magazine article, Hustlers follows a crew of savvy former strip club employees " \
    "who band together to turn the tables on their Wall Street clients."

print("\npredicted: ", predict(t))

sys.exit(0)