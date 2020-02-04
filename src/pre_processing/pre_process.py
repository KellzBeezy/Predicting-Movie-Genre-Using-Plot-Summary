import pymysql
import os
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def get_ids(stripper):
    stripper = stripper.strip('IMDB_ID=')
    stripper = stripper.strip('.txt')
    return stripper


def create_dir(title, overview, r_date, genre, mid, fold):
    os.chdir(fold)
    f = open("S_IMDB_ID=" + mid + ".txt", "w+", encoding="utf-8")
    f.write(title + "\n\nOVERVIEW:\t" + overview + '\n\n\n' + r_date + ' \n\n' + genre)
    os.chdir('C:\\Users\\KELLZ BITCHES\\Documents\\predicting-movie-genre-using-plot-summary\\src\\pre_processing')


def main():
    path = './movies'
    folder = [r for r in os.listdir(path)]

    for t in folder:
        print(os.getcwd())
        os.chdir('C:\\Users\\KELLZ BITCHES\Documents\\predicting-movie-genre-using-plot-summary\\src\\pre_processing')
        print(os.getcwd())
        fold = t + '_stop'
        print(fold)
        os.mkdir(fold)
        os.chdir('C:\\Users\\KELLZ BITCHES\\Documents\\predicting-movie-genre-using-plot-summary\\src\\pre_processing')

        p = './movies/' + t + '/'
        text = [r for r in os.listdir(p)]
        print(text)

        index = 0

        for rd in text:
            with open(p + rd, 'r', encoding='utf8') as f:

                mid = get_ids(text[index])
                index = index + 1
                all_i = f.readlines()
                data = all_i[4].strip('OVERVIEW:')
                data = data.lower()
                title = all_i[2]
                r_date = all_i[6]
                genre = all_i[8]
                print(title, r_date, genre)
                data = data.strip('\t')
                string = stop_words_fn(data)
                create_dir(title, string, r_date, genre, mid, fold)
                # print(mid)


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


if __name__ == '__main__':
    main()
