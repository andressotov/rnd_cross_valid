import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time # provides various time-related functions.
from sklearn.preprocessing import LabelEncoder # encode labels with value between 0 and n_classes-1.

titles = [] # list of news titles
ncategories = [] # number of categories
labels = [] # list of different categories (without repetitions)
nlabels = 4 # number of categories in this case
lnews = [] # list of dictionaries with two fields: a news and its category

def import_data():
    global titles, ncategories, labels, lnews
    print()
    print('Importing data...')
    start = time.time()
    # importing news aggregator data via Pandas (Python Data Analysis Library)
    news = pd.read_csv("uci-news-aggregator.csv")
    # function head gives us the first 5 items in a column (or
    # the first 5 rows in the DataFrame)
    print(news.head())
    # we want to predict the category of a news article based only on its title

    categories = news['CATEGORY']
    titles = news['TITLE']
    # labels.extend(list(set(categories)))
    labels = list(set(categories))
    print()
    print('possible categories', labels)
    suma = 0
    for l in labels:
        dic = {
            'category':l,
            'Nnews':len(news.loc[news['CATEGORY'] == l])
        }
        lnews.append(dic)
        print('number of ', l, ' news', dic['Nnews'])
        suma=suma+dic['Nnews']
    print("total number of news",suma)

    # categories are literal labels, but it is better for
    # machine learning algorithms just to work with numbers, so we will
    # encode them
    # LabelEncoder: encode labels with value between 0 and n_classes-1.

    encoder = LabelEncoder()
    ncategories = encoder.fit_transform(categories)
    # return titles, ncategories, labels
    return time.time() - start, labels

from sklearn.utils import shuffle # Shuffle arrays in a consistent way

X_train = []
y_train = []
X_test = []
y_test = []

def split_data():
    global titles, ncategories
    global X_train, y_train, X_test, y_test
    print()
    print('Splitting data...')
    start = time.time()
    # Now we should split our data into two sets:
    # 1) a training set which is used to discover potentially predictive relationships, and
    # 2) a test set which is used to evaluate whether the discovered relationships
    #    hold and to assess the strength and utility of a predictive relationship.
    N = len(titles)
    #print('Number of news', N)
    Ntrain = int(N * 0.7)

    titles, ncategories = shuffle(titles, ncategories, random_state=0)

    # in this case, we do not use the raw titles but the normalized titles
    # normalize the TITLE column: remove punctuation and lowercase everything
    # news['TEXT'] = [normalize_text(s) for s in news['TITLE']]
    # titles = news['TEXT']
    # titles = [normalize_text(s) for s in titles]

    X_train = titles[:Ntrain]
    print('X_train.shape', X_train.shape)
    y_train = ncategories[:Ntrain]
    print('y_train.shape', y_train.shape)
    X_test = titles[Ntrain:]
    print('X_test.shape', X_test.shape)
    y_test = ncategories[Ntrain:]
    print('y_test.shape', y_test.shape)
    # return X_train, y_train, X_test, y_test
    return time.time() - start

from sklearn.feature_extraction.text import CountVectorizer # implements both tokenization and occurrence counting
from sklearn.feature_extraction.text import TfidfTransformer # Transform a count matrix to a normalized tf-idf representation
from sklearn.naive_bayes import MultinomialNB # implements Multinomial naive Bayes algorithm
from sklearn.pipeline import Pipeline # chain multiple estimators into one
from sklearn import metrics # includes score functions, metrics and distances
import numpy as np
import pprint


lmats = []
nrows = nlabels
ncols = nlabels
conf_mat_sum = np.zeros((nrows, ncols))
f1_acum = []

def train_test():
    global X_train, y_train, X_test, y_test, labels, lmats, \
            conf_mat_sum, f1_acum
    print()
    print('Training and testing...')
    start = time.time()
    # CountVectorizer implements both tokenization and occurrence counting
    # in a single class
    # TfidfTransformer: Transform a count matrix to a normalized tf or tf-idf
    # representation
    # MultinomialNB: implements the naive Bayes algorithm for multinomially
    # distributed data, and is one of the two classic naive Bayes variants
    # used in text classification
    # Pipeline: used to chain multiple estimators into one.
    # All estimators in a pipeline, except the last one,
    # must be transformers (i.e. must have a transform method).
    # The last estimator may be any type (transformer, classifier, etc.).
    print('Training...')

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
    # clf.fit: Fit Naive Bayes classifier according to X, y
    text_clf = text_clf.fit(X_train, y_train)
    print('Predicting...')
    # clf.predict: Perform classification on an array of test vectors X.
    predicted = text_clf.predict(X_test)

    # sklearn.metrics module includes score functions, performance metrics
    # and pairwise metrics and distance computations.
    # accuracy_score: computes subset accuracy; used to compare set of
    #       predicted labels for a sample to the corresponding set of true labels
    print('accuracy_score', metrics.accuracy_score(y_test, predicted))
    print('Reporting...')

    # classification_report: Build a text report showing the main classification metrics
    print(metrics.classification_report(y_test, predicted, target_names=labels))
    f1s = metrics.f1_score(y_test, predicted, average=None) #, labels=labels)
    # print('f1_score',f1s)
    f1_acum.append(f1s)

    print('normalized confusion_matrix')

    mat = metrics.confusion_matrix(y_test, predicted)
    # print('mat.shape',mat.shape)
    # for row in mat:
    #     s=np.sum(row)
    #     print('\t'.join(map(str, row/s)))
    cm = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
    pprint.pprint(cm)
    # lmats.append(mat)
    lmats.append(cm)
    conf_mat_sum += cm
    # diag = np.diag(mat)
    # print('diagonal',diag)

    return time.time() - start,f1s

