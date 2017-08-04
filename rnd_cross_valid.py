#!/usr/bin/env python
"""
The objective of this program is to show how to use Multinomial Naive Bayes
method to classify news according to some predefined classes.

We want to measure
1) how much time does it takes to train and test the classifier.
2) the influence on the results obtained while normalizing the text
(i.e. remove punctuation and lowercase it)

The News Aggregator Data Set comes from the UCI Machine Learning Repository.
Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.
This specific dataset can be found in the UCI ML Repository at this URL:
http://archive.ics.uci.edu/ml/datasets/News+Aggregator

"""

# Import built-in and third-party modules


import numpy as np
import scipy.stats as st
# import statsmodels.stats.api as sms
import pprint
import matplotlib.pyplot as plt

from random_cross_validation import *

__author__ = 'andres.soto'
__email__ = "soto_andres@yahoo.com"
__status__ = "Prototype"

Ntimes = 3

def conf_interv(tit,data):
    print("confidence interval for "+tit, end=' ')
    print(st.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=st.sem(data)))

import_time, labels = import_data()
# print(">>>labels>>>",labels)

split_time = [0]*Ntimes
train_time = [0]*Ntimes
f1s = [0]*Ntimes
for k in range(Ntimes):
    print()
    print("cross-validation repetition ",k)
    split_time[k] = split_data()
    print()
    print("cross-validation repetition ",k)
    train_time[k], f1s[k] = train_test()

print()
print('Time consumed in each step (in seconds)')
print('import_time',import_time)
print()
# print('\t mean \t std')
# print('split_time',np.mean(split_time),np.std(split_time))
# print('train_time',np.mean(train_time),np.std(train_time))
conf_interv("split_time",split_time)
conf_interv("train_time",train_time)
# print("confidence interval for split_time")
# print(st.t.interval(0.95, len(split_time)-1, loc=np.mean(split_time), scale=st.sem(split_time)))
# print("confidence interval for train_time")
# print(st.t.interval(0.95, len(train_time)-1, loc=np.mean(train_time), scale=st.sem(train_time)))
form = '{:5.2f}'*Ntimes
af1s = np.asarray(f1s)
print()
print('Collected F1-score by category')
# print('Category \t mean \t std')
# print("nlabels",nlabels)
# print("labels",labels)
# print("af1s",af1s)
for c in range(nlabels):
    # print("index c",c)
    # print(labels[c],np.mean(af1s[:,c]),np.std(af1s[:,c]))
    # print("confidence interval for split_time")
    conf_interv("F1-score for category "+labels[c], af1s[:,c])
    # aux = af1s[:,c]
    # print(st.t.interval(0.95, len(split_time) - 1, loc=np.mean(split_time), scale=st.sem(split_time)))

print()
print('Category\t# of news')
s=0
for l in lnews:
    s+=l['Nnews']
    print(l['category'], '\t',l['Nnews'])
print('Total number of news',s)

prom = conf_mat_sum / Ntimes
print()
print("confusion matrix mean")
pprint.pprint(prom)

plt.matshow(prom)
plt.colorbar()
plt.show()

##########################################################################
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import time # provides various time-related functions.
# import re # provides regular expression matching operations
# from sklearn.preprocessing import LabelEncoder # encode labels with value between 0 and n_classes-1.
# from sklearn.utils import shuffle # Shuffle arrays in a consistent way
# from sklearn.feature_extraction.text import CountVectorizer # implements both tokenization and occurrence counting
# from sklearn.feature_extraction.text import TfidfTransformer # Transform a count matrix to a normalized tf-idf representation
# from sklearn.naive_bayes import MultinomialNB # implements Multinomial naive Bayes algorithm
# from sklearn.pipeline import Pipeline # chain multiple estimators into one
# from sklearn import metrics # includes score functions, metrics and distances

# titles = [] # list of news titles
# ncategories = [] # number of categories
# labels = [] # list of different categories (without repetitions)
# nlabels = 4 # number of categories in this case
# lnews = [] # list of dictionaries with two fields: a news and its category
#
# X_train = []
# y_train = []
# X_test = []
# y_test = []
# lmats = []
# nrows = nlabels
# ncols = nlabels
# conf_mat_sum = np.zeros((nrows, ncols))
# f1_acum = []

# def import_data():
#     global titles, ncategories, labels, lnews
#     print()
#     print('Importing data...')
#     start = time.time()
#     # importing news aggregator data via Pandas (Python Data Analysis Library)
#     news = pd.read_csv("uci-news-aggregator.csv")
#     # function head gives us the first 5 items in a column (or
#     # the first 5 rows in the DataFrame)
#     print(news.head())
#     # we want to predict the category of a news article based only on its title
#
#     categories = news['CATEGORY']
#     titles = news['TITLE']
#     labels = list(set(categories))
#     print()
#     print('possible categories', labels)
#     suma = 0
#     for l in labels:
#         dic = {
#             'category':l,
#             'Nnews':len(news.loc[news['CATEGORY'] == l])
#         }
#         lnews.append(dic)
#         print('number of ', l, ' news', dic['Nnews'])
#         suma=suma+dic['Nnews']
#     print("total number of news",suma)
#
#     # categories are literal labels, but it is better for
#     # machine learning algorithms just to work with numbers, so we will
#     # encode them
#     # LabelEncoder: encode labels with value between 0 and n_classes-1.
#
#     encoder = LabelEncoder()
#     ncategories = encoder.fit_transform(categories)
#     # return titles, ncategories, labels
#     return time.time() - start

    #@staticmethod
# def split_data():
#     global titles, ncategories
#     global X_train, y_train, X_test, y_test
#     print()
#     print('Splitting data...')
#     start = time.time()
#     # Now we should split our data into two sets:
#     # 1) a training set which is used to discover potentially predictive relationships, and
#     # 2) a test set which is used to evaluate whether the discovered relationships
#     #    hold and to assess the strength and utility of a predictive relationship.
#     N = len(titles)
#     #print('Number of news', N)
#     Ntrain = int(N * 0.7)
#
#     titles, ncategories = shuffle(titles, ncategories, random_state=0)
#
#     # in this case, we do not use the raw titles but the normalized titles
#     # normalize the TITLE column: remove punctuation and lowercase everything
#     # news['TEXT'] = [normalize_text(s) for s in news['TITLE']]
#     # titles = news['TEXT']
#     # titles = [normalize_text(s) for s in titles]
#
#     X_train = titles[:Ntrain]
#     print('X_train.shape', X_train.shape)
#     y_train = ncategories[:Ntrain]
#     print('y_train.shape', y_train.shape)
#     X_test = titles[Ntrain:]
#     print('X_test.shape', X_test.shape)
#     y_test = ncategories[Ntrain:]
#     print('y_test.shape', y_test.shape)
#     # return X_train, y_train, X_test, y_test
#     return time.time() - start
#
#     #@staticmethod
# def train_test():
#     global X_train, y_train, X_test, y_test, labels, lmats, \
#             conf_mat_sum, f1_acum
#     print()
#     print('Training and testing...')
#     start = time.time()
#     # CountVectorizer implements both tokenization and occurrence counting
#     # in a single class
#     # TfidfTransformer: Transform a count matrix to a normalized tf or tf-idf
#     # representation
#     # MultinomialNB: implements the naive Bayes algorithm for multinomially
#     # distributed data, and is one of the two classic naive Bayes variants
#     # used in text classification
#     # Pipeline: used to chain multiple estimators into one.
#     # All estimators in a pipeline, except the last one,
#     # must be transformers (i.e. must have a transform method).
#     # The last estimator may be any type (transformer, classifier, etc.).
#     print('Training...')
#
#     text_clf = Pipeline([('vect', CountVectorizer()),
#                          ('tfidf', TfidfTransformer()),
#                          ('clf', MultinomialNB()),
#                          ])
#     # clf.fit: Fit Naive Bayes classifier according to X, y
#     text_clf = text_clf.fit(X_train, y_train)
#     print('Predicting...')
#     # clf.predict: Perform classification on an array of test vectors X.
#     predicted = text_clf.predict(X_test)
#
#     # sklearn.metrics module includes score functions, performance metrics
#     # and pairwise metrics and distance computations.
#     # accuracy_score: computes subset accuracy; used to compare set of
#     #       predicted labels for a sample to the corresponding set of true labels
#     print('accuracy_score', metrics.accuracy_score(y_test, predicted))
#     print('Reporting...')
#
#     # classification_report: Build a text report showing the main classification metrics
#     print(metrics.classification_report(y_test, predicted, target_names=labels))
#     f1s = metrics.f1_score(y_test, predicted, average=None) #, labels=labels)
#     # print('f1_score',f1s)
#     f1_acum.append(f1s)
#
#     print('normalized confusion_matrix')
#
#     mat = metrics.confusion_matrix(y_test, predicted)
#     # print('mat.shape',mat.shape)
#     # for row in mat:
#     #     s=np.sum(row)
#     #     print('\t'.join(map(str, row/s)))
#     cm = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
#     pprint.pprint(cm)
#     # lmats.append(mat)
#     lmats.append(cm)
#     conf_mat_sum += cm
#     # diag = np.diag(mat)
#     # print('diagonal',diag)
#
#     return time.time() - start,f1s


