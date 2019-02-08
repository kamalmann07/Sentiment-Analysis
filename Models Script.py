import pandas as pd
import tensorflow as tf
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib as plt

# Import data from CSV
data = pd.read_csv('cleanData.csv')

data = data[['cleanedText', 'Score']]

Y = [0 if (y == 1 or y == 2 or y == 3) else 1 for y in data.Score]

# print (Y)

X = np.array(data.cleanedText)
Y = np.array(Y)

print(Y)

# from sklearn.feature_extraction.text import TfidfVectorizer
#
# tfidf = TfidfVectorizer(min_df=1, ngram_range=(2, 2))
# features = tfidf.fit_transform(X)
# X_vec = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())


vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# print(vectorizer.get_feature_names())
# print(X_vec.toarray())

# print(X_vec)

X_train, X_test, y_train, y_test = train_test_split(X_vec, Y, test_size=0.2, random_state=101)

# Nayes Bayes
# nb = MultinomialNB()
# nb.fit(X_train, y_train)
# preds = nb.predict(X_test)

# SVM Machine
# from sklearn import svm
# model = svm.LinearSVC()
# model.fit(X_train, y_train)
# preds = model.predict(X_test)

# Decision Tree
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# preds = clf.predict(X_test)

# Random Forest
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# clf.fit(X_train, y_train)
# preds = clf.predict(X_test)

# KNN Model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)

print ("Accuracy of the model is:", accuracy_score(y_test, preds))
print (classification_report(y_test, preds))



labels = [0, 1]
cm = confusion_matrix(y_test, preds)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt="d"); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['Positive', 'Negative']);
ax.yaxis.set_ticklabels(['Negative', 'Positive']);

plt.show()

# plt.figure(figsize=(12,8))
# from matplotlib import pyplot
# pyplot.scatter(y_test, marker='d', c='red')
# pyplot.scatter(preds, marker='d', c='blue')
# pyplot.show()
