#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fake_real_reviews_helpers import remove_nan_values, from_list_of_lists
import pandas as pd

# Predict using KMeans Clustering
# NOTE: We know there are two clusters so no need to find number of clusters

# Importing the dataset
dataset = pd.read_excel('restaurant_reviews.xlsx')
X = dataset.iloc[:, [1]].values     # Reviews
y = dataset.iloc[:, 2].values       # Real or Fake

# Make X and y lists
X = from_list_of_lists(X)
y = list(y)

# Get rid of missing data (nan) values
X = remove_nan_values(X)
y = remove_nan_values(y)

# Split into Training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, 
                                                    random_state = 0)
# Vectorize training data
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)

# Train K-Means++ algorithm
from sklearn.cluster import KMeans
true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X_train)

# Print the top clusters of words between the two clusters
print("\nTop terms per cluster:\n")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("10 words from Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print()

print("Predictions:")
y_pred = []
for i in range(len(y_test)):
    vectorizer_input = []
    print("Review {0}: ".format(i), end="")
    Y = vectorizer.transform([X_test[i]])
    prediction = model.predict(Y)
    print("Prediction {0} Actual {1}".format(prediction[0], y_test[i]), end=" ")
    
    if prediction[0] == y_test[i]:
        print("Correct")
    else:
        print("Wrong")
    
    y_pred.append(prediction[0])

# Show Predictions with Logistic Regression
from sklearn.metrics import accuracy_score, f1_score
print("\nAccuracy Rate using K-Means Algorithm")
print('Testing Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Testing Recall score: {}'.format(recall_score(y_test, y_pred)))
print('Testing Precision score: {}'.format(precision_score(y_test, y_pred)))
print()