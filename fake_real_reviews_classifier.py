#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uncomment the below lines and download the following tokenizer from the Natural
Language Processing ToolKit (Only has to be done once)
"""
## Import NLTK (just once)
#import nltk
#nltk.download('punkt')

# Importing the libraries and dependencies
from fake_real_reviews_helpers import (remove_nan_values, from_list_of_lists, 
    tokenize_text, vec_for_learning)
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
    
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
# Tokenize the text
train_tagged = tokenize_text(X_train)
test_tagged = tokenize_text(X_test)

max_epochs = 100
vec_size = 20
alpha = 0.025

# Turn Document into number vector with Doc2Vec
model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(train_tagged)

# Train Doc2Vec model
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(train_tagged,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("\nModel Saved\n")

# Infer X to classifier (number) readable format
X_train = vec_for_learning(model, train_tagged)
X_test = vec_for_learning(model, test_tagged)

# Predict and train using LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(n_jobs=1, C=1e5, solver="lbfgs")
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)     

# Show Predictions with Logistic Regression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
print("Accuracy Rate using Logistic Regression")
print('Testing Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Testing Recall score: {}'.format(recall_score(y_test, y_pred)))
print('Testing Precision score: {}'.format(precision_score(y_test, y_pred)))
print()

#NOTE: F1 score combines precision and recall relative to a specific positive class 
#The F1 score can be interpreted as a weighted average of the precision and recall

# Predict and train using Support Vector Machine
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Show Predictions with SVM
print("Accuracy Rate using Support Vector Machine")
print('Testing Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Testing Recall score: {}'.format(recall_score(y_test, y_pred)))
print('Testing Precision score: {}'.format(precision_score(y_test, y_pred)))
print()

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Show Prediction with Random Forest Classifier
print("Accuracy Rate using Random Forest Classifier")
print('Testing Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Testing Recall score: {}'.format(recall_score(y_test, y_pred)))
print('Testing Precision score: {}'.format(precision_score(y_test, y_pred)))
print()



