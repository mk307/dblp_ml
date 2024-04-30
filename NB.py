import numpy as np
import pandas as pd
import pickle

#loadng Dataset
dataset = pd.read_csv("features_and_label.csv")

import matplotlib.pyplot as plt
import seaborn as sns

before16=dataset.loc[dataset["year"]<=2016]
after16=dataset.loc[dataset["year"]>2016]

#training data
X_train=before16.drop(["Unnamed: 0","id","Count","year"],axis=1)
y_train=before16.drop(["Unnamed: 0","id","journal_class","year"],axis=1)

#testing data
X_test=after16.drop(["Unnamed: 0","id","Count","year"],axis=1)
y_test=after16.drop(["Unnamed: 0","id","journal_class","year"],axis=1)
#y_test=y['journal_class']
dataset.head(6)

#Modeling
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
nb = classifier.fit(X_train, y_train)

#Prediction
y_pred = classifier.predict(X_test) 
y_pred

#Accuracy 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))
print (cm)

rp = sns.regplot(x=y_train, y=y_pred)

with open('article.pkl', 'wb') as file:
    pickle.dump(nb, file)