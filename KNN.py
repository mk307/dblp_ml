
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#loadng Dataset
dataset = pd.read_csv("features_and_label.csv")

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


X_train.shape
X_test.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.fit_transform(X_test)

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor()
model.fit(train_scaled, y_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

mse = mean_squared_error(y_train, model.predict(train_scaled))
mae = mean_absolute_error(y_train, model.predict(train_scaled))


from math import sqrt

print("mse = ",mse," & mae = ",mae," & rmse = ", sqrt(mse))

test_mse = mean_squared_error(y_test, model.predict(test_scaled))
test_mae = mean_absolute_error(y_test, model.predict(test_scaled))


print("mse = ",test_mse," & mae = ",test_mae," & rmse = ", sqrt(test_mse))

#Prediction
y_pred = model.predict(X_test) 
y_pred

#Accuracy 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))
print (cm)

rp = sns.regplot(x=y_test, y=y_pred)
