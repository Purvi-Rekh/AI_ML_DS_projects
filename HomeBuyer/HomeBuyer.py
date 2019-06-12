# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 01:08:53 2019

@author: purvi
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib import style
df = pd.read_csv("C:\purvi_me\machine-learning\simplilearn\hands-on-assignment\HomeBuyer.csv")

print(df.head())
print(df.describe())

X=df.iloc[:,0:2]
y=df.iloc[:,-1]

x1=df.iloc[:,0:1]
style.use('classic')

plt.plot(x1,y,color='c', linestyle='-', linewidth=2.5)
plt.show()
#print("x = {} \n\n y= {} ".format(X,y))
m = KNeighborsClassifier()
NBModel = GaussianNB()

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.4)

m.fit(x_train,y_train)
y_pred=m.predict(x_test)

print("KNN Accuracy score = {} ".format(accuracy_score(y_test,y_pred)))
print("KNN Confusion matrix = \n{} \n".format(confusion_matrix(y_test,y_pred)))
print("KNN Classification report = \n{} \n".format(classification_report(y_test,y_pred)))

NBModel.fit(x_train,y_train)
y_pred=NBModel.predict(x_test)

print(" NB Accuracy score = {} ".format(accuracy_score(y_test,y_pred)))
print("NB Confusion matrix = \n{} \n".format(confusion_matrix(y_test,y_pred)))
print("NB Classification report = \n{} \n".format(classification_report(y_test,y_pred)))


