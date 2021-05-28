# -*- coding: utf-8 -*-
"""
Created on Mon May 24 22:22:10 2021

@author: aelen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import tensorflow as tf
from tensorflow import keras

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


data=pd.read_csv('mushrooms.csv')

pd.set_option('display.max_columns', None)

labelEncoder = LabelEncoder()
data=data.apply(LabelEncoder().fit_transform)

# X = data.iloc[:,1:23]
X=data.drop('class', axis=1)
T = data['class']

#splitting the dataset

xTrain, xTest, tTrain, tTest = train_test_split(X,T, test_size=0.5)

scaler = StandardScaler()
scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

xTrain=xTrain/255.0
xTest=xTest/255.0

mlp = MLPClassifier(solver='adam', alpha=1e-5, verbose=1,max_iter=1000, activation='relu',
                    hidden_layer_sizes=(3,3,3))
# knc = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)

#Training the data

mlp.fit(xTrain, tTrain)

yTrain = mlp.predict(xTrain)
yTest = mlp.predict(xTest)

print('Train accuracy is: ', accuracy_score(tTrain, yTrain))
print('Test accuracy is: ', accuracy_score(tTest, yTest))


print('Classification report is: ') 
print(classification_report(yTest, yTrain))

print('Confusion matrix is:')
print(confusion_matrix(yTest, yTrain))

# plt.close('all')
# plt.figure()
# loss_values=knc.loss_curve_
# plt.plot(loss_values)
# plt.title("Loss function")