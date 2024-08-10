import os
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('creditcard.csv')

dataX = data.copy().drop(['Class'],axis=1)
dataY = data['Class'].copy()

featuresToScale = dataX.drop(['Time'],axis=1).columns
sX = pp.StandardScaler(copy=True)
dataX.loc[:,featuresToScale] = sX.fit_transform(data.loc[:,featuresToScale])

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_train.shape: ", y_train.shape)
print("y_test.shape: ", y_test.shape)

#k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
#for k, (train, test) in enumerate(k_fold.split(X_train, y_train)):
#    print("train: ", train.shape)
#    print("test: ", test.shape)
