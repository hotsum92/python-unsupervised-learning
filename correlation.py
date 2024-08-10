import os
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp
from scipy.stats import pearsonr

data = pd.read_csv('creditcard.csv')

dataX = data.copy().drop(['Class'],axis=1)
print(dataX.describe())
dataY = data['Class'].copy()

featuresToScale = dataX.drop(['Time'],axis=1).columns
sX = pp.StandardScaler(copy=True)
dataX.loc[:,featuresToScale] = sX.fit_transform(data.loc[:,featuresToScale])

correlationMatrix = pd.DataFrame(data=[],index=dataX.columns,columns=dataX.columns)

print(dataX.head())
print(pearsonr(dataX.loc[:,'V1'],dataX.loc[:,'V2']))

for i in dataX.columns:
    for j in dataX.columns:
        correlationMatrix.loc[i,j] = np.round(pearsonr(dataX.loc[:,i],dataX.loc[:,j])[0],2)

print(correlationMatrix)
