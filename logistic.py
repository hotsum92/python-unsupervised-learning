import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

data = pd.read_csv('creditcard.csv')

dataX = data.copy().drop(['Class'],axis=1)
dataY = data['Class'].copy()

featuresToScale = dataX.drop(['Time'],axis=1).columns
sX = pp.StandardScaler(copy=True)
dataX.loc[:,featuresToScale] = sX.fit_transform(data.loc[:,featuresToScale])

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)

penalty = 'l2'
C = 1.0
class_weight = 'balanced'
random_state = 2018
solver = 'liblinear'
n_jobs = 1

logReg = LogisticRegression(penalty=penalty, C=C,
            class_weight=class_weight, random_state=random_state,
                            solver=solver, n_jobs=n_jobs)

model = logReg

filename = 'logistic_model.sav'

if os.path.isfile(filename):
    model = joblib.load(filename)
else:
    model.fit(X_train, y_train)
    joblib.dump(model, filename)

loglossTraining = log_loss(y_train,
                           model.predict_proba(X_train)[:,1])
print('Training Log Loss: ', loglossTraining)

loglossTest = log_loss(y_test,
                       model.predict_proba(X_test)[:,1])
print('Test Log Loss: ', loglossTest)

y_test.reset_index(drop=True, inplace=True)
preds = pd.concat([y_test, pd.Series(model.predict_proba(X_test)[:,1])], axis=1)
preds.columns = ['trueLabel', 'prediction']
predictionsBasedOnKFoldsLogisticRegression = preds.copy()

precision, recall, thresholds = \
    precision_recall_curve(preds['trueLabel'],
                           preds['prediction'])

average_precision = average_precision_score(preds['trueLabel'],
                                            preds['prediction'])

plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
          average_precision))
plt.savefig('precision_recall.png')

fpr, tpr, thresholds = roc_curve(preds['trueLabel'],
                                 preds['prediction'])

areaUnderROC = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic: \
          Area under the curve = {0:0.2f}'.format(areaUnderROC))
plt.legend(loc="lower right")
plt.savefig('roc.png')
