import numpy as np
import math
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm  
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.datasets import make_classification

df = pd.read_csv('TrainDataBinaryClassification.csv')
pf = pd.read_csv('TestDataTwoClass.csv')
X = np.array(df)
X = np.array(df.drop(['class', 'Id'], 1))
y = np.array(df['class'])
df = X
rf = y
pf = pf.drop(pf.columns[0],axis = 1)

from imblearn.over_sampling import SMOTE
sm = SMOTE(kind='regular')
df,rf = sm.fit_sample(df, rf)


clf1 = RandomForestClassifier(max_features='sqrt')
clf1.fit(df, rf)
gh1 = clf1.predict(pf)

clf4 = svm.SVC(kernel='rbf', C=100, gamma=0.001)
clf4.fit(df, rf)
gh4 = clf4.predict(pf)

clf6 = GradientBoostingClassifier()
clf6.fit(df, rf)
gh6 = clf6.predict(pf)

clf112 = ExtraTreesClassifier(n_estimators=10, max_depth=100, min_samples_split=1, random_state=0)
clf112.fit(df, rf)
gh112 = clf112.predict(pf)


gh8 = 1*gh1 + 3*gh4 + 3*gh6 + 5*gh112
gh9 = np.zeros(len(gh8))
for i in range(len(gh8)):
    if gh8[i] > 6 :
        gh9[i] = 1


f = open('ans.csv', 'w')
np.savetxt("ans.csv",gh9)
