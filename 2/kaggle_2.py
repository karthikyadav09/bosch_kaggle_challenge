import numpy as np
import math
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv('TrainDataMultiClassClassification.csv')
pf = pd.read_csv('TestDataMultiClass.csv')
df = df.dropna()
rf = df['class']
df.drop(df.columns[[0,129]], axis=1, inplace=True)
pf = pf.drop(pf.columns[0],axis = 1)

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm  
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
import xgboost as xgb
np.set_printoptions(threshold=np.inf)
from xgboost.sklearn import XGBClassifier


X_train = df
y_train = rf
X_test = pf
dim_red = PCA(n_components=12)
dim_red.fit(X_train)
X_train=dim_red.transform(X_train)
X_test=dim_red.transform(X_test)

from imblearn.over_sampling import SMOTE
sm = SMOTE(kind='regular')
X_train,y_train = sm.fit_sample(X_train,y_train)


clf1 = RandomForestClassifier(max_features='sqrt')
clf1.fit(X_train,y_train)
gh1 = clf1.predict(X_test)

clf2 = DecisionTreeClassifier()
clf2.fit(X_train,y_train)
gh2 = clf2.predict(X_test)

clf3 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', weights='distance')
clf3.fit(X_train,y_train)
gh3 = clf3.predict(X_test)

clf4 = svm.SVC(decision_function_shape='ovo',kernel= 'rbf')
clf4.fit(X_train,y_train)
gh4 = clf4.predict(X_test)

clf6 = GradientBoostingClassifier()
clf6.fit(X_train,y_train)
gh6 = clf6.predict(X_test)


eclf = VotingClassifier(estimators=[ ('rf', clf1), ('dt', clf2), ('knn', clf3), ('svm', clf4), ('gbc', clf6)],voting='hard', weights=[1,1,2,2,1])
eclf = eclf.fit(X_train,y_train)
egh = eclf.predict(X_test)

f = open('mut5.csv', 'w')
np.savetxt("mut5.csv",egh)


# f1 = pd.read_csv("8140.csv")
# f2 = pd.read_csv('mut5.csv')
# f1 = np.array(f1['class'])
# f2 = np.array(f2['class'])

# f3 = np.zeros(len(f2))
# for i in range(0,len(f2)):
#     if f1[i] == 0:
#         f3[i] = 0
#     else:
#         f3[i] = f2[i]
        
# f3 = pd.DataFrame(f3)
# f3.to_csv('new4.csv')