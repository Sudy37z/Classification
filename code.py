import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("D:\Mushroom Classification data\mushrooms.csv")
dataset.describe()

dataset.info()
dataset1 = pd.get_dummies(dataset)

x = dataset1.iloc[:,2:]
y = dataset1.iloc[:,1]

# Training and testing set from KNeighborsC
from sklearn.neighbors import KNeighborsClassifier 


from sklearn.model_selection import train_test_split

kn = KNeighborsClassifier(n_neighbors = 5,metric='minkowski', p =1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,train_size=0.9,random_state=88,shuffle=True)


kn.fit(x_train,y_train)

predictionKN = kn.predict(x_test)


# Training and testing set from Decision tree


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

predictionDT = dt.predict(x_test)

### Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

predictionNB = nb.predict(x_test)

## Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

predictionLR = lr.predict(x_test)


### Cross validation

from sklearn.model_selection import cross_val_score


scoreDT = cross_val_score(dt,x,y,cv = 10)
scoreKN = cross_val_score(kn,x,y,cv=10)
scoreNB = cross_val_score(nb,x,y,cv=10)
scoreLR = cross_val_score(lr,x,y,cv=10)

### Confusion Metrics and Classification report

from sklearn.metrics import confusion_matrix,classification_report

#CONFUSSION MATRIX

cmdt = confusion_matrix(y_test, predictionDT)

cmkn = confusion_matrix(y_test, predictionKN)

cmnb = confusion_matrix(y_test, predictionNB)

cmlr = confusion_matrix(y_test, predictionLR)

#CLASSIFICATION REPORT

crdt = classification_report(y_test, predictionDT)

crkn = classification_report(y_test, predictionKN)

crnb = classification_report(y_test, predictionNB)

crlr = classification_report(y_test, predictionLR)

# ROC CURVE
from sklearn.metrics import roc_curve

y_prob = lr.predict_proba(x_test)

y_prob = y_prob[:,1]

FPR, TPR, Thresholds = roc_curve(y_test, y_prob)

plt.plot(FPR,TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')

plt.show()

# ROC AUC Score
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_prob)



