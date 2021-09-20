#!/usr/bin/env python
# coding: utf-8

# TABLE OF CONTENTS¶
# IMPORTING LIBRARIES
# 
# LOADING DATA
# 
# DATA PREPROCESSING
# 
# DATA ANALYSIS
# 
# MODEL BUILDING
# 
# CONCLUSIONS

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
np.random.seed(0)


# In[2]:


##Importing Important Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Loading the dataset
data = pd.read_csv("data/fetal_health.csv")


# In[4]:


#Print the first 5 rows of the dataframe.
data.head()


# In[5]:


data.head().T


# In[6]:


data.info


# In[7]:


data['fetal_health'].unique()


# In[8]:


sns.countplot(data['fetal_health'])


# In[9]:


sns.countplot(data['fetal_health'])


# In[10]:


hist_plot = data.hist(figsize=(20,20))


# Correlation Matrix

# In[11]:


#correlation matrix
corrmat= data.corr()
plt.figure(figsize=(15,15))  
sns.heatmap(corrmat,annot=True, cmap="PuOr", center=0)


#  Data Pre-processing

# From the Correlation matrix, we can say that 'histogram_mode', 'histogram_mean' and 'histogram_median' 
# are highly correlated to each other. Also, 'histogram_min' and 'histogram_width' are highly negatively 
# correlated. So we will remove 'histogram_mode', 'histogram_median' and 'histogram_min' 
# columns from the dataset.

# In[12]:


data = data.drop(['histogram_min','histogram_median','histogram_mode'], axis=1)
data


# In[13]:


data.isnull().sum()


# Find Missing Values :
# The real-world data often has a lot of missing values. The cause of missing values can be data corruption 
# or failure to record data. The handling of missing data is very important during the preprocessing of the dataset 
# as many machine learning algorithms do not support missing values.

# Splitting the Data

# In[14]:


# Splitting data into 75% train set and 25% test set

X = data.drop(['fetal_health'], axis=1)
y = data['fetal_health']


# In[15]:


#importing train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[17]:


## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[18]:


## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[19]:


##Result Visualisation
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')


# In[20]:


##The best result is captured at k = 4 hence 4 is used for the final model
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(4)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)


# In[29]:


# for no.of nighbors from 1 -10, graph the k_fold scores
nighb = []
max = 0
k = 0

for i in range(1,41,1):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance')
    score = cross_val_score(knn, X, y, cv=4).mean()
    if max < score:
        max = score 
        k = i
    nighb.append(score)

print('The Optimal K :', k)    
nighb


# In[30]:


import matplotlib.pyplot as plt
plt.plot(range(1,41,1), nighb)
plt.xlabel('No.of Nighbors')
plt.ylabel('K-fold Scores')


# In[29]:


#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[30]:


y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[31]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='sag',tol=0.01,random_state=0)


# In[32]:


X_train.shape, y_train.shape


# In[33]:


from sklearn import datasets
from sklearn import svm


# In[34]:


##K-fold cross-validation score for k=5.
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)
scores


# In[35]:


from sklearn.model_selection import ShuffleSplit
n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(clf, X, y, cv=cv)


# In[36]:


#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# Find out the best parameters using GridSearchCV

# In[37]:


params = {"tol": [0.0001,0.0002,0.0003],
          "intercept_scaling": [1, 2, 3, 4]
         }


# In[38]:


import numpy as np
from sklearn.model_selection import StratifiedKFold
cv_method = StratifiedKFold(n_splits=3, 
                            random_state=42)


# In[39]:


from sklearn.model_selection import GridSearchCV
GridSearchCV_LR = GridSearchCV(estimator=LogisticRegression(), 
                       param_grid=params,
                       cv=cv_method,
                       n_jobs=2,
                       scoring="accuracy"
                      )


# In[40]:


GridSearchCV_LR.fit(X_train, y_train)


# In[41]:


best_params_LR = GridSearchCV_LR.best_params_
best_params_LR


# In[42]:


lr = LogisticRegression(C=10, intercept_scaling=1, tol=0.0001, penalty="l2", solver="liblinear", random_state=42)
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# Prediction

# In[43]:


pred = lr.predict(X_test)


# In[44]:


#Classification Report
print("Classification Report")
print(classification_report(y_test, pred))


# In[45]:


#Confusion Matrix
ax= plt.subplot()
sns.heatmap(confusion_matrix(y_test, pred), annot=True, ax = ax, cmap = "BuPu");

# labels, title and ticks
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Normal", "Suspect", "Pathological"])


# In[46]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[47]:


#Find out the best parameters using GridSearchCV
params_RF = {"min_samples_split": [2, 6, 20],
             "min_samples_leaf": [1, 4, 16],
             "n_estimators" :[100,150, 200, 250],
             "criterion": ["gini"]             
            }


# In[48]:


GridSearchCV_RF = GridSearchCV(estimator=RandomForestClassifier(), 
                                param_grid=params_RF, 
                                cv=cv_method,
                                n_jobs=2,
                                scoring="accuracy"
                                )


# In[49]:


GridSearchCV_RF.fit(X_train, y_train)


# In[50]:


best_params_RF = GridSearchCV_RF.best_params_
best_params_RF


# In[51]:


rf = RandomForestClassifier(criterion="gini", n_estimators = 100, min_samples_leaf=1, min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)


# In[52]:


pred_rf = rf.predict(X_test)


# In[53]:


#Classification Report
print("Classification Report")
print(classification_report(y_test, pred_rf))


# In[54]:


#Confusion Matrix
ax= plt.subplot()
sns.heatmap(confusion_matrix(y_test, pred_rf), annot=True, ax = ax, cmap = "BuPu")

# labels, title and ticks
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Normal", "Suspect", "Pathological"])


# In[60]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
params = {
            'base_estimator': [
                                    DecisionTreeClassifier(max_depth=1), 
                                    DecisionTreeClassifier(max_depth=5),
                                    DecisionTreeClassifier(max_depth=10)
                              ],
            'n_estimators':[20,40,60,80,100]
            }
ada = GridSearchCV(AdaBoostClassifier(),param_grid=params,cv=4)
ada.fit(X_train,y_train)


# In[61]:


ada.best_params_


# In[62]:


ada.best_score_


# In[63]:


y_pred_ada = ada.predict(X_test)
confusion_matrix(y_test,y_pred_ada)


# In[64]:


print(classification_report(y_test,y_pred_ada))


# In[55]:


from sklearn.ensemble import AdaBoostClassifier

adaBoost = AdaBoostClassifier(base_estimator=None,
                              learning_rate=1.0,
                              n_estimators=100)

adaBoost.fit(X_train, y_train)

y_pred = adaBoost.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[59]:


import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
n_estimators = [100,140,145,150,160, 170,175,180,185];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
learning_r = [0.1,1,0.01,0.5]

parameters = {'n_estimators':n_estimators,
              'learning_rate':learning_r
              
        }
grid = GridSearchCV(AdaBoostClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.
                                     ),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(X,y) 


# In[69]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
model.score(X_test, y_test)


# In[69]:


#A quick model selection process
#pipelines of models( it is short was to fit and pred)
pipeline_lr=Pipeline([('lr_classifier',LogisticRegression(random_state=42))])

pipeline_dt=Pipeline([ ('dt_classifier',DecisionTreeClassifier(random_state=42))])

pipeline_rf=Pipeline([('rf_classifier',RandomForestClassifier())])

pipeline_svc=Pipeline([('sv_classifier',SVC())])


# In[70]:


# List of all the pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_svc]

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest', 3: "SVC"}


# In[71]:


# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)


# In[72]:


#cross validation on accuracy 
cv_results_accuracy = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train, cv=10 )
    cv_results_accuracy.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))


# So Random Forest does best amongst the models to be the most accurate. Let us build a better random forest with grid search cv. Let's find out how it performs on testset
