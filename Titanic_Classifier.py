#!/usr/bin/env python
# coding: utf-8

# In[93]:


#import python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


train=pd.read_csv("train.csv")


# In[3]:


train.head()


# In[4]:


train.tail()


# In[5]:


train.info()


# In[6]:


train.isna().sum()


# In[7]:


sum(train.duplicated())


# In[8]:


train_clean=train.copy()


# In[9]:


train_clean.head()


# In[10]:


cols = ["Age","SibSp","Parch","Ticket","Fare","Pclass","Sex"]
train_clean[cols] = train_clean[cols].replace({'0':np.nan, 0:np.nan})


# In[11]:


train_clean.isna().sum()


# In[12]:


train_clean.drop(['PassengerId','Name','Cabin','Ticket','SibSp','Parch'],axis=1,inplace=True)


# In[13]:


train_clean.isna().sum()


# In[14]:


train_clean['Age'].mean()


# In[15]:


train_clean['Age'].median()


# In[16]:


train_clean['Age'].mode()


# In[17]:


train_clean['Age'].fillna(30.0,inplace=True)


# In[18]:


train_clean.isna().sum()


# In[19]:


train_clean['Fare'].mean()


# In[20]:


train_clean['Fare'].median()


# In[21]:


train_clean['Fare'].mode()


# In[22]:


train_clean['Fare'].fillna(33,inplace=True)


# In[23]:


train_clean['Embarked'].mode()


# In[24]:


train_clean['Embarked'].fillna('S',inplace=True)


# In[25]:


train_clean.isna().sum()


# In[26]:


train_clean['Age'].plot(kind='box')


# In[27]:


#how many male and female survived
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train_clean,palette='rainbow')


# In[28]:


#passenger classwise survival
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train_clean,palette='rainbow')


# In[29]:


train_clean.info()


# In[ ]:





# In[ ]:





# In[30]:


#passenger classwise survival
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train_clean,palette='rainbow')


# In[31]:


sex=pd.get_dummies(train_clean.Sex)


# In[44]:


sex.head()


# In[33]:


embarked=pd.get_dummies(train_clean.Embarked)


# In[43]:


embarked.head()


# In[36]:


new_train=pd.concat([train_clean,sex,embarked],axis='columns')


# In[37]:


new_train.head()


# In[39]:


new_train.drop(['Sex','Embarked'],inplace=True,axis=1)


# In[40]:


new_train.head()


# In[42]:


new_train.drop('Survived',axis=1).head()


# In[45]:


from sklearn.model_selection import train_test_split


# In[60]:


x_train,x_test,y_train,y_test=train_test_split(new_train.drop('Survived',axis=1),new_train['Survived'],test_size=0.30,random_state=101)


# In[61]:


from sklearn.linear_model import LogisticRegression


# In[62]:


logmodel=LogisticRegression()


# In[63]:


logmodel.fit(x_train,y_train)


# In[64]:


prediction=logmodel.predict(x_test)


# In[65]:


from sklearn.metrics import confusion_matrix


# In[66]:


accuracy=confusion_matrix(y_test,prediction)


# In[67]:


accuracy


# In[68]:


from sklearn.metrics import accuracy_score


# In[69]:


accuracy=accuracy_score(y_test,prediction)


# In[70]:


accuracy


# #### Cross Validation

# In[74]:


from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
scores = cross_val_score(lr, new_train.drop('Survived',axis=1),new_train['Survived'] , cv=10)
accuracy_cross = np.mean(scores)
print(scores)
print(accuracy_cross)


# In[86]:


rf=RandomForestClassifier(n_estimators=40)
scores_rf = cross_val_score(rf, new_train.drop('Survived',axis=1),new_train['Survived'] , cv=11)
accuracy_cross_rf = np.mean(scores_rf)
print(scores_rf)
print(accuracy_cross_rf)


# In[88]:


#svm=SVC()
#scores_svm = cross_val_score(svm, new_train.drop('Survived',axis=1),new_train['Survived'] , cv=11)
#accuracy_cross_svm = np.mean(scores_svm)
#print(scores_svm)
#print(accuracy_cross_svm)


# In[91]:


ada = AdaBoostClassifier(n_estimators=80, random_state=0)
scores_ada = cross_val_score(ada, new_train.drop('Survived',axis=1),new_train['Survived'] , cv=11)
accuracy_cross_ada = np.mean(scores_ada)
print(scores_ada)
print(accuracy_cross_ada)


# In[94]:


gbc = GradientBoostingClassifier(random_state=0)
scores_gbc = cross_val_score(gbc, new_train.drop('Survived',axis=1),new_train['Survived'] , cv=11)
accuracy_cross_gbc = np.mean(scores_gbc)
print(scores_gbc)
print(accuracy_cross_gbc)


#Hyperparameter Tuning in svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

x_train,x_test,y_train,y_test=train_test_split(new_train.drop('Survived',axis=1),new_train['Survived'],test_size=0.30,random_state=50)

svm_model=SVC(kernel='linear',random_state=0)
svm_model.fit(x_train,y_train)
y_predict=svm_model.predict(x_test)
accuracy_svm=accuracy_score(y_test,y_predict)
print(accuracy_svm)
#Apply GridSearchCV
parameter=[{'C':[1,10,100,1000],'kernel':['linear']},{'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7]}]

grid_search=GridSearchCV(estimator=svm_model,
                        param_grid=parameter,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1)
grid_search=grid_search.fit(x_train,y_train)
accuracy=grid_search.best_score_
accuracy
grid_search.best_params_
svm_model=SVC(kernel='linear',C=1000)
svm_model.fit(x_train,y_train)
y_predict=svm_model.predict(x_test)
accuracy_svm=accuracy_score(y_test,y_predict)
print(accuracy_svm)
