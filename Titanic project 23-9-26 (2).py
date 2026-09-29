#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib


# In[2]:


titanic=pd.read_csv("Titanic-Dataset (1).csv")


# In[3]:


titanic


# In[4]:


titanic.info()


# In[5]:


titanic.describe()


# In[6]:


titanic.isnull().sum()


# In[7]:


titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())


# In[8]:


titanic.isnull().sum()


# In[9]:


titanic['Cabin']=titanic['Cabin'].fillna(titanic['Cabin'].mode()[0])


# In[10]:


titanic.isnull().sum()


# In[11]:


titanic['Embarked']=titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])


# In[12]:


titanic.isnull().sum()


# In[13]:


titanic.nunique()


# In[14]:


titanic['Survived'].unique()


# In[15]:


titanic['Pclass'].unique()


# In[16]:


titanic['Sex'].unique()


# In[17]:


titanic['SibSp'].unique()


# In[18]:


titanic['Parch'].unique()


# In[19]:


titanic['Embarked'].unique()


# In[20]:


titanic['Survived'].value_counts()


# In[21]:


titanic['Pclass'].value_counts()


# In[22]:


titanic['Sex'].value_counts()


# In[23]:


titanic['SibSp'].value_counts()


# In[24]:


titanic['Parch'].value_counts()


# In[25]:


a=titanic['Survived'].value_counts()


# In[26]:


a.plot(kind="bar")


# In[27]:


b=titanic['Pclass'].value_counts()


# In[28]:


b.plot(kind="bar")


# In[29]:


c=titanic['Sex'].value_counts()


# In[30]:


c.plot(kind="bar")


# In[31]:


d=titanic['SibSp'].value_counts()
d.plot(kind="bar")


# In[32]:


e=titanic['Parch'].value_counts()
e.plot(kind="bar")


# In[33]:


titanic['Embarked'].value_counts()


# In[34]:


f=titanic['Embarked'].value_counts()
f.plot(kind="bar")


# In[35]:


survival_gender = titanic.groupby('Sex')['Survived'].mean()
print(survival_gender)


# In[36]:


titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1


# In[37]:


titanic


# In[38]:


titanic['IsAlone'] = (titanic['FamilySize'] == 1).astype(int)


# In[39]:


titanic


# In[40]:


onehot = pd.get_dummies(titanic,columns=['Sex','Embarked'], dtype=int)


# In[41]:


onehot


# In[42]:


onehot = onehot.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)


# In[43]:


onehot


# In[44]:


X = onehot.drop('Survived', axis=1)
y = onehot['Survived']


# In[45]:


X


# In[46]:


y


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
)


# In[48]:


print(X_train.shape)


# In[49]:


print(X_test.shape)


# In[50]:


print(y_train.shape)


# In[51]:


print(y_test.shape)


# In[52]:


model=LogisticRegression()


# In[53]:


model.fit(X_train,y_train)


# In[54]:


y_pred=model.predict(X_test)


# In[55]:


y_pred


# In[56]:


y_test.values


# In[57]:


print(X_train.columns)


# In[58]:


new_passenger = pd.DataFrame({
    'Pclass': [3],
    'Age': [22],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25],
    'Sex_female': [1],
    'Sex_male': [0],
    'Embarked_C': [0],
    'Embarked_Q': [0],
    'Embarked_S': [1],
    'FamilySize': [2],
    'IsAlone': [0]
})


# In[59]:


new_passenger = new_passenger[model.feature_names_in_]


# In[60]:


prediction = model.predict(new_passenger)
print(prediction)
if prediction== 1:
    print("Passenger is predicted to survive")
else:
    print("Passenger is predicted not to survive")


# In[63]:


model_svm = SVC()
model_svm.fit(X_train, y_train)


# In[64]:


prediction_svm = model_svm.predict(new_passenger)
print(prediction_svm)


# In[65]:


if prediction_svm[0] == 1:
    print("SVM: Passenger is predicted to survive")
else:
    print("SVM: Passenger is predicted not to survive")


# In[66]:


joblib.dump(model_svm,"SVM.pkl")


# In[67]:


model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)


# In[68]:


prediction_knn = model_knn.predict(new_passenger)
print(prediction_knn)


# In[69]:


if prediction_knn[0] == 1:
    print("KNN: Passenger is predicted to survive")
else:
    print("KNN: Passenger is predicted not to survive")


# In[70]:


joblib.dump(model_knn,"KNN.pkl")


# In[ ]:




