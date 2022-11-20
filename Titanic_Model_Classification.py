#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.metrics import plot_confusion_matrix, accuracy_score, classification_report, roc_curve, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# In[3]:


# filter the warnings
import warnings
warnings.filterwarnings("ignore")


# In[4]:


# Load the dataset -->read_csv()
data = pd.read_csv('titanic.csv')
data


# In[5]:


# It is like hello world for EDA in Data Science
# frst step is get the information
# data.info()


# In[6]:


# To check the count of missing values
data.isnull().sum()


# In[7]:


# First let's understand important columns information
data.columns


# In[8]:


# Data Visualization
data['Age'].plot.hist()
plt.show()


# In[9]:


sns.boxplot(x="Embarked", y="Age", data=data)
plt.show()


# In[10]:


sns.boxplot(x="Sex", y="Age", data=data)
plt.show()


# In[11]:


# Filling the age column with mean-->as majority age group is from 20-30 we fill with mean
# inplace will make permanent changes to your dataframe
data['Age'].fillna(data['Age'].mean(), inplace=True)


# In[12]:


data.isnull().sum()


# In[13]:


# we can drop column permanently
# data.dropna() #don't do this we will be loosing rows completely
data.drop(columns=['Cabin'], inplace=True)


# In[14]:


# As majority are travlling from Southampton we can fill with 'S'
data['Embarked'].fillna('S', inplace=True)


# In[15]:


data.isnull().sum()


# In[16]:


# Feature Encoding -->converting categorical data to numerical values (0,1..)
# pandas -->get_dummies()
sex = pd.get_dummies(data['Sex'], drop_first=True)
sex


# In[17]:


pclass = pd.get_dummies(data['Pclass'], drop_first=True)
pclass


# In[18]:


embarked = pd.get_dummies(data['Embarked'], drop_first=True)
embarked


# In[19]:


# combining all the datafrmes
final_data = pd.concat([data, sex, pclass, embarked], axis='columns')
final_data


# In[20]:


# finally we will drop all unnecessary columns
final_data.drop(columns=['PassengerId', 'Name', 'Pclass',
                         'Sex', 'Embarked', 'Ticket'], inplace=True)


# In[21]:


final_data


# In[22]:


final_data['Survived'].value_counts()


# In[23]:


# Training and Testing data
X = final_data.drop('Survived', axis=1)
y = final_data['Survived']


# In[25]:


# Splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)


# In[26]:


# In[27]:


logmodel = LogisticRegression()


# In[28]:


logmodel.fit(X_train, y_train)  # Estimators


# In[29]:


# predictors
predictions = logmodel.predict(X_test)


# In[30]:


# Classification metrics


# In[31]:


# dir(sklearn.metrics)


# In[33]:


print(accuracy_score(predictions, y_test)*100)


# In[34]:


print(confusion_matrix(predictions, y_test))


# In[36]:


print((132+77)/(132+21+38+77))  # same as accuracy


# In[37]:


print(classification_report(predictions, y_test))


# In[38]:


X_test


# In[39]:


new_predictions = logmodel.predict([[22.0, 1, 0, 7.25, 1, 0, 1, 0, 1]])
new_predictions


# In[40]:


# plot_confusion_matrix
plot_confusion_matrix(logmodel, X_test, y_test)
plt.show()


# In[41]:


# define the metrics


# In[42]:


y_pred_proba = logmodel.predict_proba(X_test)[::, 1]
# y_pred_proba
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# create the ROC Curve
plt.plot(fpr, tpr)
plt.xlabel("True Positive Rate")
plt.ylabel('False Postive Rate')
plt.show()


# In[43]:


y_pred_proba = logmodel.predict_proba(X_test)[::, 1]
# y_pred_proba
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
# create the ROC Curve
plt.plot(fpr, tpr, label="AUC="+str(auc))
plt.xlabel("True Positive Rate")
plt.ylabel('False Postive Rate')
plt.legend()
plt.show()


# In[ ]:

def survive(arr):
    predictions = logmodel.predict(arr)
    return predictions
