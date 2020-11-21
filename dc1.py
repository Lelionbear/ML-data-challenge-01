#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv('aggregated.csv', nrows=1000)# 5 129 354
#df = pd.read_csv('aggregated.csv')


# In[2]:


date = df['FL_DATE'].str.split('-', n = 2, expand = True).astype('float64')
df['YEAR'] = date[0]
df['DAY'] = date[2]


################################################################
# i = 0
# for index, row in df.iterrows():
#     temp = row.DAY.split("-")
#     day = temp[2]
#     year = temp[0]
#     df.iloc[i, df.columns.get_loc('DAY')] = float(day)
#     df.iloc[i, df.columns.get_loc('YEAR')] = float(year)
#     i += 1

# df.YEAR = df.YEAR.astype('float64')
# df.DAY = df.DAY.astype('float64')
################################################################

df


# In[3]:


df.columns


# # GOAL: create a model to predict flight delays 
# * ARR_DEL15 boolean target

# ## DROP any values that are not going to be used then encode the remaining values into numbers
# 
# ---ELIMINATE--
# 
# Unnamed: 13
# 
# FL_DATE
# 
# FL_NUM
# 
# ORIGIN_CITY_NAME
# 
# DEST_CITY_NAME
# 
# 
# ---MAYBE---
# 
# UNIQUE_CARRIER	
# 
# CRS_DEP_TIME

# In[4]:


df = df.replace([np.inf, -np.inf], np.nan)

df = df.drop(["Unnamed: 13","FL_DATE", "FL_NUM", "ORIGIN_CITY_NAME", "DEST_CITY_NAME"], axis=1).dropna()

df = df.drop(["UNIQUE_CARRIER", "CRS_DEP_TIME"], axis=1).dropna()         # tweaking
# df = df.drop(["YEAR","MONTH","DAY", "CRS_ELAPSED_TIME"], axis=1).dropna() # tweaking


target = df.ARR_DEL15
df = df.drop(["ARR_DEL15"], axis=1)


# ## ENCODE columns with object values into numbers

# In[5]:


df.info()


# In[6]:


def encode_labels(labels):
    from sklearn import preprocessing
    
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le


# In[7]:


# carrier_encoder = encode_labels(df.UNIQUE_CARRIER)
# df.UNIQUE_CARRIER = carrier_encoder.transform(df.UNIQUE_CARRIER).astype('float64')

origin_encoder = encode_labels(df.ORIGIN)
df.ORIGIN = origin_encoder.transform(df.ORIGIN).astype('float64')

dest_encoder = encode_labels(df.DEST)
df.DEST = dest_encoder.transform(df.DEST).astype('float64')


# In[8]:


df.info()


# 
# ## TRAIN/TEST DATA

# In[18]:


from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y= train_test_split(df, target, test_size=0.15, random_state=42)

# print ("train_x: ", train_x)
# print ("train_y: ", train_y)
# print('\n===========================================================\n')
# print("test_x: ", test_x)
# print ("test_y: ", test_y)

# np.any(np.isnan(df))
# np.all(np.isfinite(df))


# # LogisticRegression

# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

import seaborn as sns

C = [0.0001, 0.001, 0.01, 0.1, 1]

# for c in C:
#     clf = LogisticRegression(solver='lbfgs', penalty='l2', C=c, max_iter=1000)
#     clf.fit(train_x, train_y)
#     hyp = clf.predict(test_x)
    
#     print("SCORE: ",c,"\t", accuracy_score(test_y, hyp))

    
    
clf = LogisticRegression(solver='lbfgs', penalty='l2', C=C[0], max_iter=1000)
clf.fit(train_x, train_y)
hyp = clf.predict(test_x)

## Determine performance
ac = accuracy_score(test_y, hyp)
matrix = confusion_matrix(test_y, hyp)


print("\n\n")
print("ACCURACY SCORE: ", ac)
print("\n\n")
print("CONFUSION MATRIX: ")
sns.heatmap(matrix, fmt='.5g', annot=True)


# # KNN

# In[19]:


from sklearn.neighbors import KNeighborsClassifier

for k in range(5,11):
    
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(train_x, train_y)
    hyp = clf.predict(test_x)

    print("ACCURACY SCORE: ",k,'\t', accuracy_score(test_y, hyp))
# print("\n\n")
# print ('Confusion Matrix:\n', )
# cfm = confusion_matrix(test_y, hyp)
# sns.heatmap(cfm, fmt='.5g', annot=True)


# # LogisticRegression

# In[13]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr = lr.fit(train_x,train_y)
pred = lr.predict(test_x)
print("Logistic regression")
print("Accuracy: ",accuracy_score(test_y,pred))
print ('Confusion Matrix:\n', confusion_matrix(test_y, pred))


# # SVC

# In[14]:


# from sklearn.svm import SVC
# svclassifier = SVC(kernel='linear')
# svclassifier = svclassifier.fit(train_x, train_y)
# pred = svclassifier.predict(test_x)
# print("SVM")
# print("Accuracy: ",accuracy_score(test_y,pred))
# print ('Confusion Matrix:\n', confusion_matrix(test_y, pred))


# # GaussianNB

# In[15]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb = nb.fit(train_x, train_y)
pred = nb.predict(test_x)
print("nb")
print("Accuracy: ",accuracy_score(test_y,pred))
print ('Confusion Matrix:\n', confusion_matrix(test_y, pred))


# # DecisionTreeClassifier

# In[16]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc = dtc.fit(train_x,train_y)
pred = dtc.predict(test_x)
print("Decision tree with no parameter training")
print("Accuracy: ",accuracy_score(test_y,pred))
print ('Confusion Matrix:\n', confusion_matrix(test_y, pred))


# # RandomForestClassifier

# In[17]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(train_x,train_y)
pred = rfc.predict(test_x)
print("Random forest with no parameter training")
print("Accuracy: ",accuracy_score(test_y,pred))
print ('Confusion Matrix:\n', confusion_matrix(test_y, pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




