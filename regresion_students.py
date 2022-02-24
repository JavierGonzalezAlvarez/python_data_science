#!/usr/bin/env python
# coding: utf-8

# # supervised algorithm

# In[1]:


import numpy as np
import pandas as pd
#import statsmodels.api as sm
import matplotlib.pyplot as plt
#import sklearn as sl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


data_all = pd.read_csv( './data/students/student-mat.csv' , sep = ';' )


# In[3]:


data_all


# # select features relevant for the study

# In[4]:


data = data_all[['age','sex','studytime','absences','G1','G2','G3']]


# # g1, g2 and g3 are the three grades that the students have

# In[5]:


data


# # select some variables

# In[6]:


df = data_all[['age','studytime']]


# In[7]:


df


# ## df to array

# In[8]:


array = np.array(df.to_numpy())
array


# ## do we need indexes?

# In[9]:


#df1 = pd.DataFrame(array, index=['object_1','object_2', ... 'object_396'], columns=['age','studytime'])
#df1
#df1.loc['studytime']


# # convert variable sex into number

# In[10]:


data[ 'sex' ] = data[ 'sex' ].map({ 'F' : 0 , 'M' : 1 })


# In[11]:


data['sex']


# In[12]:


prediction = 'G3'


# # preparing data
# ## sklearn don't accept pandas dataframes just numpy arrays

# In[13]:


X = np.array(data.drop([prediction], 1 ))
Y = np.array(data[prediction])


# In[14]:


## all columns
X


# In[15]:


## final grade (G3)
Y


# # training and testing
# X_train, X_test, Y_train, Y_test = train_test_split(X,
# Y, test_size = 0.1 )

# ## create a model

# In[16]:


model = LinearRegression()


# ## trainign data witg fit function

# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1 )
model.fit(X_train, Y_train)


# ## how well our model performs
# ## use score. we will have different results on each run

# In[18]:


accuracy = model.score(X_test, Y_test)
accuracy


# ## if score > 75%, it's a good data, then we can enter new data and predict the final grade (G3)

# In[19]:


## variables to input data => age, sex, studytime, absences, G1, G2
X_new = np.array([[ 18 , 1 , 3 , 40 , 15 , 16 ]])
Y_new = model.predict(X_new)


# In[20]:


print('the final grade will be:', Y_new)


# # visualizing correlations

# ## use scatter

# In[21]:


plt.scatter(data[ 'studytime' ], data[ 'G3' ])
plt.title("Correlation")
plt.xlabel('Study Time',fontsize=20)
plt.ylabel('Final Grade 3',fontsize=20)
plt.show()


# In[22]:


plt.scatter(data[ 'G2' ], data[ 'G3' ])
plt.title('Correlation')
plt.xlabel('Second Grade')
plt.ylabel('Final Grade')
plt.show()


# # classification
# ## decision trees

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer


# In[24]:


data = load_breast_cancer()


# In[25]:


data


# In[26]:


print (data.target_names)


# In[27]:


print (data.feature_names)


# ## preparing data

# In[28]:


X = np.array(data.data)
Y = np.array(data.target)


# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
test_size = 0.1 )


# ## training and testing - define K-Nearest-Neighbors classifier

# In[30]:


knn = KNeighborsClassifier( n_neighbors = 5 )
knn.fit(X_train, Y_train)


# # test the model

# In[31]:


accuracy = knn.score(X_test, Y_test)
print ("accuracy of the model: ", accuracy)


# # select an algorithm

# In[32]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ## create 5 classifiers to train and test algorithm , which one scores better?

# In[33]:


clf1 = KNeighborsClassifier( n_neighbors = 5 )
clf2 = GaussianNB()
clf3 = LogisticRegression()
clf4 = DecisionTreeClassifier()
clf5 = RandomForestClassifier()
clf1.fit(X_train, Y_train)
clf2.fit(X_train, Y_train)
clf3.fit(X_train, Y_train)
clf4.fit(X_train, Y_train)
clf5.fit(X_train, Y_train)
print (clf1.score(X_test, Y_test))
print (clf2.score(X_test, Y_test))
print (clf3.score(X_test, Y_test))
print (clf4.score(X_test, Y_test))
print (clf5.score(X_test, Y_test))


# In[ ]:




