#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
#import seaborn as sns
#import sklearn


# In[2]:


data = pd.read_csv('./data/students.csv')


# In[3]:


data


# In[4]:


data.describe()


# In[5]:


data_store = pd.read_csv('./data/store.csv')


# In[6]:


data_store.describe()


# # Regression (y = b0 + b1*x1)

# In[7]:


y = data['POINT_AVERAGE']
x1 = data['SCORE']


# In[10]:


plt.scatter(x1,y)
plt.xlabel('SCORE',fontsize=20)
plt.ylabel('POINT_AVERAGE',fontsize=20)
plt.show()


# x = sm.add_constant(x1)
# result = sm.OLS(y,x).fit()
# result.summary()

# # Regression: b0 = b0 * 1

# In[11]:


x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
result.summary()


# In[12]:


plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
plt.xlabel('SCORE', fontsize = 20)
plt.ylabel('POINT_AVERAGE', fontsize = 20)
plt.show()


# In[ ]:




