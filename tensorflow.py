#!/usr/bin/env python
# coding: utf-8

# # neuronal network

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ## data, arrays and equivalents in meters and feet

# In[2]:


meters = np.array([4.3, 5, 5.4, 5.6, 6.1, 6.9, 7.2], dtype=float)
feet = np.array([1.310640, 1.524000, 1.645920, 1.706880, 1.859280, 2.103120, 2.194560], dtype=float )


# ## keras - tensorflow

# In[3]:


layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])


# In[4]:


model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)


# ## training the model

# In[5]:


# laps = 1500, number of attemps
training = model.fit(meters, feet, epochs = 1500, verbose = False)
print("model already trained!")


# ## graphic

# In[6]:


plt.title("Correlation")
plt.xlabel("Laps",fontsize=20)
plt.ylabel("Losses",fontsize=20)
plt.plot(training.history["loss"])


# ## making a prediction

# In[7]:


result = model.predict([6.5])
print("real result => 6.5ft = 1.981200 meters")
print("model result => " + str(result) + "meters")
print("error model => " + str(result-1.98120) + "meters")


# In[ ]:





# In[ ]:




