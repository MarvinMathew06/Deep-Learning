#!/usr/bin/env python
# coding: utf-8

# # Import Data import and Pre-processing 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


ds=pd.read_csv("F:/Analytics/Learnings/Python/Pratice data/Social_Network_Ads.csv")
ds.head(3)


# In[3]:


# Checking null values 

ds.isnull().sum()


# In[19]:


# Checking the predicator variable

ds["Purchased"].value_counts()


# In[7]:


# Convert categorical variables to numeric

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds["Gender"]=le.fit_transform(ds["Gender"])


# # Model built

# #### Defining Target and Indepenent variables

# In[13]:


y=ds.iloc[:,-1]
x=ds.iloc[:,[1,2,3]]


# #### Standarisation 
# 
# 

# In[14]:


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
x=sc.fit_transform(x)


# #### Test Train Split

# In[15]:


from sklearn.model_selection import train_test_split  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 


# #### Defining Keras Model

# In[23]:


from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(6,input_dim=3,activation="relu"))
model.add(Dense(6,activation="relu"))
model.add(Dense(1,activation="sigmoid"))


# #### Compile the Keras model

# In[25]:


model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()


# #### Fit the keras model

# In[26]:


model.fit(x_train,y_train,epochs=50)


# # Model evaluation and testing 

# #### evaluate the training dataset

# In[29]:


#### _, accuracy = model.evaluate(x_train,y_train)
print('Accuracy: %.2f' % (accuracy*100))


# #### evaluate the test dataset

# In[30]:


_, accuracy = model.evaluate(x_test,y_test)
print('Accuracy: %.2f' % (accuracy*100))


# # Conclusion
1. For Binary Classification - Sigmoid Activation Function is used 
2. Number of hidden layers used is 6
3. Training and testing accuracy is 80%