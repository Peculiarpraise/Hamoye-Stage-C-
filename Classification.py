#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import seaborn as sns 
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score 


# In[33]:


electricity = pd.read_csv ("Data_for_UCI_named.csv")
electricity.head(10)


# In[34]:


electricity.describe()


# In[35]:


electricity.info()


# In[36]:


electricity = electricity.dropna()
electricity.isna().sum()


# In[37]:


x = electricity['tau1']
y = electricity['stabf']
 


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8 ,test_size = 0.2 ,random_state = 1 )
y_train.value_counts()


# In[48]:


log_reg = LogisticRegression()
log_reg.fit(normalised_train, y_balanced)


# In[49]:


scores = cross_val_score(log_reg, normalised_train_df, y_balanced, cv= 5 , scoring= 'f1_macro' )
scores


# In[ ]:




