import sys
# coding: utf-8

# In[85]:


import pandas as pd
import numpy as np


# In[86]:


col = ["sex","length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]


# In[87]:


train =pd.read_csv(sys.argv[1], names = col)
test  =pd.read_csv(sys.argv[2], names = col) 


# In[88]:


train["Rings"] = train["Rings"].map({3:0, 1:1 ,2:2})
test["Rings"] = test["Rings"].map({3:0, 1:1 ,2:2})


# In[89]:


train_x = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test_x = test.iloc[:,:-1]
test_y = test.iloc[:,-1]


# In[105]:


train_x_du = pd.get_dummies(train_x)
test_x_du = pd.get_dummies(test_x)
train_x_value = train_x_du.values
test_x_value = test_x_du.values


# In[120]:


trainbox = []
testbox = []
num_of_att = 10
with open('abalone.tr', 'w') as f:
    for x, y in zip( train_x_value , train_y):
        f.write(str(y)+" ")
        for i in range(len(x)):
            if x[i] != 0 :
                f.write( str(i+1) + ":"+str(x[i]) + " ")
        f.write("\n")
        
with open('abalone.te', 'w') as f:
    for x, y in zip( test_x_value , test_y):
        f.write(str(y)+" ")
        for i in range(len(x)):
            if x[i] != 0 :
                f.write( str(i+1) + ":"+str(x[i]) + " ")
        f.write("\n")    



