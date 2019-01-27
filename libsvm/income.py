import sys
# coding: utf-8

# In[31]:


columns_ = ["age","workclass","fnlwgt","education","education-num","marital-status",
           "occupation","relationship","race","sex","capital-gain",
           "capital-loss","hours-per-week","native-country","label"]

columns_1 =  ["age","workclass","fnlwgt","education","education-num","marital-status",
           "occupation","relationship","race","sex","capital-gain",
           "capital-loss","hours-per-week","native-country"]

import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing

df_train = pd.read_csv(sys.argv[1] , names =columns_)  
df_test = pd.read_csv(sys.argv[2] , names =columns_1)


df_train_c = df_train.iloc[:,:-1]
df_train_label = df_train.iloc[:,-1]



df = pd.concat([df_train_c ,df_test])




df = df.drop("relationship" , axis = 1)
df = df.drop("fnlwgt" , axis = 1)
df = df.drop("sex" , axis = 1)
df = df.drop("workclass" , axis = 1)


df_dumm = pd.get_dummies(df)
col = df_dumm.columns


x = df_dumm.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)



x = df_dumm.iloc[:32562,:] #32562
y = df_train_label


test = df_dumm.iloc[32562:,:] #16280




# In[32]:


x_va = x.values
y_va = y.values
test_va = test.values


# In[33]:


with open('income.tr', 'w') as f:
    for x, y in zip( x_va , y_va):
        f.write(str(y)+" ")
        for i in range(len(x)):
            if x[i] != 0 :
                f.write( str(i+1) + ":"+str(x[i]) + " ")
        f.write("\n")
        
with open('income.te', 'w') as f:
    for x in test_va:
        f.write(str(0)+" ")
        for i in range(len(x)):
            if x[i] != 0 :
                f.write( str(i+1) + ":"+str(x[i]) + " ")
        f.write("\n")    

