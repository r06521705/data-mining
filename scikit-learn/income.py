
# coding: utf-8

# In[1]:


columns_ = ["age","workclass","fnlwgt","education","education-num","marital-status",
           "occupation","relationship","race","sex","capital-gain",
           "capital-loss","hours-per-week","native-country","label"]


# In[2]:


columns_1 =  ["age","workclass","fnlwgt","education","education-num","marital-status",
           "occupation","relationship","race","sex","capital-gain",
           "capital-loss","hours-per-week","native-country"]


# In[3]:


import numpy as np
import pandas as pd
import csv, math, sys, os, time
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[4]:


df_train = pd.read_csv(sys.argv[2] , names =columns_)  
df_test = pd.read_csv(sys.argv[3], names =columns_1)


# In[5]:


df_train_c = df_train.iloc[:,:-1]
df_train_label = df_train.iloc[:,-1]


# In[6]:


df = pd.concat([df_train_c ,df_test])
df = df.drop("age", axis = 1)
df = df.drop("education-num", axis = 1)    #0.797
df = df.drop("relationship", axis = 1)    #0.824
df = df.drop("race", axis = 1)           #0.796
df = df.drop("native-country", axis = 1) #0.796
df = df.drop("workclass", axis = 1) 


# In[7]:


df_dumm = pd.get_dummies(df)
col = df_dumm.columns


# In[8]:


x = df_dumm.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)


# In[9]:


df.columns = col #(48842, 44)


# In[11]:


x = df.iloc[:32562,:] #32562
y = df_train_label
test = df.iloc[32562:,:] #16280


# In[13]:


#-------decision tree


# In[14]:


#x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)


# In[16]:


clf = DecisionTreeClassifier(max_depth = 9, random_state = 42)                               
clf.fit(x,y)
y_pre = clf.predict(test)
print("Accuracy on training set: %.3f" %clf.score(x, y))
with open(sys.argv[4], 'w' , newline ="") as out:
    writer = csv.writer(out, delimiter=',')
    for element in y_pre:
        writer.writerow([str(element)])


# In[ ]:


#-------naive base


# In[ ]:


nb = MultinomialNB()
nb.fit(x,y)
print("Accuracy on training set: %.3f" %nb.score(x, y))


# In[ ]:


#---------------其他naive bayes


# In[ ]:


#from sklearn.naive_bayes import GaussianNB
#nb_gu = GaussianNB()
#nb_gu.fit(x_train,y_train)

#print("Accuracy on training set: %.3f" %nb_gu.score(x_train, y_train))
#print("Accuracy on testing set: %.3f" %nb_gu.score(x_test , y_test))


# In[ ]:


#from sklearn.naive_bayes import BernoulliNB
#nb_br = BernoulliNB()
#nb_br.fit(x_train,y_train)

#print("Accuracy on training set: %.3f" %nb_br .score(x_train, y_train))
#print("Accuracy on testing set: %.3f" %nb_br .score(x_test , y_test))

