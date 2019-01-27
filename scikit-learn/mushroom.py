
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv, math, sys, os, time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree


# In[2]:



x_origi = []

with open(sys.argv[2], 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        x_origi.append(row)
x_origi = np.array(x_origi)            #(6500, 23)
y = x_origi[: , 22:]
x = x_origi[:,:-1]


# In[3]:


temp = []
with open(sys.argv[3], 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        temp.append(row)
temp = np.array(temp)
x_test = temp[:,:-1]                 #(1624, 22)
y_test = temp[: , 22:]


# In[4]:


columns_ = ["cap-shape","cap-surface","cap-color","bruises?","odor","gill-attachment",
           "gill-spacing","gill-size","gill-color","stalk-shape","stalk-root",
           "stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring",
           "stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type",
          "spore-print-color","population","habitat"]


# In[5]:


x_total = np.concatenate((x, x_test), axis=0)


# In[6]:


df = pd.DataFrame(x_total , columns = columns_)


# In[7]:


df_dumm = pd.get_dummies(df)


# In[8]:


df_dumm_ = df_dumm.values


# In[9]:


x = df_dumm_ [:6500]
x_test = df_dumm_[6500:]


# In[10]:


#-----------decision tree


# In[11]:


if sys.argv[1] == "D":
    clf = DecisionTreeClassifier()                               
    clf.fit(x,y)   
    y_pred = clf.predict(x_test)

    print("Accuracy on training set: %.3f" %clf.score(x, y))
    print("Accuracy on testing set: %.3f" %accuracy_score(y_test , y_pred))
    with open(sys.argv[4], 'w' , newline ="") as out:
        writer = csv.writer(out, delimiter=',')
        for element in y_pred:
            writer.writerow([str(element)])


# In[12]:


#------------Naive Bayes


# In[13]:


if sys.argv[1] == "N":
    nb = MultinomialNB(alpha = 0.001)
    nb.fit(x,y)
    y_pre_nb = nb.predict(x_test)
    print("Accuracy on training set: %.3f" %nb.score(x, y))
    print("Accuracy on testing set: %.3f" %accuracy_score(y_test , y_pre_nb))
    with open(sys.argv[4], 'w' , newline ="") as out:
        writer = csv.writer(out, delimiter=',')
        for element in y_pre_nb:
            writer.writerow([str(element)])


# In[15]:


tree.export_graphviz(clf,out_file='tree.dot')   


# In[16]:


#---------------其他naive bayes


# In[17]:


#from sklearn.naive_bayes import GaussianNB
#nb_gu = GaussianNB()
#nb_gu.fit(x,y)
#y_pre = nb_gu.predict(x_test)
#print("Accuracy on training set: %.3f" %nb_gu.score(x, y))
#print("Accuracy on testing set: %.3f" %accuracy_score(y_test , y_pre))


# In[18]:


#from sklearn.naive_bayes import BernoulliNB
#nb_br = BernoulliNB()
#nb_br.fit(x,y)
#y_pre = nb_br.predict(x_test)
#print("Accuracy on training set: %.3f" %nb_br.score(x, y))
#print("Accuracy on testing set: %.3f" %accuracy_score(y_test , y_pre))

