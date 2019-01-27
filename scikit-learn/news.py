
# coding: utf-8

# In[1]:


import numpy as np
import csv, math, sys, os, time
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


x = []

with open(sys.argv[2], 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        a = row
        b = list(map(float,a))
        x.append(b)



# In[2]:


temp = []
with open(sys.argv[3], 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        a = row
        b = list(map(float,a))
        temp.append(b)



# In[3]:


temp = np.array(temp)
x_test = temp[:,:-1]
y_test = temp[: , 23909:]


# In[4]:


temp.shape


# In[5]:


def test_accuracy(predict , test):
    right = 0
    for i in range(len(predict)):
        if predict[i] == test[i]:
            right = right + 1
    ans = right / len(predict)
    return ans


# In[6]:


x = np.array(x)
x.shape


# In[7]:


y = x[: , 23909:]


# In[8]:


x = x[:,:-1]


# In[9]:


#-----------decision tree


# In[10]:


if sys.argv[1] == "D":
    clf = DecisionTreeClassifier(max_depth = 50, random_state = 42)                               
    clf.fit(x,y) 
    y_pre = clf.predict(x_test)

    print("Accuracy on training set: %.3f" %clf.score(x, y))
    print("Accuracy on testing set: %.3f" %accuracy_score(y_test , y_pre))   

    with open(sys.argv[4], 'w' , newline ="") as out:
        writer = csv.writer(out, delimiter=',')
        for element in y_pre:
            writer.writerow([str(element)])


# In[11]:


#------------Naive Bayes


# In[12]:


if sys.argv[1] == "N":

    nb = MultinomialNB(alpha = 0.1)
    nb.fit(x,y)
    y_pre_nb = nb.predict(x_test)
    print("Accuracy on training set: %.3f" %nb.score(x, y))
    print("Accuracy on testing set: %.3f" %accuracy_score(y_test , y_pre_nb))

    with open(sys.argv[4], 'w' , newline ="") as out:
        writer = csv.writer(out, delimiter=',')
        for element in y_pre_nb:
            writer.writerow([str(element)])


# In[13]:


#---------------其他naive bayes


# In[14]:


#from sklearn.naive_bayes import GaussianNB
#nb_gu = GaussianNB()
#nb_gu.fit(x,y)
#y_pre = nb_gu.predict(x_test)
#print("Accuracy on training set: %.3f" %nb_gu.score(x, y))
#print("Accuracy on testing set: %.3f" %accuracy_score(y_test , y_pre))


# In[15]:


#from sklearn.naive_bayes import BernoulliNB
#nb_br = BernoulliNB()
#nb_br.fit(x,y)
#y_pre = nb_br.predict(x_test)
#print("Accuracy on training set: %.3f" %nb_br.score(x, y))
#print("Accuracy on testing set: %.3f" %accuracy_score(y_test , y_pre))

