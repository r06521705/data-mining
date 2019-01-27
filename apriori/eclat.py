
# coding: utf-8

# In[1]:


import numpy as np                                          #dbscan==> 掃一筆transaction 後對c2做數量增加
import time
import sys                                                  #而非一筆資料掃一次db (c * n)

min_sup = float(sys.argv[2])


with open(sys.argv[1], 'r') as f:
    line = f.readlines()
    
with open(sys.argv[1], 'r') as g:
    bigbag = g.read()
    
    
    

transaction = []
for i in line:
    box=[]
    temp = i.replace('\n','').split(" ")
    for j in temp:
        box.append(int(j))
    transaction.append(set(box))
        
    #transaction.append(set(i.replace('\n','').split(" ")))
#8416,23(193568)
transaction_np = np.array( transaction)
#transaction_np.shape
#print(transaction_np.shape) 8416,23

min_count = int(len(transaction_np) * min_sup)
#print(transaction_np)


# In[2]:


bigbag1 = bigbag.replace("\n" , " ")
bigbag2 = bigbag1.split(" ") #bigbig2 ==> 交易中細項總和     
bag_solo = set(bigbag2)  #bag_solo ==> 交易中出現的細項(不重複)


# In[3]:


bag_solo_int = []
bag_solo_modi = list(bag_solo)
bag_solo_modi.sort()
bag_solo_modi.pop(0)



for i in list(bag_solo_modi):
    bag_solo_int.append(int(i))

bag_solo_int.sort()


# In[4]:


# 先做一個用編號當key的transaction dict
trans_dict = {}
for i in range(len(transaction_np)):
    trans_dict[i+1] = transaction_np[i]


# In[5]:


c1 = {}                                            #c1 key ==> 
for i in bag_solo_int:
    temp = []
    for key,value in trans_dict.items():
        if i in value:
            temp.append(key)
    temp = set(temp)
    c1[i] = temp
        


# In[6]:


def prune(c , count):
    items_list = list(c.items())                            # 過濾數量小於min_count的項            
    for k in items_list:
        if len(k[1]) < count:
            del c[k[0]]
            
    return c


# In[7]:


l1 = prune(c1 , min_count)


# In[8]:


items_list = list(l1.items())
a = items_list[0][1]
b = items_list[2][1]

#cross = items_list[0][1].intersection(items_list[1][1])

#print(a&b)


# In[9]:


c2 = {}
items_list = list(l1.items())
for i in range(len(items_list)):
    for j in range(i + 1 ,len(items_list)):
        temp = []
        temp.append(items_list[i][0])
        temp.append(items_list[j][0])
        temp.sort()
        temp = tuple(temp)
        cross = items_list[i][1].intersection(items_list[j][1])
        c2[temp] = cross



# In[10]:


l2 = prune(c2 , min_count)   #c 861  l 370


# In[11]:


#l2 正確


# In[12]:


def canfindsuperornot(first , second , level):
    
    first = list(first)
    second = list(second)
    count = 0

    for i,j in zip(first , second):
        if i==j :
            count = count + 1
    
    if count == level :
        return True
    else:
        return False
    
    
    


# In[13]:


def findsupertarget(first , second ):
    target = []
    
    s1 = set(first)
    s2 = set(second)  
    samepart = s1.intersection(s2)
    diffpart = s1.symmetric_difference(s2)

    samepart = list(samepart)
    diffpart = list(diffpart)
    
    for i in range(len(samepart)):
        temp = samepart.copy()                     #兩個list的copy不能直接用等於"=" ,這樣只是一個list有兩個reference指向同一個地方
        temp.pop(i)
        temp.extend(diffpart)
        temp.sort()
        target.append(tuple(temp))
    
    return target
        
        
        
    


# In[14]:


def decidesuper(first , second ,needtofind, l):
    
    vote = []
    first = set(first)
    seconf = set(second)
    superone = list(first.union(second))
    superone.sort()
    superone = tuple(superone)
    
    for i in needtofind:
        judge = 1
        if i in l:
            judge = judge * 0
        else:
            judge = judge * 1
        
        if judge == 0 :
            vote.append(1)
        else:
            vote.append(0)  
            break
    
    decide = 1
    for i in vote:
        decide = decide * i
    

     
    if decide == 1 :
        return (True, superone)
    else:
        return (False , [])
                            
        
    
    
    
    
    


# In[15]:


def apriori(l , min_count ):
    tStart2 = time.time()#計時開始
    level = 1
    items_list = list(l.items())
    items_list.sort()  # watch
    iteration = True
    newc = {}
    final = l.copy()
    
    while iteration:
        tStart = time.time()#計時開始
        #print(level)
        #print(keys_list)
        for i  in range(len(items_list)):
            for j in range(i+1 , len(items_list)):
                if canfindsuperornot(items_list[i][0] , items_list[j][0] , level):
                    target = findsupertarget(items_list[i][0], items_list[j][0])   #target list裡裝tuple
                    door , temp = decidesuper(items_list[i][0], items_list[j][0], target ,final)
                    if door :
                        if temp not in newc:
                            #listout.append(temp)  
                            #newc[temp] = 0
                            cross = items_list[i][1].intersection(items_list[j][1])
                            newc[temp] = cross
                
                else:
                    break
                
        #newc = dbscan(newc , transaction_np)
        newl = prune(newc , min_count)

        items_list = list(newl.items())
        items_list.sort()
        final.update(newl)
        #print(keys_list)
        #print(newl)
        #print(len(newl))
        if len(newl) == 0:
            break
        newl.clear()
        newc.clear()
        level = level + 1
        
        tEnd = time.time()#計時結束            
        #print (tEnd - tStart)#原型長這樣 
    tEnd2 = time.time()#計時結束            
    #print (tEnd2 - tStart2)#原型長這樣     
    return final
        
        
        
                    
        
    


# In[16]:


z = apriori(l2 , min_count )


# In[53]:


l1.update(z)


# In[54]:


f = open(sys.argv[3] , 'w')

for key , value in l1.items():
    temp = ""
    if isinstance(key , int):
        temp = temp+str(key)
    else:
        for i in key:
            temp = temp + str(i) + " "
            
    temp = temp + "("+str(len(value))+")" + "\n"
    f.write(temp)
f.close()

