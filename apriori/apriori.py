
# coding: utf-8

# In[301]:


import numpy as np                                          #dbscan==> 掃一筆transaction 後對c2做數量增加
import time
import sys
                                                            #而非一筆資料掃一次db (c * n)

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
#print(min_count)


# In[302]:


bigbag1 = bigbag.replace("\n" , " ")
bigbag2 = bigbag1.split(" ") #bigbig2 ==> 交易中細項總和     
bag_solo = set(bigbag2)  #bag_solo ==> 交易中出現的細項(不重複)


# In[303]:


bag_solo_int = []
bag_solo_modi = list(bag_solo)
bag_solo_modi.pop(0)

for i in list(bag_solo_modi):
    bag_solo_int.append(int(i))

bag_solo_int.sort()


# In[304]:


c1 = {}
for i in bag_solo_int:
    #i = tuple(i)
    c1[i] = 0



# In[305]:


tStart = time.time()#計時開始          #tuple ! 可以做為dict 得key

for i in transaction_np:              #不要都全拿 要啥拿啥 要key拿key
    for key in c1.keys():
        if key in i:
            c1[key] = c1[key] + 1
        




tEnd = time.time()#計時結束            
#print (tEnd - tStart)#原型長這樣         


# In[306]:


def prune(c , count): 
    items_list = list(c.items())                            # 過濾數量小於min_count的項            
    for k in items_list:
        if k[1] < count:
            del c[k[0]]
            
    return c


# In[307]:


l = prune(c1 , min_count)


# In[308]:


c2 = {}
keys_list = list(c1.keys())
for i in range(len(keys_list)):
    for j in range(i + 1 ,len(keys_list)):
        temp = []
        temp.append(keys_list[i])
        temp.append(keys_list[j])
        temp.sort()
        c2[tuple(temp)] = 0
        
        
    


# In[309]:


def dbscan(c , db):
    #tStart = time.time()#計時開始                     
    
    for key in c.keys():
        key_set = set(key)
        for i in db:
            if key_set.issubset(i):
                c[key] = c[key] + 1
    
    #tEnd = time.time()#計時結束            
    #print ("dbscan",tEnd - tStart)#原型長這樣 
    return c
    


# In[310]:


c2 = dbscan(c2 , transaction_np )


# In[311]:


l2 = prune(c2 , min_count)


# In[312]:


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
    
    
    


# In[313]:


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
        
        
        
    


# In[314]:


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
                            
        
    
    
    
    
    


# In[315]:


def apriori(l , min_count , db):
    tStart2 = time.time()#計時開始
    level = 1
    keys_list = list(l.keys())
    keys_list.sort()
    iteration = True
    newc = {}
    final = l.copy()
    
    while iteration:
        #tStart = time.time()#計時開始
        #print(level)
        #print(keys_list)
        for i in range(len(keys_list)):
            for j in range(i+1 , len(keys_list)):
                if canfindsuperornot(keys_list[i] , keys_list[j] , level):
                    target = findsupertarget(keys_list[i], keys_list[j])   #target list裡裝tuple
                    door , temp = decidesuper(keys_list[i], keys_list[j], target ,final)
                    if door :
                        if temp not in newc:
                            #listout.append(temp)  
                            newc[temp] = 0
                
                else:
                    break
                
        newc = dbscan(newc , transaction_np)
        newl = prune(newc , min_count)
        templ = newl.copy()
        keys_list = list(templ.keys())
        keys_list.sort()
        final.update(templ)
        #print(keys_list)
        #print(newl)
        #print(len(newl))
        if len(newl) == 0:
            break
        
        newc.clear()
        level = level + 1
        
        tEnd = time.time()#計時結束            
        #print (tEnd - tStart)#原型長這樣 
    tEnd2 = time.time()#計時結束            
    print (tEnd2 - tStart2)#原型長這樣     
    return final
        
        
        
                    
        
    


# In[316]:


z = apriori(l2 , min_count , transaction_np)  # 1 4.3   2 10.77   3 23   4 44   5 70   6 91   7 77   8 45   9 16   10 3.8   11----
#len(z)


# In[266]:


l.update(z)


# In[267]:


#len(l) #53337 for support = 0.2 total amount of frequent item set


# In[268]:


f = open(sys.argv[3] , 'w')

for key , value in l.items():
    temp = ""
    if isinstance(key , int):
        temp = temp+str(key)
    else:
        for i in key:
            temp = temp + str(i) + " "
            
    temp = temp + "("+str(value)+")" + "\n"
    f.write(temp)
f.close()

