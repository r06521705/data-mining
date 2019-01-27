
# coding: utf-8

# In[2]:


import numpy as np
import sys
import time
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpa
from pycuda.compiler import SourceModule
import sys


BLOCK_NUM = 16
THREAD_NUM = 16


# In[3]:


Bit32Table = [-2147483648, 1073741824, 536870912, 268435456,       #-2147483648 watch out  
    134217728, 67108864, 33554432, 16777216,
    8388608, 4194304, 2097152, 1048576,
    524288, 262144, 131072, 65536,
    32768, 16384, 8192, 4096,
    2048, 1024, 512, 256,
    128, 64, 32, 16,
    8, 4, 2, 1]
Bit32Table = np.array(Bit32Table).astype(np.int32)                #watch out
#用於之後bitvector中，利用 or 指令創建對應的integer來表示候選人在transaction中的出現情況


# In[4]:



class Eclass():
    def __init__(self, items = [], parents = [] ):
        self.items = items
        self.parents = parents
        
class itemB():
    def __init__(self, Id, bitVector, support ):
        self.Id = Id
        self.bitVector = bitVector
        self.support = support


# In[6]:


mod = SourceModule("""
__device__  int NumberOfSetBits(int i)
{
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}                   

__global__ void intersection(int *a,int *b,int *c,int length,int *support)
{
     __shared__ int result[16];
    
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int temp = 0;
    while (tid < length) {
         c[tid]=a[tid] & b[tid];
         temp += NumberOfSetBits(c[tid]);
         tid += blockDim.x * gridDim.x;  
     }

    result[threadIdx.x] = temp;
    __syncthreads();



    if (threadIdx.x == 0)
        for(int i = 1 ; i < blockDim.x ; i++){
            result[0] += result[i];
        }
    
    
    
    if (threadIdx.x == 0)
        support[blockIdx.x] = result[0];
}
"""
)

intersection = mod.get_function("intersection")


# In[7]:


def prune(c1, count):
    L1 = c1.copy()
    for key in list(L1.keys()):
        if len(L1[key]) < count:
            del L1[key]
    return L1  


# In[8]:


index = []
def Bitvector(total_num , l , box):
    bitlength = total_num + 32 -(total_num % 32) #將transaction的數目湊成32的倍數以利後續作出integer
    key_num = len(list(l.keys()))
    key_list = list(l.keys())#把dict type 轉成 list 以便之後做sort
    key_list.sort()
    for i in range(key_num):
        key_name = key_list[i]
        item = l[key_name]
        index.append(key_name) #之後 Eclass ==> itemb ==> 裡面的id存的是索引值，要再去index找對應的key名稱
        bitvector = [0] * int(bitlength / 32) #裡面應該有16890個integer
        bitvector = np.array(bitvector).astype(np.int32) #要轉成32位元for後續cuda使用
        for j in item:
            bitvector[int(j/32)] |= Bit32Table[j%32]
        box.items.append(itemB(i , bitvector , len(item)))# 在哪個transaction裡面包含過，現在是由bitvector裏頭的integer以32bit觀點來呈現


# In[9]:


def eclat_cpu(box , count , length , f ):
    size = len(box.items)      #看有幾個候選人(items)
    for i in range(size):
        child = Eclass([],[]) #要給一個新的list 不然會重複使用到相同地址的list
        child.parents = box.parents.copy() #不能直接等於rrrrrr 不同pointer 指向同一個記憶體位置而已
        child.parents.append(box.items[i].Id)
        s1 = box.items[i].bitVector
        for j in range(i+1 , size):  #itemb 和下一個itemb要比較
            s_conclu = [0] * length
            s2 = box.items[j].bitVector #兩個bitvector 要做intersection 後續平行處理能夠實施在這個地方
            support = 0
            for k in range(length):
                s_conclu[k] = s1[k] & s2[k]
                support = support + bin(s_conclu[k] & 0xffffffff).count('1')  # & | 這些運算子都是去看2進位
            if support >= count :                                      # bin ==>  轉成2進位   &0xffffffff 把bin回傳的"字串"中的負號削掉，最後在count看有幾個1
                child.items.append(itemB(box.items[j].Id, s_conclu , support))
            else:
                del s_conclu
        if len(child.items) != 0:
            eclat_cpu(child, count , length , f)                      #每次iterate出來的output會成為下一次的input，進入疊代後，會在呼叫一個children來儲存可能出現的下一層候選人
        for item in child.items:
            del item                                                  #當children找不到任何後代可能產生的候選集之後，停止疊代
        del child                                                     #depth-first search!!
    
    for item in box.items:
        for parent in box.parents:
            f.write(str(index[parent]) + ' ')
        f.write(str(index[item.Id]) + " ("+str(item.support)+")" + '\n')
            


# In[10]:


def eclat_gpu(box , count , length , f ):
    size = len(box.items)      #看有幾個候選人(items)
    for i in range(size):
        child = Eclass([],[]) #要給一個新的list 不然會重複使用到相同地址的list
        child.parents = box.parents.copy() #不能直接等於rrrrrr 不同pointer 指向同一個記憶體位置而已
        child.parents.append(box.items[i].Id)
        s1 = box.items[i].bitVector
        for j in range(i+1 , size):  #itemb 和下一個itemb要比較
            s_conclu = [0] * length
            s_conclu = np.array(s_conclu).astype(np.int32)
            s2 = box.items[j].bitVector #兩個bitvector 要做intersection 後續平行處理能夠實施在這個地方
            support = 0
#--------------------------------------------------------------------------------------------------------------------            
#GPU                
            support_for_gpu = [0] * BLOCK_NUM
            support_for_gpu = np.array(support_for_gpu)
            intersection(drv.In(s1), 
                         drv.In(s2), 
                         drv.Out(s_conclu), 
                         np.int32(length),
                         drv.Out(support_for_gpu),
                        block=(THREAD_NUM,1,1), grid=(BLOCK_NUM,1))
            for k in range(BLOCK_NUM):
                support += support_for_gpu[k]
#--------------------------------------------------------------------------------------------------------------------           
            if support >= count :                                      # bin ==>  轉成2進位   &0xffffffff 把bin回傳的"字串"中的負號削掉，最後在count看有幾個1
                child.items.append(itemB(box.items[j].Id, s_conclu , support))
            else:
                del s_conclu
        if len(child.items) != 0:
            eclat_gpu(child, count , length , f)                      #每次iterate出來的output會成為下一次的input，進入疊代後，會在呼叫一個children來儲存可能出現的下一層候選人
        for item in child.items:
            del item                                                  #當children找不到任何後代可能產生的候選集之後，停止疊代
        del child                                                     #depth-first search!!
    
    for item in box.items:
        for parent in box.parents:
            f.write(str(index[parent]) + ' ')
        f.write(str(index[item.Id]) + " ("+str(item.support)+")" + '\n')
            


# In[11]:


#-----start-----
#read in data
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
transaction_np = np.array( transaction)
total_num = len(transaction_np) # 總資料筆數


# In[12]:


bigbag1 = bigbag.replace("\n" , " ")
bigbag2 = bigbag1.split(" ") #bigbig2 ==> 交易中細項總和     
bag_solo = set(bigbag2)  #bag_solo ==> 交易中出現的細項(不重複)


# In[13]:


bag_solo_int = []
bag_solo_modi = list(bag_solo)
bag_solo_modi.sort()
bag_solo_modi.pop(0)



for i in list(bag_solo_modi):
    bag_solo_int.append(int(i))

bag_solo_int.sort()


# In[14]:


# 先做一個用編號當key的transaction dict
trans_dict = {}
for i in range(len(transaction_np)):
    trans_dict[i] = transaction_np[i]


# In[15]:


c1 = {} 
start = time.time()        #c1 key ==> 
for i in bag_solo_int:
    temp = []
    for key,value in trans_dict.items():
        if i in value:
            temp.append(key)
    temp = (temp)
    c1[i] = temp
end = time.time()
exe_time = (end - start)
print(exe_time)


# In[16]:


count = total_num * min_sup


# In[17]:


l1 = prune(c1 , count) #l1 ==> 652個


# In[18]:


box = Eclass()
Bitvector(total_num , l1 , box) #製作容器存放每個候選人交易資料的bit vector


# In[19]:


f = open(sys.argv[3], 'w')
length = total_num + 32 - (total_num % 32)         #length ==> 540455筆資料 ==> 湊成540480(32倍數) ==> 在變成16890 (個integer去存)
length = int(length/32)
start = time.time()
eclat_cpu(box, count , length, f)
end = time.time()
exe_time = (end - start)
print(exe_time)

