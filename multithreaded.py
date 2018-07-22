
# coding: utf-8

# In[142]:


import time
start1=time.time()
import pandas as pd


# In[143]:


data=pd.read_csv('data.csv')
sentances_train=list(data['Queries'])


# In[144]:


import numpy as np
import queue
import multiprocessing as mp
from threading import Thread
from multiprocessing import Process
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
maxig=[]
maxpg=[]
q1=mp.Queue()
q2=mp.Queue()
def bubbleSort():
    global maxpg,maxig
    for passnum in range(len(maxig)-1,0,-1):
        for i in range(passnum):
            if maxpg[i]<maxpg[i+1]:
                temp = maxpg[i]
                temp1=maxig[i]
                maxpg[i] = maxpg[i+1]
                maxig[i] = maxig[i+1]
                maxpg[i+1] = temp
                maxig[i+1] = temp1
def printrecommandations(sentance_test,start,end):
    global maxig,maxpg,q1,q2
    maxi=[None]*3
    maxp=[0]*3
    for j in range(start,end):
        if(len(sentances_train[j])>1):
            tuple1=(sentance_test,sentances_train[j])
            count_vectorizer = CountVectorizer(analyzer='word',stop_words='english')
            count_matrix = count_vectorizer.fit_transform(tuple1)
            result_cos = cosine_similarity(count_matrix[0:1],count_matrix)
            p=int(result_cos[0][1]*100)
            if p in maxp:
                continue
            if p>maxp[0]:
                maxp[2]=maxp[1]
                maxp[1]=maxp[0]
                maxp[0]=p
                maxi[2]=maxi[1]
                maxi[1]=maxi[0]
                maxi[0]=j
            elif p>maxp[1]:
                maxp[2]=maxp[1]
                maxp[1]=p
                maxi[2]=maxi[1]
                maxi[1]=j
            elif p>maxp[2]:
                maxp[2]=p
                maxi[2]=j
    print(q1.empty())
    q1.put(maxp)
    print(q1.empty(),end)
    q2.put(maxi)
#     print(list(q1.queue),end)
def recommend(sentance_test):
    global maxpg,maxig,q1,q2
#     lock=multiprocessing.lock()
    t1=Process(target=printrecommandations,args=(sentance_test,0,500))
    t2=Process(target=printrecommandations,args=(sentance_test,500,1000))
    t3=Process(target=printrecommandations,args=(sentance_test,1000,1500))
    t4=Process(target=printrecommandations,args=(sentance_test,1500,len(sentances_train)))
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
#     print(q1.empty())
#     print(list(q1.queue))
    while(not q1.empty()):
        maxpg=maxpg+q1.get()
    while(not q2.empty()):
        maxig=maxig+q2.get()
    bubbleSort()
    print(maxpg,maxig)
    print("Setence Test: \n",sentance_test)
    for i in range(3):
        print(maxpg[i])
        print("-"*50)
        print("Sentence Matched \n",data['Queries'][maxig[i]])
        print("Sentence Matched \n",data['Responses'][maxig[i]])
        print("="*90)


# In[145]:


from threading import Thread
string="mutual account update"
import time
start=time.time()
recommend(string)
end=time.time()
print(end-start)
print(end-start1)

