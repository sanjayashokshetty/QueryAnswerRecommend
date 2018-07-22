
# coding: utf-8

# In[ ]:


import pandas as pd


# In[4]:


data=pd.read_csv('data.csv')
sentances_train=list(data['Queries'])


# In[5]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def printrecommandations(sentance_test):
    maxpercent=0
    maxindex=-1
    for j in range(len(sentances_train)):
        if(len(sentances_train[j])>1):
            tuple1=(sentance_test,sentances_train[j])
            count_vectorizer = CountVectorizer(analyzer='word',stop_words='english')
            count_matrix = count_vectorizer.fit_transform(tuple1)
            result_cos = cosine_similarity(count_matrix[0:1],count_matrix)
            if(result_cos[0][1]*100>maxpercent):
                maxpercent=result_cos[0][1]*100
                maxindex=j
    print("Setence Test: \n",sentance_test)
    print(maxpercent)
    print("-"*50)
    print("Sentence Matched \n",data['Queries'][maxindex])
    print("Sentence Matched \n",data['Responses'][maxindex])
    print("="*90)


# In[8]:


string="i forget my password"
printrecommandations(string)


# In[ ]:


count_vectorizer = CountVectorizer(analyzer='word',stop_words='english')
count_matrix = count_vectorizer.fit_transform(sen)

