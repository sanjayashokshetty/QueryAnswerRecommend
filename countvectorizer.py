
# coding: utf-8

# In[2]:


#loading the trained tfidf and countvectoriser model-------------
import pickle
with open('tfidf.pkl', 'rb') as pickle_file:
    tfidf = pickle.load(pickle_file)
with open('count_vec.pkl', 'rb') as pickle_file:
    count_vect = pickle.load(pickle_file)


# In[4]:


#loading the data
import pandas as pd
data=pd.read_csv("processed1.csv")


# In[5]:


from sklearn.cross_validation import train_test_split
X_train,X_test=train_test_split(data,test_size=0.3)


# In[6]:


from sklearn.metrics.pairwise import cosine_similarity
def get_vector(sentence):
    doc_freq_term = count_vect.transform([sentence])
    return tfidf.transform(doc_freq_term)
def cosine(str1,str2):
    return cosine_similarity(get_vector(str1),get_vector(str2))


# In[47]:


#cosine similarity naive model--------------------------------------------
def printrecommandations(sentance_test):
    maxpercent=0
    maxindex=-1
    for j,v in X_train['Contents'].items():
        cos=cosine(sentance_test,v)[0][0]
        if(cos>maxpercent):
            maxpercent=cos
            maxindex=j
    if True:    
        print("Setence Test: \n",sentance_test)
        print(maxpercent*100)
        print("-"*50)
        print("Sentence Matched \n",data['Contents'][maxindex])
        print("Sentence Matched \n",data['Responses'][maxindex])
        print("="*90)        



count_vectorizer = CountVectorizer(analyzer='word',stop_words='english')
count_matrix = count_vectorizer.fit_transform(sen)

