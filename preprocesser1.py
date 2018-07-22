
# coding: utf-8

# In[14]:


#preprossesing the data training tfidf and countvectoriser model
import pandas as pd
data=pd.read_csv("que.csv")


# In[15]:


data = data.replace('\n',' ', regex=True)


# In[16]:


data['Response'][0]


# In[17]:


Contents = data['Query'].tolist()
Responses=data['Response'].tolist()
for i in range(len(Responses)):
    count=0
    for j in range(len(Responses[i])):
        if Responses[i][j]=='-':
            count+=1
        else:
            count=0
            continue
        if count==3:
            index=j-3
            break
    Responses[i]=Responses[i][:index+1]
import copy
Unprocessed_Contents=copy.deepcopy(Contents)
Unprocessed_Responses=copy.deepcopy(Responses)
for i in range(len(Responses)):
    flag=0
    Responses[i]=Responses[i].split()
    if 'Regards' in Responses[i]:
        index1=Responses[i].index("Regards")
        flag=1
    if 'regards' in Responses[i]:
        index1=Responses[i].index("regards")
        flag=1
    if flag==1:
        Responses[i]=Responses[i][3:index1]
    else:
        Responses[i]=Responses[i][3:]
    Responses[i]=' '.join(Responses[i])
for i in range(len(Contents)):
    flag=0
    Contents[i]=Contents[i].split()
    if 'Thank' in Contents[i]:
        index1=Contents[i].index('Thank')
        flag=1
    if 'thank' in Contents[i]:
        index1=Contents[i].index("thank")
        flag=1
    if 'thanks' in Contents[i]:
        index1=Contents[i].index("thanks")
        flag=1
    if flag==1:
        Contents[i]=Contents[i][0:index1]
    else:
        pass
    Contents[i]=' '.join(Contents[i])

for i in range(len(Contents)):
    Contents[i]=Contents[i].split()
    if 'Description' in Contents[i]:
        inde=Contents[i].index("Description")
        temp=Contents[i]
        Contents[i]=temp[inde+2:]
    Contents[i]=' '.join(Contents[i])
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from nltk.stem import WordNetLemmatizer
punctuations = list(string.punctuation)
punctuations.append("''")
for i in range(len(Contents)):
    Contents[i]=[i.strip("".join(punctuations)) for i in word_tokenize(Contents[i]) if i not in punctuations]
    stemmer = WordNetLemmatizer()
    Contents[i] = [stemmer.lemmatize(t) for t in Contents[i] if t not in stopwords.words('english')]
    Contents[i]=" ".join(Contents[i])
for i in range(len(Responses)):
    Responses[i]=[i.strip("".join(punctuations)) for i in word_tokenize(Responses[i]) if i not in punctuations]
    stemmer = WordNetLemmatizer()
    Responses[i] = [stemmer.lemmatize(t) for t in Responses[i] if t not in stopwords.words('english')]
    Responses[i]=" ".join(Responses[i])


# In[18]:


import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
def extract_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]

def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
    return names

def func(string):
    numbers = extract_phone_numbers(string)
    emails = extract_email_addresses(string)
    return numbers+emails


# In[19]:


doc=Contents+Responses


# In[20]:


doc=" ".join(doc)


# In[26]:


import docx
document = docx.Document()
document.add_paragraph(doc)


# In[27]:


document.save("corpus.docx")


# In[28]:


doc_words=doc.split(" ")


# In[35]:


len(doc_words)


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
count_vect = count_vect.fit(doc_words)
freq_term_matrix = count_vect.transform(doc_words)


# In[33]:


freq_term_matrix


# In[37]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)


# In[64]:


doc_freq_term = count_vect.transform([Responses[0]])
doc_tfidf_matrix = tfidf.transform(doc_freq_term)


# In[67]:


import pickle
with open('count_vec.pkl','wb') as f:
    pickle.dump(count_vect,f)


# In[65]:


with open('tfidf.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)


# In[68]:


doc_tfidf_matrix


# In[61]:


doc_tfidf_matrix.nonzero()


# In[63]:


Responses[0]


# In[66]:




