#preprossesing the data training tfidf and countvectoriser model
import pandas as pd
data=pd.read_csv("que.csv")
data = data.replace('\n',' ', regex=True)
data['Response'][0]
Contents = data['Query'].tolist()
Responses=data['Response'].tolist()
#filtering one to one query response
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
#filter
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
#stemming and stopwords removal
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

#making the corpus
doc=Contents+Responses
doc=" ".join(doc)
import docx
document = docx.Document()
document.add_paragraph(doc)
document.save("corpus.docx")
doc_words=doc.split(" ")
len(doc_words)

#training countvectoriser
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect = count_vect.fit(doc_words)
freq_term_matrix = count_vect.transform(doc_words)

#training tfidftransformer
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)

doc_freq_term = count_vect.transform([Responses[0]])
doc_tfidf_matrix = tfidf.transform(doc_freq_term)

#saving the model
import pickle
with open('count_vec.pkl','wb') as f:
    pickle.dump(count_vect,f)

with open('tfidf.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)



