#clustering using kmeans and DBSCAN
import pandas as pd
data=pd.read_csv('pfm_data.csv')
import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize


def word_tokenizer(text):
        #tokenizes and stems the text
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
        return tokens

#sklearn cluster
def cluster_sentences(sentences, nb_of_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                        stop_words=stopwords.words('english'),
                                        analyzer='word',
                                        max_df=0.9,
                                        min_df=0.1,
                                        lowercase=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)

#DBSCAN clustering
def cluster_sentencesDBSCAN(sentences, nb_of_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                        stop_words=stopwords.words('english'),
                                        analyzer='word',
                                        max_df=0.9,
                                        min_df=0.1,
                                        lowercase=True)
        #builds a tf-idf matrix for the sentences
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        kmeans = DBSCAN()
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)


if __name__ == "__main__":
        sentences = list(data['Contents'])
        print(len(sentences))
        for i in range(len(sentences)):
            sentences[i]=sentences[i].split()
            if 'Description' in sentences[i]:
                inde=sentences[i].index("Description")
                temp=sentences[i]
                sentences[i]=temp[inde+2:]
            sentences[i]=' '.join(sentences[i])
        punctuations = list(string.punctuation)
        punctuations.append("''")
        for i in range(len(sentences)):
            sentences[i]=[i.strip("".join(punctuations)) for i in word_tokenize(sentences[i]) if i not in punctuations]
            stemmer = WordNetLemmatizer()
            sentences[i] = [stemmer.lemmatize(t) for t in sentences[i] if t not in stopwords.words('english')]
            sentences[i]=" ".join(sentences[i])
        nclusters= 200
        clusters = cluster_sentences(sentences, nclusters)
        for cluster in range(nclusters):
                print("cluster ",cluster,":")
                for i,sentence in enumerate(clusters[cluster]):    
                    print("\tsentence ",sentence,": ",sentences[sentence],"\n","="*90)

