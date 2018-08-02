# QueryAnswerRecommend
Summer Intership 2018 Project at Perfios software solution Pvt Ltd

Model for recommending answers to customer support queries was built using tensorflow and trained on previous query responses.

## Datapreparation
![alt text](images/steps.png "Description goes here")

## Running the program

![alt text](images/hiw.png "Description goes here")

## Different Approaches Used

* Cosine Similarity With Tf-idf Vectors
* K-means clustering with doc2vec vectors
* Two LSTM with Tf-idf 


### Cosine Similarity with Tf-idf Vectors
 1. Create a big coprpus using the data.
 2. Train the count vectoriser and tfidf model.
 3. Get most similar by comparing the vector for each query from training data and find its consine similarity with the test       query vector.

### K-means clustering with doc2vec vectors

1. Create Doc-2-vec model and Train it. 
2. Get most similar vector to test vector.

### 2-LSTM aprroach
[Link to paper on the two LSTM approach](https://arxiv.org/pdf/1707.01378.pdf)

### Chrome  Extension 
The chrome extension was made for deployment purpose which sends the post request to the code running in Flask server and gets back the suggested response.
