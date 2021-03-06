'''
2-LSTM approach trains the model using query and its response and predits the output by comparing train data vectors.
 '''
import pickle
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#loading the trained tfidf and countvectoriser model-------------
with open('tfidf.pkl', 'rb') as pickle_file:
    tfidf = pickle.load(pickle_file)
with open('count_vec.pkl', 'rb') as pickle_file:
    count_vect = pickle.load(pickle_file)

#loading the data
data=pd.read_csv("processed1.csv")
X_train,X_test=train_test_split(data,test_size=0.3)

def get_vector(sentence):
    doc_freq_term = count_vect.transform([sentence])
    return tfidf.transform(doc_freq_term)
def cosine(str1,str2):
    return cosine_similarity(get_vector(str1),get_vector(str2))

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

for k,v in X_test.Contents.items():
    printrecommandations(v)


#end of cosine similarity model

#get tfidf vectors for train data
X=[]
y=[]
for i,v in X_train.Contents.items():
    X.append(get_vector(v).todense())
    y.append(get_vector(X_train['Responses'][i]).todense())

#get tfidf vectors for test data
x_test_vec=[]
for i,v in X_test.Contents.items():
    x_test_vec.append(get_vector(v).todense())
x_test_vec=np.array(x_test_vec)
x_test_vec=x_test_vec.reshape(3077,1,50,589)

#reshape to feed as input to the model
X=np.array(X).reshape(7178,50,589)
y=np.array(y).reshape(7178,50,589)


#2 -LSTM model-------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

#model characteristics
time_steps=50
num_units=128
n_input=589
learning_rate=0.001
n_classes=300
batch_size=97
no_batch=int(len(X)/batch_size)

g1 = tf.Graph() ## This is first graph
g2 = tf.Graph() ## This is second graph



#GRAPH 1

with g1.as_default():
    x1=tf.placeholder("float",[None,time_steps,n_input])
    y1=tf.placeholder("float",[None,n_classes])
    total_outputs1=tf.placeholder(dtype=tf.float32)
    
    
    out_weights1=tf.Variable(tf.random_normal([time_steps*num_units,n_classes],seed=123))
    out_bias1=tf.Variable(tf.random_normal([n_classes],seed=123))
    inp1=tf.unstack(x1,time_steps,1)
    lstm_layer1=rnn.BasicLSTMCell(num_units,forget_bias=1)
    outputs1,_=rnn.static_rnn(lstm_layer1,inp1,dtype="float32")

    prediction1=tf.matmul(total_outputs1,out_weights1)+out_bias1
    
    tv1 = tf.trainable_variables()
    regularization_cost1 = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv1 ])
    #LOSS
    normalize_a1 = tf.nn.l2_normalize(y1,0)        
    normalize_b1 = tf.nn.l2_normalize(prediction1,0)
    loss1=1-tf.reduce_sum(tf.multiply(normalize_a1,normalize_b1))
    #LOSS END
    opt1=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss1)
    init1=tf.global_variables_initializer()
    correct_prediction1=tf.equal(tf.argmax(prediction1,1),tf.argmax(y1,1))

    
#GRAPH 2

with g2.as_default():
    x2=tf.placeholder("float",[None,time_steps,n_input])
    y2=tf.placeholder("float",[None,n_classes])
    total_outputs2=tf.placeholder(dtype=tf.float32)
    
    out_weights2=tf.Variable(tf.random_normal([time_steps*num_units,n_classes],seed=123))
    out_bias2=tf.Variable(tf.random_normal([n_classes],seed=123))
    x2=tf.placeholder("float",[None,time_steps,n_input])
    inp2=tf.unstack(x2,time_steps,1)
    lstm_layer2=rnn.BasicLSTMCell(num_units,forget_bias=1)
    outputs2,_=rnn.static_rnn(lstm_layer2,inp2,dtype="float32")
    prediction2=tf.matmul(total_outputs2,out_weights2)+out_bias2
    
    tv2 = tf.trainable_variables()
    regularization_cost2 = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv2 ])
    #LOSS
    normalize_a2 = tf.nn.l2_normalize(y2,0)        
    normalize_b2 = tf.nn.l2_normalize(prediction2,0)
    loss2=1-tf.reduce_sum(tf.multiply(normalize_a2,normalize_b2))
    #LOSS END
    
    opt2=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss2)
    init2=tf.global_variables_initializer()
    correct_prediction2=tf.equal(tf.argmax(prediction2,1),tf.argmax(y2,1))

#create two sessions
sess1 = tf.Session(graph=g1)
sess2 = tf.Session(graph=g2)
sess1.run(init1)
sess2.run(init2)

#training
iter=1
batch=1
count=0
prev_epoch_loss,epoch_loss=0,0
while True:
    batch_x1=X[(batch-1)*batch_size:(batch*batch_size)]
    batch_x2=y[(batch-1)*batch_size:(batch*batch_size)]
    batch=(batch+1)%74
    
    total_outputs_1=sess1.run(outputs1,feed_dict={x1:batch_x1})
    total_outputs_2=sess2.run(outputs2,feed_dict={x2:batch_x2})
    total_outputs_1=np.hstack(np.array(total_outputs_1))
    total_outputs_2=np.hstack(np.array(total_outputs_2))
    p1=sess1.run(prediction1,feed_dict={total_outputs1:total_outputs_1})
    p2=sess2.run(prediction2,feed_dict={total_outputs2:total_outputs_2})

    
    sess1.run(opt1,feed_dict={y1:p2,total_outputs1:total_outputs_1})
    sess2.run(opt2,feed_dict={y2:p1,total_outputs2:total_outputs_2})
    
#     Evaluating Model
    
    prev_epoch_loss=epoch_loss
    epoch_loss=sess1.run(loss1,feed_dict={total_outputs1:total_outputs_1,y1:p2})
    if abs(prev_epoch_loss-epoch_loss)<0.0001:
        count+=1
        if count==3:
            break
    else:
        count=0
    if iter%10==0:
        print("epoch ",iter,"Done!"," Loss is : ",epoch_loss)
    iter=iter+1
    if iter>=10000:
        break
print( epoch_loss)



#getting the vectors for training set from trained model
no_batch=74
answer_vectors=[]
query_vectors=[]
for i in range(no_batch):
    batch_x1=X[(batch-1)*batch_size:(batch*batch_size)]
    batch_x2=y[(batch-1)*batch_size:(batch*batch_size)]
    #start
    total_outputs_1=sess1.run(outputs1,feed_dict={x1:batch_x1})
    total_outputs_2=sess2.run(outputs2,feed_dict={x2:batch_x2})
    total_outputs_1=np.hstack(np.array(total_outputs_1))
    total_outputs_2=np.hstack(np.array(total_outputs_2))
    p1=sess1.run(prediction1,feed_dict={total_outputs1:total_outputs_1})
    p2=sess2.run(prediction2,feed_dict={total_outputs2:total_outputs_2})
    #end
    query_vectors.append(p1)
    answer_vectors.append(p2)

query_vectors=np.array(query_vectors).reshape(74*97,1,n_classes)
answer_vectors=np.array(answer_vectors).reshape(74*97,1,n_classes)

#testing the model
for query,vector in zip(X_test.Contents.items(),enumerate(x_test_vec)):
    query_index=query[0]
    query=query[1]
    vect_ind=vector[0]
    vector=vector[1].reshape(1,time_steps,589)
    #start
    total_outputs_1=sess1.run(outputs1,feed_dict={x1:vector})
    total_outputs_1=np.hstack(np.array(total_outputs_1))
    que_vec=sess1.run(prediction1,feed_dict={total_outputs1:total_outputs_1})
    #end
    minval = float('inf')
    minind=None
    for resp,ans_vec in zip(X_train.Responses.items(),enumerate(answer_vectors)):
        resp_ind=resp[0]
        if resp_ind in [188,4066,8466]:
            continue
        resp=resp[1]
        ans_vec_ind=ans_vec[0]
        ans_vec=answer_vectors[ans_vec_ind]
        temp = ((ans_vec-que_vec) ** 2).mean(axis=None)
        if temp<minval:
            minval=temp
            minind=resp_ind
            print(minval,resp_ind)
    print("test sent: ",query)
    print("------"*10)
    print("Matched Query: ",X_train["Contents"][minind])
    print("\nResponse : ",X_train["Responses"][minind])
    print("="*90)
#test end
