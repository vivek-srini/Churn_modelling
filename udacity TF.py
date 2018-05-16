import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_train=y_train.reshape(-1,1).astype('float32')
y_test=y_test.reshape(-1,1).astype('float32')
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')


import tensorflow as tf
learning_rate=0.1
num_hidden=10
num_epochs=100
num_features=X_train.shape[1]
num_classes=1
X=tf.placeholder(tf.float32,[None,num_features])
Y=tf.placeholder(tf.float32,[None,num_classes])

weights = {
    'hidden_layer': tf.Variable(tf.random_normal([num_features,num_hidden])),
    'out': tf.Variable(tf.random_normal([num_hidden,num_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([num_hidden])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

hidden_layer=tf.add(tf.matmul(X,weights['hidden_layer']),biases['hidden_layer'])
hidden_layer=tf.nn.relu(hidden_layer)
logits=tf.add(tf.matmul(hidden_layer,weights['out']),biases['out'])


cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_train))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(y_train,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for e in range(num_epochs):
        
        _,c=sess.run([optimizer,cost],feed_dict={X:X_train,Y:y_train})
        if e%5==0:
            print("Cost= ",c)
        


        
    
    
