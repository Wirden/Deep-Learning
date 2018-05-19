# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:03:18 2018

@author: Antoine
"""
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from amazon_parser import AmazonReviewsParser
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

def main():
    
    # learning rate (alpha)
    learning_rate = 0.05
    # batch size
    batch_size = 128
    # number of epochs
    num_steps = 5001
    
    nlp = spacy.load('en_core_web_lg')
    
    for word in STOP_WORDS:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True

    df = AmazonReviewsParser.parse("../sorted_data/software/all.review")
    df['length_review'] = df.review_text.apply(lambda x:len(x))
    df['review_docs'] = df.review_text.apply(lambda x:nlp(x))
    df['review_vector'] = df.review_docs.apply(lambda x:x.vector)   
    df.rating = df.rating.astype(int)
    
    one_hot_labels = np.zeros((len(df),5))
    one_hot_labels[np.arange(len(df)), df.rating.values-1] = 1
    
    X_train, X_test, y_train, y_test = train_test_split(
            df.review_vector.values,one_hot_labels, test_size=0.10)
    
    print(type(X_test))
    
    graph = tf.Graph()
 
    with graph.as_default():
    
        x = tf.placeholder(dtype = tf.float32)
        
        W = tf.Variable(tf.zeros([300, 5]))
        b = tf.Variable(tf.zeros([5]))
        test_dataset = tf.constant(X_test)
    
        logits = tf.matmul(x, W) + b
    
        y_ = tf.placeholder(tf.float32, [None, 5])
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=y_, logits=logits))
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(tf.matmul(test_dataset, W) + b)
        
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("Initialized")
            
            for step in range(num_steps):
            # pick a randomized offset
                offset = np.random.randint(0, y_train.shape[0] - batch_size - 1)
                
                # Generate a minibatch.
                batch_data = X_train[offset:(offset + batch_size), :]
                batch_labels = y_train[offset:(offset + batch_size), :]
                
                # Prepare the feed dict
                feed_dict = {x : batch_data, y_ : batch_labels}
                
                # run one step of computation
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                
                if (step % 500 == 0):
                    print("Minibatch loss at step {0}: {1}".format(step, l))
                    print("Minibatch accuracy: {:.1f}%".format(accuracy(predictions, batch_labels)))
                
                print("\nTest accuracy: {:.1f}%".format(accuracy(test_prediction.eval(), y_test)))
            
            

if __name__ == '__main__':
    main()