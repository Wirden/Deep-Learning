# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:03:18 2018

@author: Antoine
"""
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from amazon_parser import AmazonReviewsParser
import numpy as np
import tensorflow as tf

def main():
    
    nlp = spacy.load('en_core_web_lg')
    
    for word in STOP_WORDS:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True

    df = AmazonReviewsParser.parse("../sorted_data/software/all.review")
    df['length_review'] = df.review_text.apply(lambda x:len(x))
    df['review_docs'] = df.review_text.apply(lambda x:nlp(x))
    df['review_vector'] = df.review_docs.apply(lambda x:x.vector)   
    df.rating = df.rating.astype(int)
    
    b = np.zeros((len(df),5))
    b[np.arange(len(df)), df.rating.values-1] = 1

    print(b)
    
  #  train_features = df['review_vector'].sample(frac=0.8,random_state=200)
   # train_labels = df['rating_hot'].sample(frac=0.8,random_state=200)
    
   # test_features = df['review_vector'].drop(train_features.index)
    #test_labels = df['rating_hot'].drop(train_labels.index)
    
   # print (test_features)
    #print(test_labels)
    
    graph = tf.Graph()
 
    with graph.as_default():
    
        x = tf.placeholder(dtype = tf.float32)
        
        W = tf.Variable(tf.zeros([300, 4]))
        b = tf.Variable(tf.zeros([4]))
    
        logits = tf.matmul(x, W) + b
    
        y_ = tf.placeholder(tf.float32, [None, 4])
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=y_, logits=logits))
        
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("Initialized")
            
            


    
    
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = x))
    

if __name__ == '__main__':
    main()