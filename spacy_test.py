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
    
    one_hot_labels = np.zeros((len(df),5))
    one_hot_labels[np.arange(len(df)), df.rating.values-1] = 1
    
    X_train, X_test, y_train, y_test = 
        train_test_split(df.review_vector,one_hot_labels, test_size=0.10)
    
    graph = tf.Graph()
 
    with graph.as_default():
    
        x = tf.placeholder(dtype = tf.float32)
        
        W = tf.Variable(tf.zeros([300, 5]))
        b = tf.Variable(tf.zeros([5]))
    
        logits = tf.matmul(x, W) + b
    
        y_ = tf.placeholder(tf.float32, [None, 5])
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=y_, logits=logits))
        
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("Initialized")
            
            
            
            

if __name__ == '__main__':
    main()