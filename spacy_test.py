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
    df['review_vectors'] = df.review_docs.apply(lambda x:x.vector)
       
    train=df.sample(frac=0.8,random_state=200)
    test=df.drop(train.index)
    
    x = tf.placeholder(dtype = tf.float32)
    y = tf.placeholder(dtype = tf.int32)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = x))
    

if __name__ == '__main__':
    main()