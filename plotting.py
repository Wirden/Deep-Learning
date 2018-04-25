# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:23:47 2018

@author: Wirden
"""
import pandas as pd
import amazon_parser as ap
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    myparse = ap.AmazonReviewsParser("../sorted_data/software/all.review")
    df = myparse.parse(myparse.filepath)
    df.set_index(df.date,inplace=True)

    
    df.rating = pd.to_numeric(df.rating)
    df['length_text'] = df.review_text.apply(lambda r : len(r))
    
    df_date = df['2006-Oct':'2007-Feb']
    
    #fig, ax = plt.subplots(nrows=2, ncols=1)
    
    #plt.grid()
    
    #df_date['length_text'].groupby(pd.TimeGrouper(freq='D')).mean().plot(ax=ax[0], grid=True)
    
    #print(df.info())
    
    print(df.groupby('rating')['length_text'].agg(np.mean).head())
    print((df.groupby('rating')['asin'].agg('count')/len(df.index)).head())
   
   # rating = df.groupby('rating')
   # serie2 = rating.length_text.agg(np.mean)
   # plt.pie(serie2,shadow=True)
    
    #df.plot(kind='scatter',x='length_text',y='rating',xax=ax[1], grid=True)
    #plt.scatter(df.length_text/1000,df.rating)

if __name__ == '__main__':
    main()