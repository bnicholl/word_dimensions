#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:50:33 2018
@author: bennicholl
"""
import pyspark
from pyspark.ml.feature import Word2Vec
import numpy as np
import pandas as pd
from gensim import models


train = pd.read_csv('/Users/bennicholl/Desktop/datasets/train.csv')
test = pd.read_csv('/Users/bennicholl/Desktop/datasets/test.csv')  
#extract_comments = train['id'],train['comment_text']
 
extract_comments = train['id'],train['comment_text']
classify = np.array([train['toxic'],train['severe_toxic'],train['obscene'],train['threat'],train['insult'],train['identity_hate']]).T

"""the object created with this function will be the input for 
   functions vector_space and matrix_space argument, 'comments' """
def wrangle(content = extract_comments):
    """this list will be used to fill our cleaned up, comma seperated data"""
    comments = []
    """iterates through each document(text"""
    for e,i in enumerate(content[1]):
        """stores our comma seperted characters, EX. d,o,g, ,r,a,n"""
        comma_seperated_chars = []
        """iterates throguh each individual element(letter) in each document"""
        for ii in i:
            """if iterator points to character that is not irrelevant, such as comma, /, etc"""    
            if ii != ',' and ii != '/' and ii != '!' and ii != '.' and ii != ';' and ii != '"' and ii != '\n' and ii != '-' and ii != '?' and ii != '(' and ii != ')' and ii != '|' and ii != '@' and ii != '#' and ii != ' [ ' and ii != ' ] ':        
                """add character to our comma_seperated_chars list"""
                comma_seperated_chars.append(ii)
        """when this code gets ran, join our comma seperated chars to produce strings"""        
        comments.append(''.join(comma_seperated_chars))
        """split each string with a comma. EX. 'the', 'dog', 'ran' """
        """wrapping the  comments[e].split()  in a [] is specifically for our createDataFrame"""
        comments[e] = comments[e].split()
    return comments


#word2Vec = Word2Vec(vectorSize=50, minCount=0, inputCol="text", outputCol="result")
#model = word2Vec.fit(a)

"""this function turns our words into vector spaces, than runs them through a neural net"""                                
"""the object created with this function will be the input for m_space's argument, 'v_space' """
def vector_space(comments, size = 40):
    model = models.Word2Vec(comments, size, min_count=1)
    return model    



"""this function puts the words together, thus getting a document matrix created out of word 
    vectors. in other words, this creates the sentences, but instead of words, we have vector
    representations of words"""
"""argument v_space's input is the output of function vecotr_space, and 
   argument comments input it the output of wrangle"""
def matrix_space(v_space, comments):
    """this list will be filled with our word vectors in the format of our original document"""
    vector_sentence = []
    """iterates through each document"""
    for e,i in enumerate(comments):
        """for each new document text, append a new list within the vector_sequence list"""
        vector_sentence.append([])
        """iterates through each word within each document"""
        for ee, ii in enumerate(i):
            """iterates through each word, than appends the word vector of that word to vector_sentence list"""                 
            vector_sentence[e].append( v_space.wv[comments[e][ee] ])   
    return vector_sentence
