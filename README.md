# word_dimensions
The Jupiter Notebook describes how words derive meaning for NLP products such as Alexa and Siri: https://github.com/bnicholl/word_dimensions/blob/master/word_vectors.ipynb

The gensim_vec_spaces.py is an algorithm that uses the gensim module to create word vectors from a pandas dataframe
The type that should be used for this algorithm is a pandas dataframe series with one column.                                 
Example shoing the type of data:
In [456]: type(extract_comments[1])                                                                                              
Out[456]: pandas.core.series.Series                                                                                             
                                                                                                                                
In [460]:extract_comments[1]
Out[460]:
0 Explanation\nWhy the edits made under my usern...
1 D'aww! He matches this background colour I'm s...
2 Hey man, I'm really not trying to edit war. It...
3 "\nMore\nI can't make any real suggestions on ...
4 You, sir, are my hero. Any chance you remember...

The methods in this algorithm go in sequential order. First you call wrangle(). The output of wrangle() then becomes the paramter for vector_space(). The output of vector_space() then becomes the parameter for matrix_space().

wrangle() simply gathers and cleans up all of the words in our corpus.
vector_space() puts those words in there respective vector spaces.
matrix_space() puts the vector spaces in the order of their original sentences.
You can get the train and test set I used to run this algorithm here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
