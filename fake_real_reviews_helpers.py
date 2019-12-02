#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np


def tokenize_text(data):
    """Makes tagged data with lowercased, tokenized words of text data.
        
    Args:
        data (list of strings): A list of text to tokenize
        
    Returns:
        tagged_data (list of TaggedDocument): A list of TaggedDocuments
    """
    tagged_data = []
    for i, d in enumerate(data):
        if type(d) is float:
            tagged_document = TaggedDocument(words=word_tokenize(str(d).lower()), tags=[str(i)])
        else:
            tagged_document = TaggedDocument(words=word_tokenize(d.lower()), tags=[str(i)])
        tagged_data.append(tagged_document)
    print(tagged_data)
    return tagged_data


def vec_for_learning(model, tagged_docs):
    """Builds the Final Vector Feature for the Classifier.
    
    The vec_for_learning function takes the trained Doc2Vec model and 
    tagged documents and infers the data into an exceptable format for 
    a ML Classifier to take as input.
    
    Args:
        model (Doc2Vec instance): The trained Doc2Vec model
        tagged_data (list of TaggedDocument): A list of TaggedDocuments
        
    """
    #print(tagged_docs)
    sents = tagged_docs
    regressors = [model.infer_vector(doc.words, steps=20) for doc in sents]
    return regressors


def from_list_of_lists(list_of_lists):
    """Takes a list of string lists and returns just a single list of strings.
    
    In a list is multiple lists with only the type str. This function gets rid 
    of inner lists and returns a single list of the strings.
    
    Args:
        list_of_lists: List of lists of type str
    
    Returns:
        list_of_strings: List of type str
    
    """
    list_of_strings = [text[0] for text in list_of_lists]
    return list_of_strings


def remove_nan_values(data):
    """Removes numpy nan values from a list of values.
    
    Args:
        data (list of values): A list of values which may contain nan values
        
    Returns:
        data (list of values): Returns same list of values back without nan data
    """
    i = 0
    while i < len(data):
        if type(data[i] != str):  
            # First type of nan value float nan
            if data[i] is np.nan:
                del data[i]
            # Second type of nan value float64 nan
            elif type(data[i]) is np.float64 and np.isnan(np.float64(data[i])):
                del data[i]
            else:
                i += 1
    return data

