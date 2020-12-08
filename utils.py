#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:28:21 2019
@author:ginnyzhu
utility functions 
hat-tips to:
* https://gist.github.com/bwhite/3726239 for ranking functions 
"""
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#baseline libray
#from rank_bm25 import BM25Okapi
#fancy torch library
import torch 
import pickle
#from termcolor import colored



#get the sentences together
#processing all the publications first 
def get_corpus_and_dict(df, id_col, filepickle1, filepickle2, out_addr = '', name1 ='corpus', name2 ='corpus_dict'):
    '''
    get a list of id order
    do the (repetitive) content list appending
    then also ids to content dictionary
    '''
    id_ls = df[id_col].tolist()
    corpus = [] #check if a list or list of listst
    corpus_dict = {}
    for id in id_ls:
        temp = filepickle1[str(id)] + ' ' + filepickle2[str(id)]
        corpus.append(temp)
        corpus_dict[id] = temp
        
    print('length of the corpus', len(corpus))# 106,446
    print('sample of the corpus', corpus[:2])
    #dump these lists:
    with open(out_addr + name1+ '.pickle', 'wb') as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_addr + name2+ '.pickle', 'wb') as f:
        pickle.dump(corpus_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return corpus, corpus_dict

#for the nih processing 
def get_corpus_and_dict2(df, id_col, filecsv, file_id_col, field1, field2, out_addr = '', name1 ='corpus', name2 ='corpus_dict'):
    '''
    get a list of id order
    do the (repetitive) content list appending
    then also ids to content dictionary
    '''
    id_ls = df[id_col].tolist()
    corpus = [] #check if a list or list of listst
    corpus_dict = {}
 
    for id in id_ls:
        temp = filecsv.loc[filecsv[file_id_col]==id, field1].iloc[0] +' '+ filecsv.loc[filecsv[file_id_col]==id, field2].iloc[0]
        #break
        corpus.append(temp)
        corpus_dict[id] = temp
        
    print('length of the corpus', len(corpus))# 106,446
    print('sample of the corpus', corpus[:2])
    #dump these lists:
    with open(out_addr + name1+ '.pickle', 'wb') as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_addr + name2+ '.pickle', 'wb') as f:
        pickle.dump(corpus_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return corpus, corpus_dict






def groupedResults(df, groupby_cols, count_col, showtop = 30):
    '''
    df: dataframe 
    groupby_cols: a list of columns you want to be grouped by, normally one. format: list[]
    count_col: the column value to count on
    showtop: # of records to print (sorted from largest to smallest)
    return: grouped results as a dataframe
    '''
    groupeddf = pd.DataFrame(df.groupby(by= groupby_cols)[count_col].nunique())
    groupeddf = groupeddf.reset_index().sort_values(by=count_col, ascending = False)
    print(colored('after grouping shape:'+ str(groupeddf.shape), 'blue'))
    print(colored('top:'+ str(showtop), 'blue'))
    print(groupeddf.head(showtop))
    return groupeddf


def select_pts(df, col, criteria):
    '''
    df; the dataframe we're filtering 
    col; the column that we focus our filtering criteria on 
    criteria; the list that contains the criteria we need 
    return; the filtered df
    '''
    df_selected = df.loc[df[col].isin(criteria),:]
    return df_selected



#rewrite this function
def slice_and_order(cos_sim_arr, idx_top30_arr, rfa_id_ls, pmid_ls, save = True, file = 'ranked'):
    '''
    take cosine similarty array
    slice based on the BM25 top30 index 
    rearank based on similarity score
    save: if true, then save re-ranked dictionary
    file: path + name for the re-ranked dictionary, if save 
    return: the reranked pmid -- ranked rfa-id dictionary
    '''
    rerank_dict = defaultdict()
    cos_sim_arr30 = np.take_along_axis(cos_sim_arr, idx_top30_arr, axis=1)     
    #re-rank based on score
    rerank30_idx = (-cos_sim_arr30).argsort(axis =1) 
    #re-arrange the top30 orginal indices based on current similarity score 
    rerank_idx_top30_arr = np.take_along_axis(idx_top30_arr, rerank30_idx, axis=1) 
    for i in range(rerank_idx_top30_arr.shape[0]): #this should be #of pmid: 254, 687
        rerank_dict[pmid_ls[i]] = list(np.take(rfa_id_ls, rerank_idx_top30_arr[i]))
    if save:
        with open(file + '.pickle', 'wb') as handle:
            pickle.dump(rerank_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    return rerank_dict

#another function 
def slice_and_order2(cos_sim_arr, cos_top30_idxarr, rfa_id_ls, pmid_ls, save = True, file = 'ranked'):
    '''
    take cosine similarty array
    slice based on the consine_top30_idxarr
    save: if true, then save re-ranked dictionary
    file: path + name for the re-ranked dictionary, if save 
    return: the reranked pmid -- ranked rfa-id dictionary
    '''
    rerank_dict = defaultdict()
    cos_sim_arr30 = np.take_along_axis(cos_sim_arr, cos_top30_idxarr, axis=1)     
    #re-rank based on score
    #rerank30_idx = (-cos_sim_arr30).argsort(axis =1) 
    #re-arrange the top30 orginal indices based on current similarity score 
    #rerank_idx_top30_arr = np.take_along_axis(idx_top30_arr, rerank30_idx, axis=1) 
    for i in range(cos_sim_arr30.shape[0]): #this should be #of pmid: 254, 687
        rerank_dict[pmid_ls[i]] = list(np.take(rfa_id_ls, cos_top30_idxarr[i]))
    if save:
        with open(file + '.pickle', 'wb') as handle:
            pickle.dump(rerank_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    return rerank_dict



def cosine_distance_torch(x1, x2=None, eps=1e-8):
    '''
    #usage:
    #if we want to use cosine similarity 
    #An array with shape (n_samples_X, n_samples_Y)
    # little helper function here
    X = np.array([[2, 3], [3, 5], [5, 8]])
    Y = np.array([[1, 0], [2, 1]])
    npout = cosine_similarity(X, Y)
    #input1 = torch.Tensor(np.array([[2, 3], [3, 5], [5, 8]]))
    #input2 = torch.Tensor(np.array([[1, 0], [2, 1]]))
    #totally same thing, return pair-wise cosine similarity
    #cosine_distance_torch(input1, input2)

    '''
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)



#check match function
def prepare_topk(ref_dict, result_dict, topk=['10','20','30'], option = 'lump'):
    '''
    compare ref_dict with result_dict and calculate indicators based on scoring system 
    k: a list of ks as strings
    ref_dict:reference dictionary with pmids as the keys and a list of (un-ordered) correct rfa matches as the values 
    result_dict: the ranking top m (> max(topk)) returned by an algorithm, as a dictionary with with pmids as the keys 
    and a list of ORDERED rfa matches as the values 
    option could be 'lump' just one number for all, or 'ordered indicator'
    More detailedly, if 'lump':  if any of the topk in result gets a correct match, for example, top3  => [1]
    if 'ordered indicator': indicator of whether that particular element gets the match right, top3 = [0,1,0]
    return: top_dict with str(k) as the keys; list of list as the value, with each inner list corresponding to the indicator 
    for each key in reference pmid.
    
    '''
    ref_keys = list(ref_dict.keys())
    topk_dict = {k:[] for k in topk}
    for key in ref_keys:
        refList = ref_dict[key]
        refSet = set(refList)
        #resultList = result_dict[str(key)] #depending on 
        resultList = result_dict[key]
        for k in topk:
            resultList_k = resultList[:int(k)]
            if option.lower() == 'lump':
                ind = bool(refSet & set(resultList_k))
                ind = [int(ind == True)]
                topk_dict[k].append(ind)
            else:
                ind = [1 if x in refList else 0 for x in resultList_k]
                topk_dict[k].append(ind)
    return topk_dict 






def flat_correct(rs):
    '''
    return the flat_correct fraction using lump methods
    rs: list of list [[1], [0]... [0]] where each element is an indicator of whether for that pmid, topk 
    result has the right match or not
    '''
    return np.mean(rs)

def recall_atK(rs_atK,refLst):
    '''
    rs_atK: should be the array of indicators of at k recomendation fpr each pmid 
    rs_atK = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]);3 pmids, each recommendations take top 3, 1 and 0 indicates whether relevant or not 
    reflst; should be the # of matches for each pmid 
    refLst = np.array([[3], [2], [1]]): 3 pmids, each has 3, 2, 1 relevant items with it.
    '''
    row_divide = np.true_divide(np.array(rs_atK).sum(axis = 1),refLst)
    recall = np.mean(row_divide)
    return recall 
    
    

### borrowed functions, not my property try out these 
def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
        (let's do if for all rank numpy)
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])





def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])

'''
#I probably dont need these'
def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


if __name__ == "__main__":
    import doctest
    doctest.testmod()
'''