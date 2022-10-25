#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nltk
# nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import porter
from nltk.text import TextCollection
import pandas as pd
import sys
import os
import json
import string
import re
from scipy import spatial
from heapq import nlargest
import operator
import time
from data_loader import DataLoader
import math


# In[2]:


# Load Data
def load():
    data = pd.concat([DataLoader(DataLoader.data_path1).load_table(),                       DataLoader(DataLoader.data_path2).load_table()], ignore_index = True)
    return data


# In[3]:


# Collect Documents
def collect(data):
    doc = []
    IDdoc = []
    
    for i in range(len(data)):
        # collection of [reviewText] and [summary]
        doc.append(preprocess(data.iloc[i]['reviewText'] + ' ' + data.iloc[i]['summary']))
        # collection of [reviewerID] and [asin]
        IDdoc.append([data.iloc[i]['reviewerID'].lower(), data.iloc[i]['asin'].lower()])
        
    return doc, IDdoc


# In[4]:


# Preprocess
def preprocess(doc):
    # case folding
    doc = doc.lower()
    
    # tokenize 
    doc = word_tokenize(doc)
    
    # remove punctuations
    doc = [w for w in doc if w.isalnum()]
    
    # remove stopwords
    stop_words = stopwords.words('english')
    doc = [w for w in doc if w not in stop_words]
    
    # stem
    stemmer = porter.PorterStemmer()
    doc = [stemmer.stem(w) for w in doc]
    
    return doc


# In[5]:


# Positional Index 
def index(doc, IDdoc):
    ind = {}
    id_collection = {}
    for i in range(len(doc)):
        if i%(int(len(doc)/10)) == 0:
            start = time.time()
        for j in range(len(doc[i])):
            if doc[i][j] not in ind.keys():
                ind[doc[i][j]] = [1]
                ind[doc[i][j]].append({})
                ind[doc[i][j]][1][i] = [j]
            elif i not in ind[doc[i][j]][1].keys():
                ind[doc[i][j]][0] += 1
                ind[doc[i][j]][1][i] = [j]
            else:
                ind[doc[i][j]][1][i].append(j)
        if IDdoc[i][0] not in id_collection.keys():
            id_collection[IDdoc[i][0]] = [i]
        else:
            id_collection[IDdoc[i][0]].append(i)
        if IDdoc[i][1] not in id_collection.keys():
            id_collection[IDdoc[i][1]] = [i]
        else: 
            id_collection[IDdoc[i][1]].append(i)
        if (i+1)%(int(len(doc)/10)) == 0:
            print("%s%% takes %.3f seconds" % (int((i+1)*10/(int(len(doc)/10))), time.time() - start))
    return ind, id_collection


# In[6]:


def rank(lis, doc, query, index, N):
    corpus = []
    for i in lis:
        corpus.extend(doc[i])
    corpus = list(set(corpus))
    df = []
    for term in corpus:
        k = 0
        for i in lis:
            if i in index[term][1]:
                k+=1
        df.append(k)
    
    tfidf = np.empty([len(lis), len(corpus)])
    for i in range(len(lis)):
        sumsquare = 0 
        for j, term in enumerate(corpus):
            if doc[lis[i]].count(term) == 0:
                tfidf[i][j] = 0
            else:
                tfidf[i][j] = (1 + math.log10(doc[lis[i]].count(term)))
            sumsquare += tfidf[i][j] * tfidf[i][j]
        norm = math.sqrt(sumsquare)
        tfidf[i] /= norm
        
    qvector = np.empty(len(corpus))
    sumsquare = 0
    for i, term in enumerate(corpus):
        if query.count(term) == 0:
            qvector[i] = 0
        elif df[i] == len(lis):
            qvector[i] = (1 + math.log10(query.count(term)))
        else:
            qvector[i] = (1 + math.log10(query.count(term))) * math.log10(len(lis)/df[i])
        sumsquare += qvector[i] * qvector[i]
    norm = math.sqrt(sumsquare)
    if norm != 0:
        qvector /= norm
    result = []
    cs = np.matmul(tfidf,qvector)
    index = nlargest(min(len(lis), N), enumerate(cs), key=operator.itemgetter(1))
    for i in range(min(len(lis), N)):
                result.append(lis[index[i][0]])
    score = nlargest(min(len(lis), N),cs)
    return result, score


# In[7]:


def check(lis, query):
    k = 0
    while len(lis) > 0:
        newlis = {}
        for i in range(len(query)-1):
            if query[i] not in lis.keys():
                continue
            if query[i+1] not in lis.keys():
                continue
            for pos in lis[query[i]]:
                if pos+1 in lis[query[i+1]]:
                    if query[i+1] not in newlis.keys():
                        newlis[query[i+1]] = [pos]
                    else:
                        newlis[query[i+1]].append(pos)
        lis = newlis  
        k += 1
    return k                       


# In[8]:


# Load Data
data = load()
# Collect Documents & Preprocess
doc, IDdoc = collect(data)


# In[9]:


# Positional Index    
ind, id_collection = index(doc, IDdoc)


# In[11]:


def result(index, id_collection, query, docum, data, N=10):
    resultIndex = []
    resultScore = []
    doubleid = []
    singleid = []
    
    for term in query:
        if term in id_collection.keys():
            for doc in id_collection[term]:
                if doc not in singleid:
                    singleid.append(doc)
                else:
                    singleid.remove(doc)
                    doubleid.append(doc)
                    
    allId = {}
    k = 0
    for term in query:
        if term in index.keys():
            k+=1
            for doc, pos in index[term][1].items():
                if doc not in allId.keys():
                    allId[doc] = [1, 0]
                    allId[doc].append({})
                else:
                    allId[doc][0] += 1
                if term not in allId[doc][2].keys():
                    allId[doc][2][term] = pos
                
    for i in range(k):
        if len(dict(filter(lambda elem: elem[1][0] >= k-i, allId.items()))) >= 50:
            allId = dict(filter(lambda elem: elem[1][0] >= k-i or elem[0] in singleid or elem[0] in doubleid, allId.items()))
            break
    
    i = 2
    while len(allId) > 200:
        allId = dict(filter(lambda elem: len(list(elem[1][2].values())[0]) >= i or elem[0] in singleid or elem[0] in doubleid, allId.items()))
        i+=1
                    
    for i in range(k):
        kIndex = []
        for doc, value in allId.items():
            if doc in doubleid:
                if value[0] >= k-i:
                    if value[1] == 0:
                        value[1] = check(value[2], query)
                    if value[1] == k-i:
                        kIndex.append(doc)
        kIndex, score = rank(kIndex, docum, query, index, N-len(resultIndex))
        for j in range(len(score)):
            score[j] += (k-i)
        resultIndex.extend(kIndex)
        resultScore.extend(score)
        if len(resultIndex) >= N:
            resultSnippet = snippet(resultIndex, query, allId, data)
            return resultIndex, resultScore, resultSnippet
    for doc in doubleid:
        if doc not in resultIndex:
            resultIndex.append(doc)
            resultScore.append(0)
            if len(resultIndex) >= N:
                resultSnippet = snippet(resultIndex, query, allId, data)
                return resultIndex, resultScore, resultSnippet
    for i in range(len(resultScore)):
        resultScore[i] += (1+k)
    
    for i in range(k):
        kIndex = []
        for doc, value in allId.items():
            if doc in singleid:
                if value[0] >= k-i:
                    if value[1] == 0:
                        value[1] = check(value[2], query)
                    if value[1] == k-i:
                        kIndex.append(doc)
        kIndex, score = rank(kIndex, docum, query, index, N-len(resultIndex))
        for j in range(len(score)):
            score[j] += (k-i)
        resultIndex.extend(kIndex)
        resultScore.extend(score)
        if len(resultIndex) >= N:
            resultSnippet = snippet(resultIndex, query, allId, data)
            return resultIndex, resultScore, resultSnippet
    for doc in singleid:
        if doc not in resultIndex:
            resultIndex.append(doc)
            resultScore.append(0)
            if len(resultIndex) >= N:
                resultSnippet = snippet(resultIndex, query, allId, data)
                return resultIndex, resultScore, resultSnippet
    for i in range(len(resultScore)):
        resultScore[i] += (1+k)
    
    for i in range(k):
        kIndex = []
        for doc, value in allId.items():
            if doc not in singleid and doc not in doubleid:
                if value[0] >= k-i:
                    if value[1] == 0:
                        value[1] = check(value[2], query)
                    if value[1] == k-i:
                        kIndex.append(doc)
        kIndex, score = rank(kIndex, docum, query, index, N-len(resultIndex))
        for j in range(len(score)):
            score[j] += (k-i)
        resultIndex.extend(kIndex)
        resultScore.extend(score)
        if len(resultIndex) >= N:
            resultSnippet = snippet(resultIndex, query, allId, data)
            return resultIndex, resultScore, resultSnippet
    
    resultSnippet = snippet(resultIndex, query, allId, data)
    return resultIndex, resultScore, resultSnippet


# In[32]:


def snippet(resultIndex, query, allId, data):
    snippets = []
    for doc in resultIndex:
        if doc not in allId:
            snippets.append("")
        else:
            allindex = {}
            k=0
            while len(allId[doc][2]) > 0:
                newlis = {}
                newterm = []
                for i in range(len(query)-1):
                    if query[i] not in allId[doc][2].keys():
                        continue
                    if query[i+1] not in allId[doc][2].keys():
                        continue
                    for pos in allId[doc][2][query[i]]:
                        if pos+1 in allId[doc][2][query[i+1]]:
                            if query[i+1] not in newlis.keys():
                                newlis[query[i+1]] = [pos]
                            else:
                                newlis[query[i+1]].append(pos)
                            for term in query[i:i+k+2]:
                                if term not in newterm:
                                    newterm.append(query[i])
                if len(allId[doc][2]) != len(newterm):
                    for term in allId[doc][2].keys():
                        if term not in newterm:
                            allindex[allId[doc][2][term][0]] = k        
                allId[doc][2] = newlis  
                k += 1
            
            text = []
            for term in (data.iloc[doc]['reviewText'] + ' ' + data.iloc[doc]['summary']).split():
                if preprocess(term) == []:
                    text.append("")
                else:
                    text.append(preprocess(term)[0])
            
            realindex = {}
            curr = -1
            prev = -1
            k = 0
            
            for i, term in enumerate(text):
                if k == 0:
                    if prev in realindex.keys():
                        realindex[prev] = i
                        prev = -1
                    if term != "":
                        curr += 1
                    if curr in allindex.keys():
                        realindex[i] = i+1
                        k = allindex[curr]
                        prev = i
                else:
                    if term != "":
                        curr += 1
                        k-=1
            
            wholeindex =[]
            for start in sorted(realindex.keys()):
                printstart = start-3
                printend = realindex[start]+3
                if printstart < 0:
                    printstart = 0
                if printend > len(text)-1:
                    printend = len(text)-1
                wholeindex.append([printstart, printend])
            
            if len(wholeindex) == 0:
                snippet = []
                snippet.append("...")
                for i in range(10):
                    snippet.append((data.iloc[doc]['reviewText'] + ' ' + data.iloc[doc]['summary']).split()[-10+i])
                snippets.append(" ".join(snippet))
                continue
                
            result = []
            for i in range(wholeindex[-1][1]+1):
                for index in wholeindex:
                    if i >= index[0] and i <= index[1]:
                        result.append(i)
                        break
            
            snippet = []
            if result[0] != 0:
                snippet.append("...")
            for i in range(result[0], result[-1]+1):
                if i in result:
                    snippet.append((data.iloc[doc]['reviewText'] + ' ' + data.iloc[doc]['summary']).split()[i])
                elif snippet[-1] != "...":
                    snippet.append("...")
            if result[-1] != len(text)-1:
                snippet.append("...")
            snippets.append(" ".join(snippet))
    return snippets


# In[ ]:


while True:
    # Query
    query = input("Search for: (type \"q\" to quit) ")
    if query == "q":
        break
    start = time.time()
    # Prepocess Query
    query = preprocess(query)
    # Result
    keyList = ["Rank", "DocID", "ReviewerID", "asin", "Snippets", "Score"]
    results = {key: [] for key in keyList}
    docIDs, scores, snippets = result(ind, id_collection, query, doc, data)

    for i, docID in enumerate(docIDs):
        results["Rank"].append(i+1)
        results["DocID"].append(docID)
        results["ReviewerID"].append(data.loc[docID]["reviewerID"])
        results["asin"].append(data.loc[docID]["asin"])
        results["Snippets"].append(snippets[i])
        results["Score"].append(scores[i])
    print(pd.DataFrame(results))
    print("Search takes %.3f seconds" % (time.time() - start))

