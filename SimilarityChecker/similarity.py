# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:07:55 2018

@author: joyce
"""
from __future__ import print_function

import pickle, gzip
import math 
import igraph
import numpy as np
import sys

sys.path.append("..\OutlierDetector")
import mnist_helpers as mh


def getSets():
    # Load the dataset
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set  = pickle.load(f, encoding='latin1')
    f.close()
    return train_set, valid_set, test_set


def getNumbers(input_set, nr):
    alltwos = []
    for i in range(0,len(input_set[0])):
        if input_set[1][i] == nr:
            alltwos.append(input_set[0][i])
    return alltwos

def getResults(input_set, counter):
    filename = 'sim_scores/sim_'+str(counter)+'.txt'
    
    def square_rooted(x):
        return round(math.sqrt(sum([a*a for a in x])),3)
     
    def cosine_similarity(x,y):
        numerator = sum(a*b for a,b in zip(x,y))
        denominator = square_rooted(x)*square_rooted(y)
        return round(numerator/float(denominator),3)

    results = {}
       
    for i in range(0,len(input_set)):
        mapped[(counter,i)] = input_set[i]
        for j in range(0,len(input_set)):
            if i is not j and (i,j) not in results and (j,i) not in results:
                results[(i,j)] = cosine_similarity(input_set[i], input_set[j])

    with open(filename, 'w+') as f:
        for key, value in results.items():
            if value > 0.7:
                f.write(str(key[0])+'\t'+str(key[1])+'\t'+str(value)+'\n')


def clustering(filename, counter):

    G = igraph.read(filename,directed=False,format='ncol')
    
    D = G.community_multilevel()
    
    MIN_CLUSTER_SIZE = 1
    
    style = {}
    style['vertex_size'] = 10
    
    kept_clusters = 0
    clusters = []
    for i in D.subgraphs():
        if len(i.vs) > MIN_CLUSTER_SIZE:
            kept_clusters += 1
            clusters.append(i)
            
            style["layout"] = i.layout("fr", maxiter=250) # "fr" or "lgl" work nicely
    
            igraph.plot(i, 'cluster_visualization/cluster_'+str(kept_clusters)+'_'+str(counter)+'.pdf', **style)
            
    print('total clusters -->',len(D.subgraphs()))
    print('kept clusters -->',kept_clusters)
    return clusters


def getAllOutliers():    
    outliers = []
    for i in range(0,10):
        outliers.append(np.load('../OutlierDetector/outliers/'+str(i)+'.npy'))
        
    return outliers

    
def printImgCluster(c, counter):
    nrs = []
    for u in c.vs:
        nrs.append(int(u['name']))
        
    images = np.array([mapped[(counter,x)] for x in nrs]) 
        
    mh.show_some_digits(images,np.array([x for x in nrs]))


mapped = {}
clusters = []    

outliers = getAllOutliers()

counter = 0
for o in outliers:
    filename = 'sim_scores/sim_'+str(counter)+'.txt'
        
    getResults(o, counter)
    clus = clustering(filename, counter)
    clusters.append((counter,clus))
    
    print('--> digits for clusters of ', counter)

    for j in range(0,len(clus)):
        printImgCluster(clus[j],counter)

    counter += 1
