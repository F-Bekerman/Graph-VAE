# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:49:58 2017

@author: Florent
"""

import numpy as np
import scipy.sparse as sp





def load_adjacency (file_name):
    '''Loads a txt file containing the list of edges and return the adjacency matrix in coo format'''
    file=open(file_name,'r')
    row=[]
    col=[]
    data=[]
    n_nodes=0
    for  line in file:
        edge= line.split()
        edge = list(map(int,edge))
        if edge[0]>10000 or edge[1]>10000:
            continue
        if edge[0]>n_nodes:
            n_nodes=edge[0]
        if edge[1]>n_nodes:
            n_nodes=edge[1]        
        row.extend([edge[0],edge[1]])
        col.extend([edge[1],edge[0]])
        data.extend([1,1])
    file.close()
    n_nodes=n_nodes+1
    adjacency = sp.coo_matrix((data,(row,col)), shape=(n_nodes,n_nodes))
    return adjacency


def normalize_adjacency(adjacency):
    '''Normalizes the adjacency matrix given in dense/coo format, returns in sparse format'''
    coo_adjacency = sp.coo_matrix(adjacency)
    adjacency_ = coo_adjacency + sp.eye(coo_adjacency.shape[0])
    d = np.array(adjacency_.sum(1))
    d_inv = sp.diags(np.power(d, -0.5).flatten())    
    normalized = adjacency_.dot(d_inv).transpose().dot(d_inv)
    return dense_to_sparse(normalized)

def normalize_dense_adjacency(adjacency):
    '''Normalizes the adjacency matrix given in dense format, returns it in dense format'''
    adjacency=np.eye(adjacency.shape[0]) + adjacency
    row_sums=adjacency.sum(1)
    d=np.diag(row_sums)
    d_inv = np.linalg.inv(np.sqrt(d))
    normalized = np.dot(d_inv,adjacency)
    normalized = np.dot(normalized,d_inv)
    return normalized


def dense_to_sparse (adjacency):
    '''Takes the adjacency matrix in dense/coo format and returns it in sparse format'''
    coo_adjacency = sp.coo_matrix(adjacency)
    indices = np.vstack((coo_adjacency.row, coo_adjacency.col)).transpose()
    values = coo_adjacency.data
    shape = np.array(coo_adjacency.shape,dtype=np.int64)
    return indices, values, shape


def train_test_split (adjacency):
    '''Generates a Training adjacency matrix, test  and validation sets of positive and negative
    examples from an adjacency matrix given in dense/coo format'''
    n_nodes=adjacency.shape[0]
    coo_adjacency = sp.coo_matrix(adjacency)
    coo_adjacency_upper = sp.triu(coo_adjacency,k=1)
    sp_adjacency = dense_to_sparse(coo_adjacency_upper)
    edges = sp_adjacency[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 10.))

    idx_all = list(range(edges.shape[0]))
    np.random.shuffle(idx_all)
    idx_test = idx_all[:num_test]
    idx_val = idx_all[num_test:(num_val + num_test)]

    test_edges_pos = edges[idx_test]
    val_edges_pos = edges[idx_val]
    train_edges = np.delete(edges, np.hstack([idx_test, idx_val]), axis=0)
    
    test_edges_neg=[]
    val_edges_neg=[]
    edge_to_add=[0,0]
    
    while(len(test_edges_neg)<len(test_edges_pos)):
        n1=np.random.randint(0,n_nodes)
        n2=np.random.randint(0,n_nodes)
        if n1==n2:
            continue        
        if n1<n2:
            edge_to_add=[n1,n2]
        else:
            edge_to_add=[n2,n1]        
        if any((edges[:]==edge_to_add).all(1)):
            continue
        test_edges_neg.append(edge_to_add)
        
    while(len(val_edges_neg)<len(val_edges_pos)):
        n1=np.random.randint(0,n_nodes)
        n2=np.random.randint(0,n_nodes)
        if n1==n2:
            continue        
        if n1<n2:
            edge_to_add=[n1,n2]
        else:
            edge_to_add=[n2,n1]        
        if any((edges[:]==edge_to_add).all(1)):
            continue
        val_edges_neg.append(edge_to_add)
    row=[]
    col=[]
    data=[]
    for edge in train_edges:
        row.extend([edge[0],edge[1]])
        col.extend([edge[1],edge[0]])
        data.extend([1,1])
    train_adjacency = sp.coo_matrix((data,(row,col)), shape=(n_nodes,n_nodes))
    return train_adjacency,test_edges_pos,test_edges_neg,val_edges_pos,val_edges_neg
        
       

 
'''
l= load_adjacency("facebook_combined.txt")
m=train_test_split(l)
print(normalize_adjacency(l))
'''