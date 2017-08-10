\\data\\# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:38:17 2017

@author: Florent
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import GCN_AE
import File_Reader
import Graph_Construct


#Load the Data
adjacency,list_adjacency,_=File_Reader.get_cluster("\\data\\facebook_combined.txt")

#Split in  Train, Test and Validation sets
train_test_split=File_Reader.train_test_split(adjacency)
train_adjacency=train_test_split[0]
sp_adjacency = File_Reader.dense_to_sparse(train_adjacency)

#Normalize the Adjacency Matrix
norm_adj_mat=File_Reader.normalize_adjacency(train_adjacency)

#Build the Variational  Graph Autoencoder
VGAE_1=GCN_AE.VGAE(n_nodes=adjacency.shape[0],n_hidden=200,n_latent=50,learning_rate=0.05)


#Train the Variational Graph Autoencoder.
for i in range(200): 
    loss,latent_loss,reconst_loss=VGAE_1.train_glob(sp_adjacency,norm_adj_mat,0.5)
    if i%10==0:
        _,ap = VGAE_1.auc_ap_scores(train_test_split[1],train_test_split[2])
        print("At step {0} \n Loss: {1}  \n Average Precision: {2}  ".format(i,loss,ap))
 

   
fpr,tpr,tresholds = VGAE_1.roc_curve_(train_test_split[1],train_test_split[2])

plt.plot(fpr, tpr)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC Curve')
plt.show()
