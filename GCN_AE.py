# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:07:32 2017

@author: Florent
"""
import tensorflow as tf
import numpy as np
import File_Reader
import Initialization
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve

class VGAE (object):
        """ Parameters:
        ------------------
        n_nodes : size of input
        n_hidden : number of hidden units        
        n_output : size of output
        learning_rate : learning rate
               
        Attributes:
        ------------------
        W_0_mu: First layer Weight_array for the mu parameter
        W_1_mu: Second layer Weight array for the mu parameter
        W_0_sigma: First layer Weight_array for the sigma parameter
        W_1_sigma: Second layer Weight array for the sigma parameter
        
        Placeholders:
        -------------------
        adjacency: The adjacency matrix
        norm_adj_mat: The renormalized n_nodes x n_nodes  adjacency matrix
        dropout: dropout parameter
        """    
    
        def __init__(self,n_hidden,n_latent,n_nodes,learning_rate=0.01):
            self.n_nodes = n_nodes
            self.n_hidden = n_hidden
            self.n_latent = n_latent
            self.learning_rate = learning_rate
            
            
            self.shape=[self.n_nodes,self.n_nodes]
            self.shape = np.array(self.shape, dtype=np.int64)
            self.adjacency = tf.sparse_placeholder(tf.float32,shape=self.shape,name='adjacency')
            self.norm_adj_mat = tf.sparse_placeholder(tf.float32,shape=self.shape,name='norm_adj_mat')
            self.keep_prob = tf.placeholder(tf.float32)

            
            self.W_0_mu = None
            self.W_1_mu = None
            self.W_2_mu = None
            
            self.W_0_sigma=None
            self.W_1_sigma=None
            self.W_2_sigma=None
            
            self.mu_np=[]
            self.sigma_np=[]
            
            
            self.build_VGAE()
            
        def build_VGAE(self):
            
            #Initialize Weights
            self.W_0_mu = Initialization.unif_weight_init(shape=[self.n_nodes,self.n_hidden])
            self.W_1_mu = Initialization.unif_weight_init(shape=[self.n_hidden,self.n_hidden])
            self.W_2_mu = Initialization.unif_weight_init(shape=[self.n_hidden,self.n_latent])
            
            self.W_0_sigma = Initialization.unif_weight_init(shape=[self.n_nodes,self.n_hidden])
            self.W_1_sigma = Initialization.unif_weight_init(shape=[self.n_hidden,self.n_hidden])
            self.W_2_sigma = Initialization.unif_weight_init(shape=[self.n_hidden,self.n_latent])
            
                        
            #Compute Graph Convolutional Layers for the mean parameter
            hidden_0_mu_=Initialization.gcn_layer_id(self.norm_adj_mat,self.W_0_mu)
            hidden_0_mu=tf.nn.dropout(hidden_0_mu_,self.keep_prob)
            self.mu = Initialization.gcn_layer(self.norm_adj_mat,hidden_0_mu,self.W_2_mu)
            
            #Compute Graph Convolutional Layers for the variance  parameter
            hidden_0_sigma_=Initialization.gcn_layer_id(self.norm_adj_mat,self.W_0_sigma)
            hidden_0_sigma=tf.nn.dropout(hidden_0_sigma_,self.keep_prob)
            log_sigma = Initialization.gcn_layer(self.norm_adj_mat,hidden_0_sigma,self.W_2_sigma)
            self.sigma=tf.exp(log_sigma)
            
            #Latent Loss  Function. It is given by the KL divergence (closed formula)
            self.latent_loss = -(0.5 / self.n_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * tf.log(self.sigma)- tf.square(self.mu) - tf.square(self.sigma), 1))           
            
            #Reconstruction Loss. We use the weighted cross_entropy to take into account the sparsity of A
            dense_adjacency=tf.reshape(tf.sparse_tensor_to_dense(self.adjacency, validate_indices=False), self.shape)
            w_1 =  (self.n_nodes * self.n_nodes - tf.reduce_sum(dense_adjacency)) / tf.reduce_sum(dense_adjacency)
            w_2 = self.n_nodes * self.n_nodes / (self.n_nodes * self.n_nodes - tf.reduce_sum(dense_adjacency))
            self.reconst_loss =  w_2* tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=dense_adjacency,logits=self.decode(),pos_weight=w_1))
         
            #Loss Function
            self.loss =   self.reconst_loss + self.latent_loss
            
            #Optimizer
            self.optimizer=tf.train.AdamOptimizer(self.learning_rate)
            self.train_step=self.optimizer.minimize(self.loss)
                                                                               
            #Variables Initializer
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            

        def encode(self):
            '''Generates a latent representation'''
            return Initialization.sample_gaussian(self.mu,self.sigma)
        
        def decode (self):
            '''Generates the reconstruction'''
            z = self.encode()
            matrix_pred =tf.matmul(z,z,transpose_a=False,transpose_b=True)
            return matrix_pred
        
        
        def train_glob (self,sp_adjacency,norm_adj_mat,keep_prob):
            '''Performs batch  gradient descent'''
            feed_dict={self.adjacency: sp_adjacency[0:2],self.norm_adj_mat:norm_adj_mat[0:2],self.keep_prob:keep_prob}
            _,loss,latent_loss,reconst_loss,self.mu_np,self.sigma_np= self.sess.run([self.train_step,self.loss,self.latent_loss,self.reconst_loss,self.mu,self.sigma],feed_dict=feed_dict)
            return loss,latent_loss,reconst_loss
        
        #The two  following methods are similar to  encode  and decode, except they use the already updated
        #mean and variance rather than computing  them from the feed dictionnary.
        
        def latent(self):
            '''Returns a  sample of the latent variable Z'''
            z = Initialization.sample_gaussian_np(self.mu_np,self.sigma_np)
            return z    
        
        def predict(self):
            '''Returns  predictions for the adjacency matrix A'''
            z = Initialization.sample_gaussian_np(self.mu_np,self.sigma_np)
            matrix_pred=np.dot(z,np.transpose(z))
            return matrix_pred
        
        def auc_ap_scores(self,pos_edges,neg_edges):  
            '''Returns the auc and average precision score on a given test set with positive and negative examples'''
            pred=self.predict()
            s=np.vectorize(Initialization.sigmoid)
            pred=s(pred)
            preds=[]
            for e in pos_edges:
                preds.append(pred[e[0],e[1]])
            for e in neg_edges:
                preds.append(pred[e[0],e[1]])
            labels=np.hstack([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])
            auc_score = roc_auc_score(labels, preds)
            ap_score = average_precision_score(labels, preds)
            return auc_score, ap_score
        
        def roc_curve_(self,pos_edges,neg_edges):
            '''Returns the ROC  curve on a given  test set with positive and negative examples'''
            pred=self.predict()
            s=np.vectorize(Initialization.sigmoid)
            pred=s(pred)
            preds=[]
            for e in pos_edges:
                preds.append(pred[e[0],e[1]])
            for e in neg_edges:
                preds.append(pred[e[0],e[1]])
            labels=np.hstack([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])
            fpr,tpr,tresholds = roc_curve(labels,preds)
            return fpr, tpr,tresholds
        
        


                
       
        
            

                        
            
            

            
            
            
            
        
        