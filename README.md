# Graph-VAE
We use the Variational Graph Autoencoder (T. Kipf, M. Welling https://arxiv.org/abs/1611.07308) for a link prediction task on network graphs. 

Variational Graph Autoencoders generate latent representations  of the adjacency matrix of a graph using  Graph Convolutional Networks https://arxiv.org/abs/1609.02907 and recover the adjacency matrix from this latent representation.


### Run_VGAE
Run this program to Build a VGAE and train/test  it on your graph. Will return the test average precision every 10 training epochs and the final ROC_Curve.

### File_Reader
Generates the Train,Test and Validations sets from  the raw .txt file.

### Initilization
Contains the Initialization, Sampling and  Graph Convolutional Layer operations.

### GCN_AE
Definition of the class  VGAE (Variational Graph Autoencoder).


# Data
The datasets are contained in the folder 'data' and correspond to text files containing the list of edges. 

The first example is a small subset of the Anonymized Facebook Graph with 4k Nodes and 88k edges. A test set of positive examples containing 10% of  the edges chosen at random is removed from the graph. An equal number of negative examples is sampled by  randomly choosing pairs of unconnected nodes. 

The algorithm achieves an average precision of roughly 99% on this test set (with  100 hidden units, and a latent vector of size 50).

# Requirements

python==2.7.6

tensorflow==1.0.0

scipy==0.19.0

numpy==1.12.1

