# Autoencoders-for-dimensionality-reduction
A simple, single hidden layer example of the use of an autoencoder for dimensionality reduction
 
A challenging task in the modern 'Big Data' era is to reduce the feature space since it is very computationally expensive to perform any kind of analysis or modelling in today's extremely big data sets. There is variety of techniques out there for this purpose: PCA, LDA, Laplacian Eigenmaps, Diffusion Maps, etc...Here I make use of a Neural Network based approach, the Autoencoders. An autoencoder is essentially a Neural Network that replicates the input layer in its output, after coding it (somehow) in-between. In other words, the NN tries to predict its input after passing it through a stack of layers. The actual architecture of the NN is not standard but is user-defined and selected. Usually it seems like a mirrored image (e.g. 1st layer 256 nodes, 2nd layer 64 nodes, 3rd layer again 256 nodes). 

In this simple, introductory example I only use one hidden layer since the input space is relatively small initially (92 variables). For larger feature spaces more layers/more nodes would possibly be needed. I am reducing the feature space from these 92 variables to only 16. The AUC score is pretty close to the best NN I have built for this dataset (0.753 vs 0.771) so not much info is sucrificed against our 5-fold reduction in data.  

After building the autoencoder model I use it to transform my 92-feature test set into an encoded 16-feature set and I predict its labels. Since I know the actual y labels of this set I then run a scoring to see how it performs. 

The data set used is the UCI credit default set which can be found here: 
https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
