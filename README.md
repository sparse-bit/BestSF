# BestSF
BestSF is a new learing-based sparse meta-format for optimizing the SpMV kernel on GPU. It is built on top of the CUDA based open source CUSP library. BestSF automatically selects the best sparse format from COO, CSR, DIA, ELL, and HYB for each input sparse matrix. 

Your can find here the C++ code used in our study for extracting the sparsity features and collecting the training data.
