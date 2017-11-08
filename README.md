# BestSF (Best Sparse Format)
BestSF is a new learing-based sparse meta-format for optimizing the SpMV kernel on GPU. It is built on top of the CUDA based open source CUSP library. BestSF automatically selects the best sparse format from COO, CSR, DIA, ELL, and HYB for each input sparse matrix. 

You can find here the C++ code used in our study for extracting the sparsity features and collecting the training data. In our work, we used CUSP v0.5.0 (https://cusplibrary.github.io/) for running the SpMV kernel on GPU. 

A technical paper describing BestSF is currently under review. A full description will be available here as well. The code used for the learning and testing will be uploaded very soon. 


