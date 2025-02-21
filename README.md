# PINN-Preprocess

This repository implements a preprocessing method for Physics-Informed Neural Networks. The basic idea of this code is exactly the same as the non-dimensionalization concept in partial differential equations, where we simply perform a linear transformation to our equations and data so the data could be transformed to a space which can be easier to train using neural networks.

# Remark
The code for Baseline (Ordinary Non-dimensional Equations), InnerNormalization (Only Preprocess the input variables), and Normalization (Normalize both input and output), are all implemented.

The same idea goes beyond NS equations. Infact it is applicable to all kinds of PDEs.

The training and reference data can be found and downloaded at <https://drive.google.com/drive/folders/1a5V4JFsTmIsGLKmCHGyrN3c7Y9isVF6i?usp=drive_link>.

Annotations are in Chinese and English.

# Reference
Xu, S., Dai, Y., Yan, C., Sun, Z., Huang, R., Guo, D., & Yang, G. (2025). On the preprocessing of physics-informed neural networks: how to better utilize data in fluid mechanics. Journal of Computational Physics, 113837.
<https://doi.org/10.1016/j.jcp.2025.113837>
