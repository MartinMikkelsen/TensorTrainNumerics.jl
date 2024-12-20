# Tensor trains

Based on the quantized tensor tensor (QTT) techniques, the asymptotic storage cost for a class of function-related $N-d$ (tensors) and matrices operated on them can be reduced to the logarithmic scale $N \rightarrow O(d \log N)$. In recent years, this breakthrough property was gainfully used for the representation and numerical treatment of various complicated multivariate functions representing basic physical quantities, as well as for solving $d$-dimensional PDEs in the QTT tensor representation. 

For the quantized representation of $f(\mathbf{u})$, each of the variables in the mathbftor $\mathbf{u}$ is rescaled between $[ 0,1 )$ and discretized on a line of $2^R$ points. This means we can represent the $i$-th element as
```math
    u_i = \sum_{b=1}^{R} \frac{\kappa_{ib}}{2^b},
```
where $\kappa_{ib}$ resolves the variable $u_i$ at the scale $2^{-b}$. If we choose $R$ sufficiently large, we can achive arbitrary high resolution. 