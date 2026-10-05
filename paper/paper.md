---
title: 'TensorTrainNumerics.jl'
tags:
  - julia
  - tensor methods
  - numerics
  - differential equations
authors:
  - name: Martin Mikkelsen
    orcid: 0009-0005-3932-6563
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: University of Copenhagen, Department of Computer Science, Denmark
   index: 1
   ror: 00hx57361
date: 5 October 2026
bibliography: paper.bib

---

# Summary

Low-rank numerical tensor methods has gained a lot of traction recently and one of these framworks is the tensor train formulation. There are already many software packages for tensor network methods but these focus primaily on physics applications. `TensorNumericalMethods.jl` offers a unified package for dealing with numerical problems, such as solving high-dimensional partial differential equations through a user-friendly interface using a math-like notation but doing all the tensor network operations under the hood. 

# Statement of need

`TensorTrainNumerics.jl` is a julia package for solving partial differential equation using the tensor train formulation. There is specific empathasis on the quantics tensor train representation where we provide explicit representation of different operators often used in numerics while also providing support for different solvers from the numerics community and the machine learning community. `TensorTrainNumerics.jl` is written such that the code support tensor notation and doing the tensor operations in a convinient manner. `TensorTrainNumerics.jl` also allows for easy extensions if the user quickly wants to implement a new tensor train algorithm and compare to existing algorithms. There is already a Python [@oseledets_software_ttpy] and Matlab [@seledets_software_TT-toolbox] alternative but we provide more recent algorithms for tensor operations while having support for state-of-the-art algorithms. 

`TensorTrainNumerics.jl` was developed as a learning tool for tensor numerics which could easily be extended to benchmark new results and test different formulations. The package has already been used a publication [@FastandFlexible] and a preprint [@mikkelsen2026tensor]. This package should be a convinient way for new students wanting to dive into quantics tensor train numerical methods with existing explicit representations of operators, numerous solvers for both parabolic equations and time evolutions. The package also provides many different examples. 

# State of the field                                                                                                                  
There are already several tensor train related packages

`ttpy` [@oseledets_software_ttpy] is a Python package for tensor computations allowing linear algebra in up to 100 dimensions
`TT-toolbox` is a Matlab implementation of `ttpy``

`MPSKit.jl` [@mpskitjl] is a Julia package for matrix product states and matrix product operators for (quasi) one-dimensional quantum lattices and two-dimensional statistical mechanics models

`TensorKit.jl` [@tensorkitjl] is a Julia package for large-scale tensor computations, with a hint of category theory

`scikit_tt` is a Python package for simulation and analysis of high-dimensional problems 

`MPSTime.jl` [@MPSTime2025] is a Julia package for learning the joint probability distribution of time series directly from data using matrix-product state

`ITensors.jl` is a very popular Julia package for efficient tensor network algorithms.
                                                                                    
`TensorCrossInterpolation.jl` [@nunez2025learning] is a Julia package for learning tensor networks with tensor cross interpolation
                      

# References