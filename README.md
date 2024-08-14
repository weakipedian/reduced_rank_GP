# Reduced Rank Gaussian Process Regression (GPR)
This repository provides example runs of reduced-rank Gaussian process regression (GPR) and kernel function approximation.

# Configuration
You can set hyperparameters for the kernel function (currently, only the Matern kernel is available) and parameters for the Hilbert space approximation method in 'config.yaml'.

# Example Scripts
- 'Kern_apx_example.py': Provides comparison results between the original kernel function and the approximated kernel function.

- 'Estimation_example.py': Provides comparison results between standard kernel-based GPR and reduced-rank GPR using a 2-D sample scalar function.

# Library
- 'ReducedRankGPLib.py':  A library for sparse GPR, including methods for model updates, query test inputs, and more.

# requirements for kernel-based GPR
pip3 install GPy