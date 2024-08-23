# Expint_mfsoc -- Exponential integrators for mean-field selective optimal control problems

This is a companion software to the manuscript

[G. Albi, M. Caliari, E. Calzola, and F. Cassini. Exponential integrators for
mean-field selective optimal control problems,
arXiv preprint arXiv:2302.00127.](https://doi.org/10.48550/arXiv.2302.00127)

In particular, this repository contains the MATLAB scripts needed to
reproduce the numerical examples of the article. The main files are:
* ```main.m``` to run the experiments;
* ```main_order.m``` to check the order of convergence of the integrators.

All the scripts contain an initial description of what is implemented therein.
They can be directly executed in MathWorks MATLAB(R). In fact, the code
has been tested with MathWorks MATLAB(R) R2019a, R2022a, R2023a, and R2024a on
Windows and Linux distributions.

The needed actions of matrix functions are computed by using the KIOPS
algorithm, described in the relevant
[manuscript](https://doi.org/10.1016/j.jcp.2018.06.026) and
encoded in the function ```kiops.m```. The source code of this
routine, already included in the repository for convenience, can be found
[here](https://gitlab.com/stephane.gaudreault/kiops).
