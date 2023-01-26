# SimGNN
This repo contains code for SimGNN, current under review on ICML 2023.

## Environment:
The implemetation is based on python and C++ (for simrank computations). 

### The python env requires:

- torch==1.10.1
- torch-scatter=2.0.9
- torch-sparse=0.6.13
- sklearn=0.0
- torch-geometric=2.2.0
- networkx=2.8.8

We also provide the conda environment yml file, which you can use to clone our environment.

### C++:
-please refer to SimRank/SimRankRelease project, which is a SOTA all-pair simrank computation algorithm. [Localpush] (https://ieeexplore.ieee.org/abstract/document/8509277)