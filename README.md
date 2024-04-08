# SIGMA: Similarity-based Efficient Global Aggregation for Heterophilous Graph Neural Networks
This repo contains code for SIGMA, current under review on KDD 2024. We provide the codes for large-scale datasets here.

## Environment:
The implemetation is based on python and C++ (for simrank computations). 

### The python env requires:

- torch==1.10.1
- torch-scatter=2.0.9
- torch-sparse=0.6.13
- sklearn=0.0
- torch-geometric=2.2.0
- networkx=2.8.8

We also provide the conda environment yml file [SimGNN/simgnn.yml], which you can use to clone our environment.

### C++:
- please refer to SimRank/SimRankRelease project, which is a SOTA all-pair simrank computation algorithm. [Localpush] (https://ieeexplore.ieee.org/abstract/document/8509277)

## Steps:

1. Download the large-scale datasets from [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) to "data/" folder.

2. Calculate the approximate simrank matrix using the Localpush implementation. The input and output of Localpush are required all txt files.\n 
Note that the input format is "node_u node_v", denoting each edge every single line. And the output format is "node_u node_v value", denoting the simrank score of node pair (node_u, node_v)


3. Convert the output txt into a sparse matrix either in ".pt" or in ".npz" format. For example, "fb100-simrank.pt".

4. Finally! you can run pipelines. For example, to evaluate SimGNN on fb100 dataset, use the following command:

```
python main.py --method simgnn --dataset fb100 --sub_dataset Penn94 --simrank_file_name fb100-simrank.pt --hiddenunits 32 --lr 0.0007 --dropout 0.5 --weight_decay 0.0001 --delta 0.78 --epochs 200 --runs 5 --propa_mode post --skip_factor 1
```

