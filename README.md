# SIGMA: Similarity-based Efficient Global Aggregation for Heterophilous Graph Neural Networks
This repo contains code for SIGMA, current under review on WWW 2024. We provide the codes for large-scale datasets here.

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

## Long-term dependacy case study.

![avatar](case.png)

The global attention based method [1] utilizes the transformer structure to learn the global attention among nodes in graph classifications. We change the original task into node classifications on the dataset Chameleon. After training, the attention matrix is retrieved from the first layer of the transformer module and normalized rowly.

In the figure, we first calculate the average number of k-hop neighbors that share the same label with a node shown as the blue bar. Then, we calculate the average number of these neighbors holding a non-trivial attention/similarity scores, i.e. $s(u,v) > \frac{1}{m}$ for GraphTrans/SimGNN, where $m$ denotes the edge amount. The x-axis denotes the hop amount and y-axis is the log scaled amount.

We can observe in the figure that both GraphTrans and SimGNN can capture long-term dependacy effectively in general. While, as the distance becomes extreme large (> 10-hop), SimRank seems to show weaker ability to link the homophily nodes than the global attention scores, which points out a potential direction to improve SimGNN and also heterophily graph learning.

[1] Representing long-range context for graph neural networks with global attention, Z. Wu et al. NeurIPS 2021.
