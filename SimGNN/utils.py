import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import warnings
import pickle as pkl
import networkx as nx
from collections import defaultdict

class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)

    
    
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# Read split data
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data_new(dataset_str, split):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    # print('dataset_str', dataset_str)
    # print('split', split)
    if dataset_str in ['citeseer', 'cora', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open('/home/lhy/GNN/GloGNN/small-scale/' + "data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            '/home/lhy/GNN/GloGNN/small-scale/' + "data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        splits_file_path = '/home/lhy/GNN/GloGNN/small-scale/' + 'splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = list(np.where(train_mask == 1)[0])
        idx_val = list(np.where(val_mask == 1)[0])
        idx_test = list(np.where(test_mask == 1)[0])

        no_label_nodes = []
        if dataset_str == 'citeseer':  # citeseer has some data with no label
            for i in range(len(labels)):
                if sum(labels[i]) < 1:
                    labels[i][0] = 1
                    no_label_nodes.append(i)

            for n in no_label_nodes:  # remove unlabel nodes from train/val/test
                if n in idx_train:
                    idx_train.remove(n)
                if n in idx_val:
                    idx_val.remove(n)
                if n in idx_test:
                    idx_test.remove(n)

    elif dataset_str in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        graph_adjacency_list_file_path = os.path.join('/home/lhy/GNN/GloGNN/small-scale', 'new_data', dataset_str, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('/home/lhy/GNN/GloGNN/small-scale', 'new_data', dataset_str, f'out1_node_feature_label.txt')
        graph_dict = defaultdict(list)
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                graph_dict[int(line[0])].append(int(line[1]))
                graph_dict[int(line[1])].append(int(line[0]))

        # print(sorted(graph_dict))
        graph_dict_ordered = defaultdict(list)
        for key in sorted(graph_dict):
            graph_dict_ordered[key] = graph_dict[key]
            graph_dict_ordered[key].sort()

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict_ordered))
        # adj = sp.csr_matrix(adj)

        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_str == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        features_list = []
        for key in sorted(graph_node_features_dict):
            features_list.append(graph_node_features_dict[key])
        features = np.vstack(features_list)
        features = sp.csr_matrix(features)

        labels_list = []
        for key in sorted(graph_labels_dict):
            labels_list.append(graph_labels_dict[key])

        label_classes = max(labels_list) + 1
        labels = np.eye(label_classes)[labels_list]

        splits_file_path = '/home/lhy/GNN/GloGNN/small-scale/' + 'splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = np.where(train_mask == 1)[0]
        idx_val = np.where(val_mask == 1)[0]
        idx_test = np.where(test_mask == 1)[0]
    
    ori_adj = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.array(np.where(labels)))[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # print('adj', adj.shape)
    # print('features', features.shape)
    # print('labels', labels.shape)
    # print('idx_train', idx_train.shape)
    # print('idx_val', idx_val.shape)
    # print('idx_test', idx_test.shape)
    return ori_adj, adj, features, labels, idx_train, idx_val, idx_test



#####################################################################################
# used in GGCN


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def row_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm='l1', axis=1)
    # row_sum = np.array(adj.sum(1))
    # row_sum = (row_sum == 0)*1+row_sum
    # adj_normalized = adj/row_sum
    return sp.coo_matrix(adj_normalized)


def precompute_degree_s(adj):
    adj_i = adj._indices()
    adj_v = adj._values()
    # print('adj_i', adj_i.shape)
    # print(adj_i)
    # print('adj_v', adj_v.shape)
    # print(adj_v)
    adj_diag_ind = (adj_i[0, :] == adj_i[1, :])
    adj_diag = adj_v[adj_diag_ind]
    # print(adj_diag)
    # print(adj_diag[0])
    v_new = torch.zeros_like(adj_v)
    for i in tqdm(range(adj_i.shape[1])):
        # print('adj_i[0,', i, ']', adj_i[0, i])
        v_new[i] = adj_diag[adj_i[0, i]]/adj_v[i]-1
    degree_precompute = torch.sparse.FloatTensor(
        adj_i, v_new, adj.size())
    return degree_precompute


def get_adj_high(adj_low):
    adj_high = -adj_low + sp.eye(adj_low.shape[0])
    return adj_high

#####################################################################################
# used in wrgat


def build_struc_layers(G, opt1=True, opt2=True, opt3=True, until_layer=None, workers=64):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    if(opt3):
        until_layer = until_layer
    else:
        until_layer = None

    G = struc2vec.Graph(G, False, workers, untilLayer=until_layer)

    if(opt1):
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs()

    if(opt2):
        G.create_vectors()
        G.calc_distances(compactDegree=opt1)
    else:
        G.calc_distances_all_vertices(compactDegree=opt1)

    G.create_distances_network()
    G.preprocess_parameters_random_walk()
    return


def build_multigraph_from_layers(networkx_graph, y, x=None):
    num_of_nodes = networkx_graph.number_of_nodes()

    x_degree = torch.zeros(num_of_nodes, 1)
    for i in range(0, num_of_nodes):
        x_degree[i] = torch.Tensor([networkx_graph.degree(i)])

    inp = open("struc_sim/pickles/distances_nets_graphs.pickle", "rb")
    distances_nets_graphs = pickle.load(inp, encoding="bytes")
    src = []
    dst = []
    edge_weight = []
    edge_color = []
    for layer, layergraph in distances_nets_graphs.items():
        filename = "struc_sim/pickles/distances_nets_weights-layer-" + \
            str(layer) + ".pickle"
        inp = open(filename, "rb")
        distance_nets_weights_layergraph = pickle.load(inp, encoding="bytes")
        for node_id, nbd_ids in layergraph.items():
            s = list(np.repeat(node_id, len(nbd_ids)))
            d = nbd_ids
            src += s
            dst += d
            edge_weight += distance_nets_weights_layergraph[node_id]
            edge_color += list(np.repeat(layer, len(nbd_ids)))
        assert len(src) == len(dst) == len(edge_weight) == len(edge_color)

    edge_index = np.stack((np.array(src), np.array(dst)))
    edge_weight = np.array(edge_weight)
    edge_color = np.array(edge_color)

    # print(edge_index.shape)
    # print(edge_weight.shape)
    if x is None:
        data = Data(x=x_degree, edge_index=torch.LongTensor(edge_index), edge_weight=torch.FloatTensor(edge_weight),
                    edge_color=torch.LongTensor(edge_color), y=y)
    else:
        data = Data(x=x, x_degree=x_degree, edge_index=torch.LongTensor(edge_index),
                    edge_weight=torch.FloatTensor(edge_weight),
                    edge_color=torch.LongTensor(edge_color), y=y)

    return data


def build_pyg_struc_multigraph(pyg_data):
    # print("Start build_pyg_struc_multigraph")
    start_time = time.time()
    # print(pyg_data)
    G = graph.from_pyg(pyg_data)
    # print('Before G', G)
    networkx_graph = to_networkx(pyg_data)
    # print('networkx_graph', networkx_graph)
    print("Done converting to networkx")
    build_struc_layers(G)
    # print('After G', G)
    print("Done building layers")
    data = build_multigraph_from_layers(networkx_graph, pyg_data.y, pyg_data.x)
    # print(data)
    if hasattr(pyg_data, 'train_mask'):
        data.train_mask = pyg_data.train_mask
        data.val_mask = pyg_data.val_mask
        data.test_mask = pyg_data.test_mask
    time_cost = time.time() - start_time
    print("build_pyg_struc_multigraph cost: ", time_cost)
    return data


def filter_rels(data, r):
    data = copy.deepcopy(data)
    mask = data.edge_color <= r
    data.edge_index = data.edge_index[:, mask]
    data.edge_weight = data.edge_weight[mask]
    data.edge_color = data.edge_color[mask]
    return data


def structure_edge_weight_threshold(data, threshold):
    data = copy.deepcopy(data)
    mask = data.edge_weight > threshold
    data.edge_weight = data.edge_weight[mask]
    data.edge_index = data.edge_index[:, mask]
    data.edge_color = data.edge_color[mask]
    return data


def add_original_graph(og_data, st_data, weight=1.0):
    st_data = copy.deepcopy(st_data)
    e_i = torch.cat((og_data.edge_index, st_data.edge_index), dim=1)
    st_data.edge_color = st_data.edge_color + 1
    e_c = torch.cat((torch.zeros(
        og_data.edge_index.shape[1], dtype=torch.long), st_data.edge_color), dim=0)
    e_w = torch.cat((torch.ones(
        og_data.edge_index.shape[1], dtype=torch.float)*weight, st_data.edge_weight), dim=0)
    st_data.edge_index = e_i
    st_data.edge_color = e_c
    st_data.edge_weight = e_w
    return st_data

#####################################################################################