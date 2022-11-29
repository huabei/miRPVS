# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:molecular_gnn.py
@time:2022/10/14
"""
import torch
from torch.nn import Linear, Parameter, Embedding, ModuleList, Module
from torch_geometric.utils import add_self_loops, degree
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_scatter import scatter_sum
import numpy as np


class MolecularGraphNeuralNetwork(Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        self.embed_fingerprint = Embedding(N_fingerprints, dim)
        self.W_fingerprint = ModuleList([Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = ModuleList([Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = Linear(dim, 1)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N)))
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

    def mlp(self, vectors):
        """Classifier or regressor based on multilayer perceptron."""
        for l in range(self.layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs

    def forward(self, data_batch):
        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])
        molecular_vectors = self.gnn(inputs)
        predicted_values = self.mlp(molecular_vectors)
        loss = F.mse_loss(predicted_values, correct_values)
        return loss


if __name__ == '__main__':
    pass
