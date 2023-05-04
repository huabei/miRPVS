# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:zinc_complex3a6p_data.py
@time:2022/10/13
"""
from torch_geometric.data import InMemoryDataset, Data
import torch
import numpy as np
from scipy import spatial
from torch.utils.data import random_split
from tqdm import tqdm
from rdkit import Chem
from collections import defaultdict
from torch_geometric.utils import dense_to_sparse, add_self_loops


def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


class ZincComplex3a6pDataSmilesRadius0(InMemoryDataset):

    def __init__(self, data_dir, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.elements_dict = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'CL': 6, 'BR': 7, 'I': 8, 'P': 9}
        super().__init__(data_dir, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['raw_data_smiles.txt']

    @property
    def processed_file_names(self):
        return ['Ligands_Graph_Data_Smiles_radius0.pt']

    def download(self):
        # Download to `self.raw_dir`
        raise IOError(f'There are No raw data in {self.raw_dir}!')

    def process(self, radius=0):
        # Read data into huge `Data` list.
        # raw_data = self.raw_dir
        # read file
        atom_dict = defaultdict(lambda: len(atom_dict))
        bond_dict = defaultdict(lambda: len(bond_dict))
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
        edge_dict = defaultdict(lambda: len(edge_dict))
        with open(self.raw_paths[0], 'r') as f:
            # first line is property_type
            # property_types = f.readline().strip().split()
            # split data
            data_original = f.read().strip().split('\n')
        # get data number
        # num_examples = len(data_original)
        # data list:[(atom, distance_matrix, label), ...]
        # items = list()
        # make every data graph
        total_ligands_graph = list()
        for data in tqdm(data_original):
            smiles, property = data.strip().split(',')
            property = torch.tensor(float(property), dtype=torch.float32)
            """Create each data with the above defined functions."""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)
            # edge_index = torch.tensor(np.where(adjacency == 1), dtype=torch.long)
            edge_index, edge_attr = dense_to_sparse(torch.tensor(adjacency, dtype=torch.float32))
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=1)
            
            d = Data(x=torch.tensor(fingerprints, dtype=torch.long), edge_index=edge_index, edge_attr=edge_attr,
                     y=property)
            # return d
            total_ligands_graph.append(d)

        if self.pre_filter is not None:
            total_ligands_graph = [data for data in total_ligands_graph if self.pre_filter(data)]

        if self.pre_transform is not None:
            total_ligands_graph = [self.pre_transform(data) for data in total_ligands_graph]
        data, slices = self.collate(total_ligands_graph)
        torch.save((data, slices), self.processed_paths[0])
        print(len(fingerprint_dict))
        self.fingerprint_dict = fingerprint_dict


if __name__ == '__main__':
    data = ZincComplex3a6pDataSmilesRadius0('3a6p/zinc_drug_like_100k/3a6p_pocket5_202020')
    pass
