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
from rdkit.Chem import Descriptors


RDKIT_PROPS = {"1.0.0": ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
                         'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
                         'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
                         'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
                         'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt',
                         'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
                         'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
                         'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex',
                         'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
                         'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex',
                         'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount',
                         'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                         'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
                         'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
                         'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
                         'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
                         'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
                         'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
                         'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount',
                         'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
                         'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
                         'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
                         'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9',
                         'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
                         'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
                         'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
                         'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
                         'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
                         'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
                         'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
                         'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
                         'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                         'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
                         'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
                         'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
                         'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
                         'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
                         'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
                         'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
                         'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
                         'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
                         'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']
               }

CURRENT_VERSION = "1.0.0"


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


class ZincComplex3a6pDataSmilesMoleculeProp(InMemoryDataset):

    def __init__(self, data_dir, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.elements_dict = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'CL': 6, 'BR': 7, 'I': 8, 'P': 9}
        self.rdkit_2d = Rdkit2d()

        super().__init__(data_dir, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['raw_data_smiles.txt']

    @property
    def processed_file_names(self):
        return ['Ligands_Graph_Data_Smiles_Molecule_prop.pt']

    def download(self):
        # Download to `self.raw_dir`
        raise IOError(f'There are No raw data in {self.raw_dir}!')

    def process(self, radius=1):
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
            # 计算分子描述符
            molecule_descriptors = self.rdkit_2d(mol)
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)
            # edge_index = torch.tensor(np.where(adjacency == 1), dtype=torch.long)
            edge_index, edge_attr = dense_to_sparse(torch.tensor(adjacency, dtype=torch.float32))
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=1)
            # 构建全连接图的edge_index
            d = Data(x=torch.tensor(fingerprints, dtype=torch.long), edge_index=edge_index, edge_attr=edge_attr,
                     y=property, num_nodes=molecular_size, molecule_descriptors=torch.tensor(molecule_descriptors, dtype=torch.float32))
            # return d
            total_ligands_graph.append(d)

        if self.pre_filter is not None:
            total_ligands_graph = [data for data in total_ligands_graph if self.pre_filter(data)]

        if self.pre_transform is not None:
            total_ligands_graph = [self.pre_transform(data) for data in total_ligands_graph]
        data, slices = self.collate(total_ligands_graph)
        torch.save((data, slices), self.processed_paths[0])


class Rdkit2d(object):
    def __init__(self):
        self.properties = RDKIT_PROPS[CURRENT_VERSION]
        self._funcs = {name: func for name, func in Descriptors.descList}

    def __call__(self, mol) -> list[float]:
        if mol is None:
            r = [np.nan]*len(self.properties)
        else:
            r = []
            for prop in self.properties:
                r += [self._funcs[prop](mol)]
        return r


if __name__ == '__main__':
    data = ZincComplex3a6pDataSmilesMoleculeProp('3a6p/zinc_drug_like_100k/3a6p_pocket5_202020')
    pass
