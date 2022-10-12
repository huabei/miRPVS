'''this file include origgnn pl model and dataset'''
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from scipy import spatial
import wandb
import time
class MolecularGNN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MolecularGNN")
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--lr_decay", type=float, default=0.99)
        # parser.add_argument("--N_atoms", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=4)
        return parent_parser
    
    def __init__(self, dim, layer_hidden, layer_output, learning_rate, lr_decay,**kwargs):
        super().__init__()
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.save_hyperparameters()
        self.predictions = defaultdict(list)
        # dataset
        # manage data
        # if elements_dict == None:
        elements_dict = defaultdict(lambda: len(elements_dict))
        self.elements_dict = elements_dict
        dataset = LigandDataset(self.hparams.data_path, self.elements_dict)
        N_atoms = len(self.elements_dict)
        train_size = int(len(dataset)*0.8)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(42))
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.embed_atom = nn.Embedding(N_atoms, dim)
        self.gamma = nn.ModuleList([nn.Embedding(N_atoms, 1)
                                    for _ in range(layer_hidden)])
        # self.gate = nn.ModuleList([nn.Linear(200, 1)
        #                     for _ in range(layer_hidden)])
        for i in range(layer_hidden):
            ones = nn.Parameter(torch.ones((N_atoms, 1)))
            self.gamma[i].weight.data = ones
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)
        self.N_atoms = N_atoms
        self.dim = dim
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output

    def forward(self, x):
        def update(matrix, vectors, layer):
            hidden_vectors = torch.relu(self.W_atom[layer](vectors))
            return vectors + torch.matmul(matrix, hidden_vectors)
        def sum(vectors, axis):
            sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
            return torch.stack(sum_vectors)

        """GNN layer (update the atom vectors)."""
        atom_vectors = self.embed_atom(x['atoms'])
        for l in range(self.layer_hidden):
            gammas = torch.squeeze(torch.sigmoid(self.gamma[l](x['atoms'])))
            M = torch.exp(-gammas*x['distance_matrix']**2)
            atom_vectors = update(M, atom_vectors, l)
            atom_vectors = F.normalize(atom_vectors, 2, 1)  # normalize.

        """Output layer."""
        for l in range(self.layer_output):
            atom_vectors = torch.relu(self.W_output[l](atom_vectors))

        """Molecular vector by sum of the atom vectors."""
        molecular_vectors = sum(atom_vectors, x['molecular_sizes'])

        """Molecular property."""
        properties = self.W_property(molecular_vectors)

        return torch.squeeze(properties, dim=1)

    def configure_optimizers(self):
        # 优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # 学习率控制,指数衰减
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.lr_decay)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': ExpLR}
        return optim_dict

    def training_step(self, train_batch, batch_idx):
        predicted_properties = self.forward(train_batch)
        # loss = F.smooth_l1_loss(predicted_properties, train_batch['label'])
        loss = F.mse_loss(predicted_properties, train_batch['label'])
        # self.log(batch_size=len(train_batch['id']))
        self.log('train_loss', loss, batch_size=len(train_batch['id']))
        return loss
    
    def training_epoch_end(self, outputs) -> None:
        return super().training_epoch_end(outputs)

    def validation_step(self, val_batch, batch_idx):
        predicted_properties = self.forward(val_batch)
        loss = F.mse_loss(predicted_properties, val_batch['label'])
        # loss = F.smooth_l1_loss(predicted_properties, val_batch['label'])
        # self.log(batch_size=len(val_batch['id']))
        self.log('val_loss', loss, batch_size=len(val_batch['id']))
    
    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch['label'].float())
        # loss = F.smooth_l1_loss(y_hat, batch['label'].float())
        self.predictions['id'].extend(batch['id'])        
        self.predictions['pred'].extend(y_hat.cpu().numpy())
        self.predictions['true'].extend(batch['label'].cpu().numpy())
        return loss

    def test_epoch_end(self, outputs) -> None:
        # dummy_input = dict()
        # dummy_input['atoms'] = torch.tensor(range(self.N_atoms), device=self.device)
        # dummy_input['distance_matrix'] = torch.ones((self.N_atoms, self.N_atoms), device=self.device)
        # dummy_input['molecular_sizes'] = self.N_atoms
        # model_filename = f'log/origgnn{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.onnx'
        # torch.onnx.export(self, dummy_input, model_filename, opset_version=11)
        # wandb.save(model_filename)
        return super().test_epoch_end(outputs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=pad_data, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=pad_data, num_workers=self.num_workers)

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['elements_dict'] = dict(self.elements_dict)
        return super().on_save_checkpoint(checkpoint)
    def on_load_checkpoint(self, checkpoint) -> None:
        self.elements_dict = checkpoint['elements_dict']
        return super().on_load_checkpoint(checkpoint)
class LigandDataset():
    '''get a datafile_path, create dataset, give a single data
        return {'atoms':, 'distance_matrix':, 'id':, 'label':}'''
    def __init__(self, file_path, elements_dict):
        """Load a dataset."""
        # read file
        with open(file_path, 'r') as f:
            # first line is property_type
            self._property_types = f.readline().strip().split()
            # split data
            self._data_original = f.read().strip().split('\n\n')
        # get data number
        self._num_examples = len(self._data_original)
        # data list:[(atom, distance_matrix, label), ...]
        self.items = list()
        # make every data
        for data in self._data_original:
            # get every row
            data = data.strip().split('\n')
            # get data id
            id = data[0]
            # get property in last row
            property = float(data[-1].strip())
            # get atoms and its coordinate
            atoms, atom_coords = [], []
            for atom_xyz in data[1:-1]:
                atom, x, y, z = atom_xyz.split()
                atoms.append(atom)
                xyz = [float(v) for v in [x, y, z]]
                atom_coords.append(xyz)
            # transform symbols to numbers, such as:{'C':0, 'N':1, ...}
            atoms = np.array([elements_dict[a] for a in atoms])
            # create distance matrix
            distance_matrix = spatial.distance_matrix(atom_coords, atom_coords)
            distance_matrix = np.where(distance_matrix == 0.0, 1e6, distance_matrix)
            item = {'atoms': torch.tensor(atoms), 
                    'distance_matrix': torch.tensor(distance_matrix), 
                    'id': id, 
                    'label': torch.tensor([property])}
            self.items.append(item)
    def __len__(self) -> int:
        return self._num_examples
    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        return self.items[index]

def pad_data(data_batch):
    '''transfer to torch dataloader, this function to pad data for batch process
        data_batch: list of dict
        return (atoms, distance_matrix, molecule_size)'''
    def pad(matrices, pad_value):
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
    batch_data = dict()
    batch_data['atoms'] = torch.cat([i['atoms'] for i in data_batch])
    batch_data['label'] = torch.cat([i['label'] for i in data_batch])
    batch_data['molecular_sizes'] = [len(i['atoms']) for i in data_batch]
    batch_data['distance_matrix'] = pad([i['distance_matrix'] for i in data_batch], 1e6)
    batch_data['id'] = [i['id'] for i in data_batch]
    return batch_data

