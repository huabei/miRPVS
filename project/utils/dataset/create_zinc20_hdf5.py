from pandas import HDFStore
from create_total_dataset_hdf5 import create_total_dataset_hdf5
import logging

import sys

data_folder = sys.argv[1]
# print(data_folder)
create_total_dataset_hdf5(data_folder)
print(f'{data_folder} Construct Done! Good Job!')


