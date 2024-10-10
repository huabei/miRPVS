
import pickle
import sys

from mlp_gcn_mgnn import test_cmpx_model
import os
os.chdir('/home/huabei/project/SMTarRNA')
args = sys.argv  # [0] is the script name, [1] is the first argument, etc.

model_type = args[1] if len(args) > 1 else 'MLP'
devices = args[2] if len(args) > 2 else '0'
if devices == '0':
    devices = [0]
elif devices == '1':
    devices = [1]
elif devices == '2':
    devices = [2]
else:
    raise ValueError(f'Invalid device number: {devices}')
STORAGE_PATH = 'sqlite:///MLP_GCN_MGNN_Tuning.db'

if model_type == 'MLP':
    BATCH_SIZE = 1024
    MAX_EPOCHS = 10
elif model_type == 'GCN':
    BATCH_SIZE = 1024
    MAX_EPOCHS = 10
elif model_type == 'MGNN':
    BATCH_SIZE = 1024
    MAX_EPOCHS = 10
else:
    print('Invalid model type')
    sys.exit()

cmpx_list = ['3a6p', '4z4c', '4z4d', '6cbd']

results = []
for cmpx in cmpx_list:
    print(f'Testing {model_type} on {cmpx}')
    results.append(test_cmpx_model(cmpx, model_type, BATCH_SIZE, MAX_EPOCHS, DEVICES=devices,
                                   STORAGE_PATH=STORAGE_PATH))

with open(f'mlp_gcn_mgnn_results/{model_type}_4cmpx_results.pkl', 'wb') as f:
    pickle.dump(results, f)
