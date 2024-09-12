import os
import sys
from lightning import seed_everything

seed_everything(42)
sys.path.append("/home/huabei/project/SMTarRNA")
os.chdir("/home/huabei/project/SMTarRNA")
from mlp_gcn_mgnn import train_cmpx_model
import optuna

BATCH_SIZE = 1024
N_TRIALS = 100
MAX_EPOCHS = 50
STORAGE_PATH = 'sqlite:///GCN_MGNN_Tuning.db'
for model_name in ['MLP', 'GCN', 'MoleculeGNN']:
    for cmpx in ['3a6p', '4z4c', '4z4d', '6cbd']:
        study = optuna.create_study(direction='minimize',
                                storage=STORAGE_PATH,
                                study_name=f'{cmpx}_{model_name}',
                                load_if_exists=True)
    for cmpx in ['3a6p', '4z4c', '4z4d', '6cbd']:
        train_cmpx_model(cmpx, model_name,
                            STORAGE_PATH=STORAGE_PATH,
                            BATCH_SIZE=BATCH_SIZE,
                            MAX_EPOCHS=MAX_EPOCHS,
                            N_TRIALS=N_TRIALS)
