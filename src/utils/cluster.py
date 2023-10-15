import pandas as pd


from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import pickle

logging.basicConfig(level=logging.INFO, filename='cluster.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def ClusterFps(fps,cutoff=0.2):
    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    logging.info('calculating distance matrix')
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])
    logging.info('cluster')
    # now cluster the data:
    cs = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
    return cs

complex_name = ['3a6p', '4z4d', '4z4c', '6cbd']


# # 读取数据
logging.info('reading data')

fps_data = {}

for name in complex_name:
    data_path = 'data_' + name + '_zinc_id_smiles_fp_frame.pkl'
    data_name = pd.read_pickle(data_path)
    fps_data[name] = data_name['fp'].tolist()
    logging.info('reading data done')

# 聚类

for name in complex_name:
    logging.info('clustering')
    cluster = ClusterFps(fps_data[name], cutoff=0.8)
    logging.info('clustering done')
    logging.info('saving cluster')
    with open(f'cluster_{name}.pkl', 'wb') as f:
        pickle.dump(cluster, f)

