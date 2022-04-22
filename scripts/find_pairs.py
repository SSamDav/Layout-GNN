import json
import gzip
import random
from pathlib import Path

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from joblib import Parallel, delayed

from functools import partial

from sklearn import neighbors
from layout_gnn.dataset.dataset import RICOSemanticAnnotationsDataset
from layout_gnn.dataset import transformations
from layout_gnn.similarity_metrics import compute_edit_distance, compute_iou
from layout_gnn.utils import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

ROOT_PATH = Path.cwd()
DATA_PATH = ROOT_PATH / '../data'
DATA_PATH.mkdir(parents=True, exist_ok=True)

IOU_NEG_THRESH = lambda x: x < 0.32
IOU_POS_THRESH = lambda x: x > 0.50


GED_NEG_THRESH = lambda x: x > 0.66
GED_POS_THRESH = lambda x: x < 0.35



class NeighborsDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
        self.rico_dataset = RICOSemanticAnnotationsDataset(
            transform=transforms.Compose([
                transformations.process_data,
                transformations.normalize_bboxes,
                transformations.add_networkx,
            ]),
            only_data=True
        )
        
        with gzip.open(DATA_PATH / 'neighbors.gzip', 'rt') as fp:
            neighbors = json.load(fp)
            
        self.filename2idx = {x.stem: i for i, x in enumerate(self.rico_dataset.files)}
        self.neighbors = [{
            'anchor':  self.filename2idx[n['anchor']],
            'closest': [self.filename2idx[j] for j in n['closest'] if j in self.filename2idx],
            'farthest': [self.filename2idx[j] for j in n['farthest'] if j in self.filename2idx]
        } for n in neighbors if n['anchor'] in self.filename2idx]
    
    def __len__(self):
        return len(self.neighbors)
    
    def __getitem__(self, idx):
        return compute_pairs(
            self.rico_dataset[self.neighbors[idx]['anchor']],
            self.neighbors[idx]['closest'],
            self.neighbors[idx]['farthest'],
            self.rico_dataset
        )
            
def compute_pairs(anchor, closest, farthest, dataset):
    datapoint = {
        'anchor': anchor['filename']
    }
    
    
    for idx in closest:
        neighbour = dataset[idx]
        if 'pos_iou' not in datapoint:
            iou_metric = compute_iou(anchor, neighbour)
            if IOU_POS_THRESH(iou_metric['iou']):
                datapoint['pos_iou'] = {
                    'pair': neighbour['filename'],
                    **iou_metric
                }
        if 'pos_ged' not in datapoint:
            ged_metric = compute_edit_distance(anchor['graph'].to_undirected(), neighbour['graph'].to_undirected())
            if GED_POS_THRESH(ged_metric['normalized_edit_distance']):
                datapoint['pos_ged'] = {
                    'pair': neighbour['filename'],
                    **ged_metric
                }
                
        if len(datapoint) == 3:
            break
                
    for idx in farthest:            
        neighbour = dataset[idx]
        if 'neg_iou' not in datapoint:
            iou_metric = compute_iou(anchor, neighbour)
            if IOU_NEG_THRESH(iou_metric['iou']):
                datapoint['neg_iou'] = {
                    'pair': neighbour['filename'],
                    **iou_metric
                }
            
        if 'neg_ged' not in datapoint:   
            ged_metric = compute_edit_distance(anchor['graph'].to_undirected(), neighbour['graph'].to_undirected())
            if GED_NEG_THRESH(ged_metric['normalized_edit_distance']):
                datapoint['neg_ged'] = {
                    'pair': neighbour['filename'],
                    **ged_metric
                }

        if len(datapoint) == 5:
            break
        
    return datapoint

if __name__ == '__main__':

    
    neighbors_data = NeighborsDataset()

    dataloader = DataLoader(neighbors_data, batch_size=1, num_workers=30, collate_fn=default_data_collate)
    dataset = []
    for data in tqdm(dataloader):
        dataset.extend(data)
        
    with open(DATA_PATH / 'pairs.json', 'w') as fp:
        json.dump(dataset, fp)
        
    # with open(DATA_PATH / 'train_split.json', 'r') as fp:
    #     train_split = json.load(fp)
    
    # with gzip.open(DATA_PATH / 'neighbors.gzip', 'rt') as fp:
    #     neighbors = json.load(fp)
    
    # train_datapoints = sorted([f[:-4] for f in train_split['train_uis']])
    # dataset = list(filter(lambda x: x['filename'] in train_datapoints, dataset))
    # dataset = sorted(dataset, key=lambda x: x['filename'])
    # dataset_indexes = list(range(len(dataset)))
    # filename2idx = {x['filename']: i for i, x in enumerate(dataset)}
    
    # neighbors_ids = {
    #     filename2idx[data['anchor']]: {
    #         'closest': [filename2idx[name] for name in data['closest']],
    #         'farthest': [filename2idx[name] for name in data['farthest']]
    #     }
    #     for data in tqdm(neighbors)
    # }
    
    # anchors = [{d: neighbors_ids[d]} for d in dataset_indexes if d in neighbors_ids]
    
    # output = np.memmap(DATA_PATH / 'pairs.npy', dtype='object', shape=len(anchors), mode='w+')
    # comput_func = partial(compute_pairs, dataset, output)
    # results = Parallel(n_jobs=30)(delayed(comput_func)(idx, neigh) for idx, neigh in tqdm(enumerate(anchors), total=len(anchors)))
    
    # # with Pool(30) as p:
    # #     results = list(tqdm(p.imap(comput_func, dataset_indexes), total=len(dataset_indexes)))
         
    # with open(DATA_PATH / 'pairs.json', 'w') as fp:
    #     json.dump(results, fp)