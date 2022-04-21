import json
import random
from pathlib import Path

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

from functools import partial
from layout_gnn.dataset.dataset import RICOSemanticAnnotationsDataset
from layout_gnn.dataset import transformations
from layout_gnn.similarity_metrics import compute_edit_distance, compute_iou
from layout_gnn.utils import *
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

ROOT_PATH = Path.cwd()
DATA_PATH = ROOT_PATH / '../data'
DATA_PATH.mkdir(parents=True, exist_ok=True)

IOU_NEG_THRESH = lambda x: x < 0.32
IOU_POS_THRESH = lambda x: x > 0.50


GED_NEG_THRESH = lambda x: x > 0.66
GED_POS_THRESH = lambda x: x < 0.35


def comput_pairs(dataset, anchor_dataset_idx):
    datapoint = {
        'anchor': dataset[anchor_dataset_idx]['filename']
    }
    
    pair_indexes = sorted([i for i in range(len(dataset)) if i != anchor_dataset_idx], key=lambda k: random.random())
    for idx in pair_indexes:
        
        if 'neg_iou' not in datapoint or 'pos_iou' not in datapoint:
            iou_metric = compute_iou(dataset[anchor_dataset_idx], dataset[idx])
            
            if IOU_NEG_THRESH(iou_metric['iou']) and 'neg_iou' not in datapoint:
                datapoint['neg_iou'] = {
                    'pair': dataset[idx]['filename'],
                    **iou_metric
                }

            if IOU_POS_THRESH(iou_metric['iou']) and 'pos_iou' not in datapoint:
                datapoint['pos_iou'] = {
                    'pair': dataset[idx]['filename'],
                    **iou_metric
                }
        if 'neg_ged' not in datapoint or 'pos_ged' not in datapoint:   
            ged_metric = compute_edit_distance(dataset[anchor_dataset_idx]['graph'].to_undirected(), dataset[idx]['graph'].to_undirected())
            if GED_NEG_THRESH(ged_metric['normalized_edit_distance']) and 'neg_ged' not in datapoint:
                datapoint['neg_ged'] = {
                    'pair': dataset[idx]['filename'],
                    **ged_metric
                }

            if GED_NEG_THRESH(ged_metric['normalized_edit_distance']) and 'pos_ged' not in datapoint:
                datapoint['pos_ged'] = {
                    'pair': dataset[idx]['filename'],
                    **ged_metric
                }

        if len(datapoint) == 5:
            return datapoint
        
    return datapoint

if __name__ == '__main__':

    rico_dataset = RICOSemanticAnnotationsDataset(
        transform=transforms.Compose([
            transformations.process_data,
            transformations.normalize_bboxes,
            transformations.add_networkx,
        ]),
        only_data=True
    )

    dataloader = DataLoader(rico_dataset, batch_size=1, num_workers=16, collate_fn=default_data_collate)
    dataset = []
    for data in tqdm(dataloader):
        dataset.extend(data)
        
    with open(DATA_PATH / 'train_split.json', 'r') as fp:
        train_split = json.load(fp)
        
    train_datapoints = [f[:-4] for f in train_split['train_uis']]
    
    dataset = list(filter(lambda x: x['filename'] in train_datapoints, dataset))
    dataset = sorted(dataset, key=lambda x: x['filename'])
    
    dataset_indexes = list(range(len(dataset)))
    comput_func = partial(comput_pairs, dataset)
    sub_set = dataset_indexes[:10000]
    with Pool(30) as p:
        results = list(tqdm(p.imap(comput_func, sub_set), total=len(sub_set)))
         
    with open(DATA_PATH / 'pairs_0_10000.json', 'w') as fp:
        json.dump(results, fp)