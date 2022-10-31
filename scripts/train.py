import argparse
import multiprocessing

import torch.cuda as cuda
import torch.nn.functional as F
import torch_geometric.nn
import yaml
from aim.pytorch_lightning import AimLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torchvision import transforms

from layout_gnn import lightning_module
from layout_gnn.dataset import collate
from layout_gnn.dataset import dataset as dataset_module
from layout_gnn.dataset.transforms.core import process_data, normalize_bboxes
from layout_gnn.dataset.transforms.image import RescaleImage
from layout_gnn.dataset.transforms.nx import ConvertLabelsToIndexes, add_networkx
from layout_gnn.dataset.transforms.pyg import convert_graph_to_pyg

# TODO: Move to a config file
# Dataset and data loader arguments
NUM_WORKERS = multiprocessing.cpu_count()
IMAGE_SIZE = 64


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train GNN models on RICO dataset.")
    parser.add_argument("-c", "--config", help="Path to the config file.", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    with open(args.config) as fp:
        config = yaml.safe_load(fp)
        
    seed_everything(config['seed'])
    
    dataset_cls = getattr(dataset_module, config['dataset_cls'])
    dataset = dataset_cls(root_dir=dataset_module.DATA_PATH, **config['dataset_config'])
    dataset.transform = transforms.Compose([
        process_data,
        normalize_bboxes,
        add_networkx,
        RescaleImage(IMAGE_SIZE, IMAGE_SIZE, allow_missing_image=True),
        ConvertLabelsToIndexes(
            node_labels=dataset.label_color_map,
            edge_labels=("parent_of", "child_of") if config['model_config']['use_edge_attr'] else None,
        ),
        convert_graph_to_pyg,
    ])
    
    config['dataloader_config']['collate_fn'] = getattr(collate , config['dataloader_config']['collate_fn'])
    data_loader = DataLoader(
       **{
            'dataset': dataset, 
            'num_workers': NUM_WORKERS,
            'shuffle': True,
            'persistent_workers': True,
             **config['dataloader_config']
       }   
    )
    
    
    model_cls = getattr(lightning_module, config['model_cls'])
    config['model_config']['gnn_model_cls'] =  getattr(torch_geometric.nn, config['model_config']['gnn_model_cls'])
    config['model_config']['readout'] =  getattr(torch_geometric.nn, config['model_config']['readout'])
    if config['model_config'].get('triplet_loss_distance_function', False):
        config['model_config']['triplet_loss_distance_function'] = lambda x1, x2: 1 - F.cosine_similarity(x1, x2)
        
    model = model_cls(
        **{
            'num_labels': len(dataset.label_color_map) + 1,
            'num_edge_labels': 2,
            **config['model_config']
        }
    )
    aim_logger = AimLogger(
        experiment='HierarchicalLayoutGNN',
        train_metric_prefix='train_',
        test_metric_prefix='test_',
        val_metric_prefix='val_',
    )

    trainer = Trainer(
        accelerator="gpu" if cuda.is_available() else None,
        default_root_dir=dataset_module.DATA_PATH,
        logger=aim_logger,
        callbacks=[
            ModelCheckpoint(
                filename='{epoch}-{loss:.2f}',
                dirpath=dataset_module.DATA_PATH / config['model_name'],
            )
        ],
        **config.get('trainer_config', {}),
    )
    trainer.fit(model, data_loader)
