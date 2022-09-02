import multiprocessing

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger
import torch.cuda as cuda
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool, GIN
from torchvision import transforms

from layout_gnn.dataset.dataset import DATA_PATH, RICOSemanticAnnotationsDataset
from layout_gnn.dataset import transformations
from layout_gnn.lightning_module.simgrace import LayoutGNNSimGRACENTXent
from layout_gnn.utils import pyg_data_collate


# TODO: Move to a config file
# Dataset and data loader arguments
BATCH_SIZE = 32
IMAGE_SIZE = 64
EPOCHS = 100
NUM_WORKERS = multiprocessing.cpu_count()
# Encoder (GNN) arguments
LABEL_EMBEDDING_DIM = 64
BBOX_EMBEDDING_LAYER_DIMS = [32, 64]
GNN_HIDDEN_CHANNELS = 300
GNN_OUT_CHANNELS = None
GNN_NUM_LAYERS = 5
GNN_MODEL_CLS = GIN
USE_EDGE_ATTR = False
EDGE_LABEL_EMBEDDING_DIM = 16
READOUT = global_mean_pool
# Projection head (MLP) arguments
PROJECTION_HEAD_DIMS = (300, 300)
# SimGRACE parameters
PERTURBATION_MAGNITUDE = 1.
# Loss/optimizer parameters
TEMPERATURE = 0.5
LR = 0.001


if __name__ == "__main__":
    dataset = RICOSemanticAnnotationsDataset(root_dir=DATA_PATH, only_data=True)
    label_mappings = {k: i for i, k in enumerate(dataset.label_color_map)}
    dataset.transform = transforms.Compose([
        transformations.process_data,
        transformations.normalize_bboxes,
        transformations.add_networkx,
        transformations.RescaleImage(IMAGE_SIZE, IMAGE_SIZE, allow_missing_image=True),
        transformations.ConvertLabelsToIndexes(
            node_label_mappings=label_mappings,
            edge_label_mappings={"parent_of": 0, "child_of": 1} if USE_EDGE_ATTR else None,
        ),
        transformations.convert_graph_to_pyg,
    ])
    
    data_loader = DataLoader(
        dataset=dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=pyg_data_collate, 
        num_workers=NUM_WORKERS,
        shuffle=True,
        persistent_workers=True
    )
    model = LayoutGNNSimGRACENTXent(
        num_labels=len(label_mappings) + 1,
        label_embedding_dim=LABEL_EMBEDDING_DIM,
        bbox_embedding_layer_dims=BBOX_EMBEDDING_LAYER_DIMS,
        gnn_hidden_channels=GNN_HIDDEN_CHANNELS,
        gnn_num_layers=GNN_NUM_LAYERS,
        gnn_out_channels=GNN_OUT_CHANNELS,
        gnn_model_cls=GNN_MODEL_CLS,
        use_edge_attr=USE_EDGE_ATTR,
        num_edge_labels=2,
        edge_label_embedding_dim=EDGE_LABEL_EMBEDDING_DIM,
        readout=READOUT,
        perturbation_magnitude=PERTURBATION_MAGNITUDE,
        projection_head_dims=PROJECTION_HEAD_DIMS,
        lr=LR,
        temperature=TEMPERATURE,
    )
    aim_logger = AimLogger(
        experiment='HierarchicalLayoutGNN',
        train_metric_prefix='train_',
        test_metric_prefix='test_',
        val_metric_prefix='val_',
    )

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if cuda.is_available() else None,
        default_root_dir=DATA_PATH,
        logger=aim_logger,
        callbacks=[ModelCheckpoint(dirpath=DATA_PATH / 'model')],
    )
    trainer.fit(model, data_loader)
