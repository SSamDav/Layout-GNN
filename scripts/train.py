import multiprocessing

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool, GCN
from torchvision import transforms

from layout_gnn.dataset.dataset import RICOTripletsDataset, DATA_PATH
from layout_gnn.dataset import transformations
from layout_gnn.model.lightning_module import LayoutGraphModelCNNNeuralRasterizer
from layout_gnn.utils import pyg_triplets_data_collate


# TODO: Move to a config file
# Dataset and data loader arguments
TRIPLETS_FILENAME = "pairs_0_10000.json"
BATCH_SIZE = 1
IMAGE_SIZE = 64
NUM_WORKERS = multiprocessing.cpu_count()
# Encoder (GNN) arguments
LABEL_EMBEDDING_DIM = 64
BBOX_EMBEDDING_LAYER_DIMS = [32, 64]
GNN_HIDDEN_CHANNELS = 256
GNN_OUT_CHANNELS = None
GNN_NUM_LAYERS = 4
GNN_MODEL_CLS = GCN
USE_EDGE_ATTR = False
EDGE_LABEL_EMBEDDING_DIM = 16
READOUT = lambda x, inputs: global_mean_pool(x, batch=inputs.batch)
# Decoder (CNN) arguments
CNN_HIDDEN_DIM = 16
TRIPLET_LOSS_DISTANCE_FUNCTION = lambda x1, x2: 1 - F.cosine_similarity(x1, x2)
# Loss/optimizer parameters
TRIPLET_LOSS_MARGIN = 1.
RECONSTRUCTION_LOSS_WEIGHT = 1
LR = 0.001


if __name__ == "__main__":
    dataset = RICOTripletsDataset(triplets=DATA_PATH / TRIPLETS_FILENAME)
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
        collate_fn=pyg_triplets_data_collate, 
        num_workers=NUM_WORKERS,
    )
    model = LayoutGraphModelCNNNeuralRasterizer(
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
        cnn_output_dim=3,
        cnn_hidden_dim=CNN_HIDDEN_DIM,
        cnn_output_size=IMAGE_SIZE,
        triplet_loss_distance_function=TRIPLET_LOSS_DISTANCE_FUNCTION,
        triplet_loss_margin=TRIPLET_LOSS_MARGIN,
        reconstruction_loss_weight=RECONSTRUCTION_LOSS_WEIGHT,
        lr=LR,
    )
    aim_logger = AimLogger(
        experiment='HierarchicalLayoutGNN',
        train_metric_prefix='train_',
        test_metric_prefix='test_',
        val_metric_prefix='val_',
    )
    trainer = Trainer(
        default_root_dir=DATA_PATH,
        logger=aim_logger,
        callbacks=[ModelCheckpoint()]
    )
    trainer.fit(model, data_loader)
