from typing import Any, Dict, List, Union

import torch
from torch_geometric.data import Batch


def default_data_collate(batch):
    return batch


def pyg_data_collate(batch: List[Dict[str, Any]]) -> Batch:
    """Creates a torch_geometric Batch from a list of samples."""
    return Batch.from_data_list([sample["graph"] for sample in batch])


def pyg_triplets_data_collate(batch: List[Dict[str, Any]]) -> Dict[str, Union[Batch, torch.Tensor]]:
    """Creates a dict with a batch of anchor, positive and negative graphs and the anchor image."""
    collated_batch = {}
    for k in ("anchor", "pos", "neg"):
        collated_batch[k] = Batch.from_data_list([sample[k]["graph"] for sample in batch])

    if "image" in batch[0]["anchor"]:
        collated_batch["image"] = torch.stack(
            [torch.as_tensor(sample["anchor"]["image"], dtype=torch.float) for sample in batch]
        )

    return collated_batch
