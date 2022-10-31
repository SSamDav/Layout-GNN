from typing import Any, Dict, Tuple

import numpy as np

from layout_gnn.utils import draw_class_image


def intersection_over_union(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.logical_and(x1, x2).sum() / np.logical_or(x1, x2).sum()


def compute_iou(datapoint1: Dict[str, Any], datapoint2: Dict[str, Any], resolution: Tuple[int, int] = (256, 256)) -> Dict[str, float]:
    """Computes the Intersection over Union of two datapoints.

    Args:
        datapoint1 (Dict[str, Any]): First datapoint.
        datapoint2 (Dict[str, Any]): Second datapoint.
        resolution (Tuple[int, int], optional): Resolution used to compute the IoU. Defaults to (256, 256).

    Returns:
        Dict[str, float]: IoU
    """    
    node_labels = sorted(set(node['label'] for _, node in datapoint1['graph'].nodes(data=True)) | 
                         set(node['label'] for _, node in datapoint2['graph'].nodes(data=True)))
    node_labels = {label: idx for idx, label in enumerate(node_labels)}
    
    img1 = draw_class_image(image_shape=resolution, node_labels=node_labels, root=datapoint1['data'])
    img2 = draw_class_image(image_shape=resolution, node_labels=node_labels, root=datapoint2['data'])
    
    return {'iou': intersection_over_union(img1, img2)}
