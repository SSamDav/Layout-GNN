from typing import Any, Dict, Tuple
import networkx as nx
import numpy as np

from layout_gnn.utils import draw_class_image


def node_subst_cost(node1: Dict[str, Any], node2: Dict[str, Any]) -> float:
    """Computes the cost of two nodes being different.

    Args:
        node1 (Dict[str, Any]): First node.
        node2 (Dict[str, Any]): Seconde node.

    Returns:
        float: Cost.
    """    
    if node1['label'] == node2['label']:
        return 0

    return 1


def node_del_cost(node: Dict[str, Any]) -> float:
    """Cost of removing a node.

    Args:
        node (Dict[str, Any]): Node to be removed.

    Returns:
        float: Cost.
    """    
    return 1  # here you apply the cost for node deletion


def node_ins_cost(node: Dict[str, Any]) -> float:
    """Cost of adding a node.

    Args:
        node (Dict[str, Any]): Node to be removed.

    Returns:
        float: Cost.
    """ 
    return 1  # here you apply the cost for node insertion


# arguments for edges
def edge_subst_cost(edge1: Dict[str, Any], edge2: Dict[str, Any]) -> float:
    """Computes the cost of two edges being different.

    Args:
        edge1 (Dict[str, Any]): First edges.
        edge2 (Dict[str, Any]): Seconde edges.

    Returns:
        float: Cost.
    """ 
    # check if the edges are equal, if yes then apply no cost, else apply 3
    if edge1['label']==edge2['label']:
        return 0
    
    return 1


def edge_del_cost(edge: Dict[str, Any]) -> float:
    """Cost of removing a edge.

    Args:
        edge (Dict[str, Any]): Edge to be removed.

    Returns:
        float: Cost.
    """ 
    return 0  # here you apply the cost for edge deletion


def edge_ins_cost(edge: Dict[str, Any]) -> float:
    """Cost of adding a edge.

    Args:
        edge (Dict[str, Any]): Edge to be removed.

    Returns:
        float: Cost.
    """
    return 0  # here you apply the cost for edge insertion


def compute_edit_distance(G1, G2) -> Dict[str, float]:
    """Compute the Graph edit distance of two Graphs.

    Args:
        G1 (_type_): First graph to compute the distance.
        G2 (_type_): Second graph to compute the distance.

    Returns:
         Dict[str, float]: Graph edit distance and normalized distance
    """    
    cost = nx.graph_edit_distance(
        G1,
        G2,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=edge_del_cost,
        edge_ins_cost=edge_ins_cost
    )
    
    normalized_cost = cost / (G1.number_of_nodes() + G2.number_of_nodes())
    
    return {'edit_distance': cost, 'normalized_edit_distance': normalized_cost}


def compute_iou(datapoint1: Dict[str, Any], datapoint2: Dict[str, Any], resolution: Tuple[int, int] = (256, 256)) -> float:
    """Computes the Intersection over Union of two datapoints.

    Args:
        datapoint1 (Dict[str, Any]): First datapoint.
        datapoint2 (Dict[str, Any]): Second datapoint.
        resolution (Tuple[int, int], optional): Resolution used to compute the IoU. Defaults to (256, 256).

    Returns:
        float: IoU
    """    
    node_labels = sorted(list(set(node['label'] for _, node in datapoint1['graph'].nodes(data=True)) | 
                              set(node['label'] for _, node in datapoint2['graph'].nodes(data=True))))
    node_labels = {label: idx for idx, label in enumerate(node_labels)}
    
    img1 = draw_class_image(resolution, node_labels, datapoint1['data'])
    img2 = draw_class_image(resolution, node_labels, datapoint2['data'])
    
    intersection = img1*img2
    union = np.clip(img1 + img2, a_max=1, a_min=0)
    indexes = np.where(union != 0)
    iou = intersection[indexes].mean()
    
    return iou