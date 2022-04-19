import networkx as nx
import numpy as np

from layout_gnn.utils import draw_class_image


def node_subst_cost(node1, node2):
    # check if the nodes are equal, if yes then apply no cost, else apply 1
    if node1['label'] == node2['label']:
        return 0

    return 1


def node_del_cost(node):
    return 1  # here you apply the cost for node deletion


def node_ins_cost(node):
    return 1  # here you apply the cost for node insertion


# arguments for edges
def edge_subst_cost(edge1, edge2):
    # check if the edges are equal, if yes then apply no cost, else apply 3
    if edge1['label']==edge2['label']:
        return 0
    
    return 1


def edge_del_cost(node):
    return 0  # here you apply the cost for edge deletion


def edge_ins_cost(node):
    return 0  # here you apply the cost for edge insertion


def compute_edit_distance(G1, G2):
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


def compute_iou(datapoint1, datapoint2, resolution=(256, 256)):
    node_labels = sorted(list(set(node['label'] for _, node in datapoint1['graph'].nodes(data=True)) | 
                              set(node['label'] for _, node in datapoint2['graph'].nodes(data=True))))
    node_labels = {label: idx for idx, label in enumerate(node_labels)}
    
    img1 = draw_class_image(resolution, node_labels, datapoint1['data'])
    img2 = draw_class_image(resolution, node_labels, datapoint2['data'])
    
    intersection = img1*img2
    union = np.clip(img1 + img2, a_max=1, a_min=0)
    indexes = np.where(union != 0)
    iou = (intersection[indexes] / union[indexes]).mean()
    
    return iou