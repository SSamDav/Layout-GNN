from typing import Any, Dict, Optional

import networkx as nx
from skimage import transform
from torch_geometric.utils import from_networkx


def process_data(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Process the raw data by selection only the usefull fields.

    Args:
        sample (Dict[str, Any]): Raw sample.

    Returns:
        Dict[str, Any]: Processed sample.
    """    
    def process_tree(root):
        return {
            'bbox': root['bounds'],
            'label': root.get('componentLabel', 'No Label'),
            'children': [process_tree(child) for child in root.get('children', [])],
            'icon_label': root.get('iconClass', None),
            'text_button_label': root.get('textButtonClass', None)
        }
    
    return {
        **sample,
        'data': process_tree(sample['data'])
    }
    
    
def normalize_bboxes(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes the bounding boxes.

    Args:
        sample (Dict[str, Any]): Sample to be processed.

    Returns:
        Dict[str, Any]: Processed sample.
    """    
    def normalize_bbox(root, w_factor, h_factor):
        return {
            **root,
            'bbox': [
                root['bbox'][0] * w_factor,
                root['bbox'][1] * h_factor,
                root['bbox'][2] * w_factor,
                root['bbox'][3] * h_factor
            ],
            'children': [normalize_bbox(child, w_factor, h_factor) for child in root.get('children', [])]
        }
        
    w_factor = 1 / sample['data']['bbox'][2]
    h_factor = 1 / sample['data']['bbox'][3]
    
    return {
        **sample,
        'data': normalize_bbox(sample['data'], w_factor, h_factor)
    }   
    
    
def add_networkx(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Adds a networkx graph.

    Args:
        sample (Dict[str, Any]): Sample to be processed.

    Returns:
        Dict[str, Any]: Processed sample.
    """    
    def convert_networkx(root, graph, parent_node):
        graph.add_node(graph.number_of_nodes() + 1, bbox=root['bbox'], label=root['label'])
        if parent_node:
            graph.add_edge(parent_node, graph.number_of_nodes(), label='parent_of')
            graph.add_edge(graph.number_of_nodes(), parent_node, label='child_of')
        
        parent = graph.number_of_nodes()
        for child in root.get('children', []):
            graph = convert_networkx(child, graph, parent)
            
        return graph
            
    G = nx.DiGraph()
    G = convert_networkx(sample['data'], graph=G, parent_node=None)
    return {
        **sample,
        'graph': G
    }
    
    
class RescaleImage:
    """Recales the image to a specified width and height.
    """    
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        
    def __call__(self, sample):
        return {
            **sample,
            'image': transform.resize(sample['image'], (self.h, self.w))
        }


class ConvertLabelsToIndexes:
    """Converts the "label" attributes in nodes and edges from a string to an int, according to the given mappings."""
    def __init__(
        self, 
        node_label_mappings: Optional[Dict[str, int]] = None, 
        edge_label_mappings: Optional[Dict[str, int]] = None,
    ):
        self.node_label_mappings = node_label_mappings
        self.edge_label_mappings = edge_label_mappings

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        g: nx.DiGraph = sample["graph"]

        if self.node_label_mappings:
            for _, attrs in g.nodes(data=True):
                attrs["label"] = self.node_label_mappings.get(attrs["label"], len(self.node_label_mappings))
        if self.edge_label_mappings:
            for _, _, attrs in g.edges(data=True):
                attrs["label"] = self.edge_label_mappings.get(attrs["label"], len(self.edge_label_mappings))

        return sample


def convert_graph_to_pyg(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Converts the networkx graph to a torch_geometric graph."""
    sample["graph"] = from_networkx(sample["graph"])
    return sample