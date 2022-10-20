from typing import Any, Dict, Iterable, Optional

import networkx as nx

from layout_gnn.dataset.utils import get_labels_mapping


def add_networkx(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Adds a networkx graph.

    Args:
        sample (Dict[str, Any]): Sample to be processed.

    Returns:
        Dict[str, Any]: Processed sample.
    """    
    def convert_networkx(root: Dict[str, Any], graph: nx.DiGraph, parent_node: Optional[int] = None) -> None:
        current_node = graph.number_of_nodes() + 1
        graph.add_node(current_node, bbox=root['bbox'], label=root['label'])
        if parent_node:
            graph.add_edge(parent_node, current_node, label='parent_of')
            graph.add_edge(current_node, parent_node, label='child_of')
        
        parent_node = current_node
        for child in root.get('children', ()):
            convert_networkx(child, graph, parent_node)
            
    G = nx.DiGraph()
    convert_networkx(sample['data'], graph=G, parent_node=None)
    return {
        **sample,
        'graph': G
    }


class ConvertLabelsToIndexes:
    """Converts the "label" attributes in nodes and edges from a string to an int, according to the given mappings."""
    def __init__(
        self, 
        node_labels: Optional[Iterable[str]] = None, 
        edge_labels: Optional[Iterable[str]] = None,
    ):
        self.node_label_mappings = get_labels_mapping(node_labels) if node_labels is not None else None
        self.edge_label_mappings = get_labels_mapping(edge_labels) if edge_labels is not None else None

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        g: nx.DiGraph = sample["graph"]

        if self.node_label_mappings:
            for _, attrs in g.nodes(data=True):
                attrs["label"] = self.node_label_mappings.get(attrs["label"], len(self.node_label_mappings))
        if self.edge_label_mappings:
            for _, _, attrs in g.edges(data=True):
                attrs["label"] = self.edge_label_mappings.get(attrs["label"], len(self.edge_label_mappings))

        return sample
