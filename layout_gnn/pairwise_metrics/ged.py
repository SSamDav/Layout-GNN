from typing import Any, Dict, Optional

import networkx as nx


class GraphEditDistanceConfig:
    def node_subst_cost(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> float:
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

    def node_del_cost(self, node: Dict[str, Any]) -> float:
        """Cost of removing a node.

        Args:
            node (Dict[str, Any]): Node to be removed.

        Returns:
            float: Cost.
        """    
        return 1  # here you apply the cost for node deletion

    def node_ins_cost(self, node: Dict[str, Any]) -> float:
        """Cost of adding a node.

        Args:
            node (Dict[str, Any]): Node to be removed.

        Returns:
            float: Cost.
        """ 
        return 1  # here you apply the cost for node insertion

    # arguments for edges
    def edge_subst_cost(self, edge1: Dict[str, Any], edge2: Dict[str, Any]) -> float:
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

    def edge_del_cost(self, edge: Dict[str, Any]) -> float:
        """Cost of removing a edge.

        Args:
            edge (Dict[str, Any]): Edge to be removed.

        Returns:
            float: Cost.
        """ 
        return 0  # here you apply the cost for edge deletion

    def edge_ins_cost(self, edge: Dict[str, Any]) -> float:
        """Cost of adding a edge.

        Args:
            edge (Dict[str, Any]): Edge to be removed.

        Returns:
            float: Cost.
        """
        return 0  # here you apply the cost for edge insertion


def compute_edit_distance(
    G1: nx.Graph,
    G2: nx.Graph,
    config: Optional[GraphEditDistanceConfig] = None,
) -> Dict[str, float]:
    """Compute the Graph edit distance of two Graphs.

    Args:
        G1 (_type_): First graph to compute the distance.
        G2 (_type_): Second graph to compute the distance.

    Returns:
        Dict[str, float]: Graph edit distance and normalized distance
    """
    config = config or GraphEditDistanceConfig()

    cost = nx.graph_edit_distance(
        G1,
        G2,
        node_subst_cost=config.node_subst_cost,
        node_del_cost=config.node_del_cost,
        node_ins_cost=config.node_ins_cost,
        edge_subst_cost=config.edge_subst_cost,
        edge_del_cost=config.edge_del_cost,
        edge_ins_cost=config.edge_ins_cost,
    )
        
    normalized_cost = cost / (G1.number_of_nodes() + G2.number_of_nodes())
    
    return {'edit_distance': cost, 'normalized_edit_distance': normalized_cost}
