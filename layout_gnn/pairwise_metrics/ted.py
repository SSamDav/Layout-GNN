from typing import Any, Dict, List, Optional
import apted

from layout_gnn.utils import get_num_nodes


class TreeEditDistanceConfig(apted.Config):
    def rename(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> int:
        return int(node1["label"] != node2["label"])

    def children(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        return node.get("children", [])


def compute_edit_distance(
    tree1: Dict[str, Any],
    tree2: Dict[str, Any],
    config: Optional[TreeEditDistanceConfig] = None,
    use_num_nodes_cache: bool = False,
) -> Dict[str, float]:
    config = config or TreeEditDistanceConfig()
    distance = apted.APTED(tree1, tree2, config=config).compute_edit_distance()
    num_nodes = get_num_nodes(tree1, use_cache=use_num_nodes_cache) + get_num_nodes(tree2, use_cache=use_num_nodes_cache)

    return {"edit_distance": distance, "normalized_edit_distance": distance/num_nodes}