from typing import Any, Dict

from torch_geometric.utils import from_networkx


def convert_graph_to_pyg(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Converts the networkx graph to a torch_geometric graph."""
    sample["graph"] = from_networkx(sample["graph"])
    return sample
