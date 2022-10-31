from collections import Counter
from typing import Any, Dict, Iterable

import numpy as np

from layout_gnn.utils import depth_first_traversal


class ComponentLabelHistogram:
    def __init__(self, labels: Iterable[str]) -> None:
        self.labels = sorted(labels)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        label_counts = Counter(node.get("componentLabel") for node in depth_first_traversal(sample["data"]))
        sample["histogram"] = np.asarray([label_counts[k] for k in self.labels])
        return sample
