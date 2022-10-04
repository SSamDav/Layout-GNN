from typing import Any, Dict

from torchvision.transforms import Compose

from layout_gnn.dataset.dataset import DATA_PATH, ENRICOSemanticAnnotationsDataset
from layout_gnn.dataset.transforms.core import normalize_bboxes, process_data
from layout_gnn.dataset.transforms.image import DrawClassImage
from layout_gnn.dataset.utils import get_labels_mapping
from layout_gnn.pairwise_metrics import iou, ted
from layout_gnn.pairwise_metrics.core import PairwiseMetricCalculator


DATASET_CLS = ENRICOSemanticAnnotationsDataset


def compute_iou(sample1: Dict[str, Any], sample2: Dict[str, Any]) -> float:
    return iou.intersection_over_union(sample1["class_image"], sample2["class_image"])


def compute_ted(sample1: Dict[str, Any], sample2: Dict[str, Any]) -> float:
    return ted.compute_edit_distance(sample1["data"], sample2["data"])["normalized_edit_distance"]


if __name__ == "__main__":
    # Compute TED between each pair in the dataset
    compute_teds = PairwiseMetricCalculator(
        dataset=DATASET_CLS(transform=process_data, only_data=True, cache_items=True), 
        distance_fn=compute_ted,
    )
    compute_teds.write_values_to_csv(filepath=DATA_PATH / f"{DATASET_CLS.__name__}_ted.csv", verbose=1)

    # Compute IOU between each pair in the dataset    
    dataset = DATASET_CLS(only_data=True, cache_items=True)
    dataset.transform = Compose([
        process_data,
        normalize_bboxes,
        DrawClassImage(node_labels=get_labels_mapping(dataset.label_color_map)),
    ])
    compute_ious = PairwiseMetricCalculator(
        dataset=dataset,
        distance_fn=compute_iou,
    )
    compute_ious.write_values_to_csv(filepath=DATA_PATH / f"{DATASET_CLS.__name__}_iou.csv", verbose=1)
