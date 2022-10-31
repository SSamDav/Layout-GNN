from typing import Any, Dict, Iterable, Tuple

from skimage import transform

from layout_gnn.dataset.utils import get_labels_mapping
from layout_gnn.utils import draw_class_image


class RescaleImage:
    """Recales the image to a specified width and height.
    """    
    def __init__(self, width: int, height: int, allow_missing_image: bool = False):
        self.w = width
        self.h = height
        self._allow_missing_image = allow_missing_image
        
    def __call__(self, sample):
        if self._allow_missing_image and "image" not in sample:
            return sample

        return {
            **sample,
            'image': transform.resize(sample['image'], (self.h, self.w))
        }


class DrawClassImage:
    def __init__(self, node_labels: Iterable[str], image_shape: Tuple[int, int] = (256, 256)) -> None:
        self.node_labels = get_labels_mapping(node_labels)
        self.image_shape = image_shape
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["class_image"] = draw_class_image(
            root=sample["data"],
            node_labels=self.node_labels,
            image_shape=self.image_shape,
        )
        return sample