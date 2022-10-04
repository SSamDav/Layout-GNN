from collections import deque
from typing import Any, Dict, Iterator, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_datapoint(datapoint: Dict[str, Any],
                   color_label_map: Dict[str, Dict[str, str]]) -> plt.figure:
    """Draws the graph, the screen rendering and the screen image (when available).

    Args:
        datapoint (Dict[str, Any]): Datapoint to be drawn.
        color_label_map (Dict[str, Dict[str, str]]): Color map for each label.

    Returns:
        plt.figure: Figure with all the drawings
    """    
    color_node_map = [
        color_label_map.get(data['label'], {'hex':'#000000'})['hex']
        for _, data in list(datapoint['graph'].nodes(data=True))
    ]
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    axes[1].invert_yaxis()
    draw_screen(datapoint['data'], axes[1], color_label_map)
    if datapoint.get('image', None) is not None:
        axes[2].imshow(datapoint['image'])
    nx.draw(datapoint['graph'], node_color=color_node_map, ax=axes[0])
    return fig
    

def draw_screen(root: Dict[str, Any],
                ax: plt.Axes, 
                color_label_map: Dict[str, Dict[str, Any]]):
    """Draw the screen using hierarchical data.

    Args:
        root (Dict[str, Any]): Root node to the hierarchical tree.
        ax (plt.Axes): Axe where the screen is drawn.
        color_label_map (Dict[str, Dict[str, Any]]): Color to be used for each label.
    """
    for node in breadth_first_traversal(root):
        w, h = node['bbox'][2] - node['bbox'][0], node['bbox'][3] - node['bbox'][1]
        color = color_label_map.get(node['label'], None)
        if color:
            rect = patches.Rectangle((node['bbox'][0], node['bbox'][1]), 
                                      w, 
                                      h, 
                                      facecolor=color['hex'],
                                      edgecolor='w',
                                      linewidth=1)
            ax.add_patch(rect)


def draw_class_image(
    root: Dict[str, Any],
    node_labels: Dict[str, int],
    image_shape: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Creates an class image, one hot image where the 3rd dimention correspond to class labels.

    Args:
        root (Dict[str, Any]): Root of the tree used to compute the class image.
        node_labels (Dict[str, Any]): Mapping label to int.
        image_shape (Tuple[int, int]): Image shape.

    Returns:
        np.ndarray: Image class.
    """    
    img_class = np.zeros((*image_shape, max(node_labels.values())+1), dtype=bool)

    for node in deph_first_traversal(root):
        label_idx = node_labels.get(node['label'])
        if label_idx is not None:
            x0, y0, x1, y1 = node['bbox']
            x0, x1 = int(image_shape[0]*x0), int(image_shape[0]*x1)
            y0, y1 = int(image_shape[1]*y0), int(image_shape[1]*y1)
            img_class[y0:y1, x0:x1, label_idx] = True

    return img_class


def deph_first_traversal(root: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    yield root
    for child in root.get("children", ()):
        yield from deph_first_traversal(child)


def breadth_first_traversal(root: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    stack = deque((root,))
    while stack:
        node = stack.popleft()
        stack.extend(node.get("children", ()))
        yield node


def get_num_nodes(root: Dict[str, Any], use_cache: bool = False) -> int:
    if not use_cache or "__num_nodes__" not in root:
        root["__num_nodes__"] = sum(1 for _ in deph_first_traversal(root))

    return root["__num_nodes__"]
