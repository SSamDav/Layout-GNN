from typing import Any, Dict, Optional, Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt


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
    stack = [root]
    while stack:
        current_node = stack.pop(0)
        stack.extend(current_node['children'])
        
        w, h = current_node['bbox'][2] - current_node['bbox'][0], current_node['bbox'][3] - current_node['bbox'][1]
        color = color_label_map.get(current_node['label'], None)
        if color:
            rect = patches.Rectangle((current_node['bbox'][0], current_node['bbox'][1]), 
                                      w, 
                                      h, 
                                      facecolor=color['hex'],
                                      edgecolor='w',
                                      linewidth=1)
            ax.add_patch(rect)
        

def default_data_collate(batch):
    return batch


def draw_class_image(image_shape: Tuple[int, int],
                     node_labels: Dict[str, Any],
                     datapoint: Dict[str, Any],
                     img_class: Optional[npt.ArrayLike] = None) -> np.ndarray:
    """Creates an class image, one hot image where the 3rd dimention correspond to class labels.

    Args:
        image_shape (Tuple[int, int]): Image shape.
        node_labels (Dict[str, Any]): Mapping label to int.
        datapoint (Dict[str, Any]): Datapoint used to compute the class image.
        img_class (Optional[npt.ArrayLike], optional): Current image class. Defaults to None.

    Returns:
        np.ndarray: Image class.
    """    
    x0, y0, x1, y1 = datapoint['bbox']
    x0, x1 = int(image_shape[0]*x0), int(image_shape[0]*x1)
    y0, y1 = int(image_shape[1]*y0), int(image_shape[1]*y1)
    
    if img_class is None:
        img_class = np.zeros((*image_shape, len(node_labels)))
        
    label_idx = node_labels[datapoint['label']]
    img_class[y0:y1, x0:x1, label_idx] = 1
    
    for child in  datapoint.get('children', []):
        img_class = draw_class_image(image_shape, node_labels, child, img_class=img_class)
        
    return img_class
