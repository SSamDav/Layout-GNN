import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_datapoint(datapoint, color_label_map, color_node_map):
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    axes[1].invert_yaxis()
    draw_screen(datapoint['data'], axes[1], color_label_map)
    axes[2].imshow(datapoint['image'])
    nx.draw(datapoint['graph'], node_color=color_node_map, ax=axes[0])
    return fig
    
    
def draw_screen(root, ax, color_label_map, queue=[]):
    queue.extend(root.get('children', []))
    while queue:
        child = queue.pop(0)
    
        w, h = child['bbox'][2] - child['bbox'][0], child['bbox'][3] - child['bbox'][1]
        rect = patches.Rectangle((child['bbox'][0], child['bbox'][1]), w, h, facecolor=color_label_map[child['label']])
        ax.add_patch(rect)
        draw_screen(child, ax, color_label_map, queue=queue)
        

def default_data_collate(batch):
    return batch


def draw_class_image(image_shape, node_labels, datapoint, img_class=None):
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
