from typing import Any, Dict


def process_data(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Process the raw data by selection only the usefull fields.

    Args:
        sample (Dict[str, Any]): Raw sample.

    Returns:
        Dict[str, Any]: Processed sample.
    """    
    def process_tree(root: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'bbox': root['bounds'],
            'label': root.get('componentLabel', 'No Label'),
            'children': [process_tree(child) for child in root.get('children', ())],
            'icon_label': root.get('iconClass', None),
            'text_button_label': root.get('textButtonClass', None)
        }
    
    return {
        **sample,
        'data': process_tree(sample['data'])
    }
    
    
def normalize_bboxes(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes the bounding boxes.

    Args:
        sample (Dict[str, Any]): Sample to be processed.

    Returns:
        Dict[str, Any]: Processed sample.
    """    
    w_factor = 1 / sample['data']['bbox'][2]
    h_factor = 1 / sample['data']['bbox'][3]

    def normalize_bbox(root: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **root,
            'bbox': [
                root['bbox'][0] * w_factor,
                root['bbox'][1] * h_factor,
                root['bbox'][2] * w_factor,
                root['bbox'][3] * h_factor
            ],
            'children': [normalize_bbox(child) for child in root.get('children', ())]
        }
    
    return {
        **sample,
        'data': normalize_bbox(sample['data'])
    }
