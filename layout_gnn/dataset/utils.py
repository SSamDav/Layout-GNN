from typing import Dict, Iterable


def get_labels_mapping(labels: Iterable[str]) -> Dict[str, int]:
    """Returns a dict that maps each label in the iterable into consecutive ints."""
    return {label: i for i, label in enumerate(sorted(set(labels)))}
