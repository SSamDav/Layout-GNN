import json
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
from google.cloud import storage
from skimage import io
from torch.utils.data import Dataset

BUCKET_ID = 'crowdstf-rico-uiuc-4540'
BUCKET_PATH = 'rico_dataset_v0.1/semantic_annotations.zip'
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / 'data'


class RICOSemanticAnnotationsDataset(Dataset):
    def __init__(self, root_dir: Optional[Path] = None, transform=None, only_data: bool = False):
        self.data_path = root_dir or DATA_PATH
        self.transform = transform
        self._only_data = only_data
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        zip_filename = self.data_path / 'semantic_annotations.zip'
        if not zip_filename.exists():
            client = storage.Client.create_anonymous_client()
            bucket = client.bucket(BUCKET_ID)
  
            blob = bucket.blob('rico_dataset_v0.1/semantic_annotations.zip')
            blob.download_to_filename(zip_filename)
        
        extracted_folder = self.data_path / 'semantic_annotations'
        if not extracted_folder.exists():
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)

        self._file_with_errors = json.load(open(self.data_path / 'file_with_error.json', 'r'))
        self.files = list(f for f in sorted((extracted_folder).glob('*.json')) if f.stem not in self._file_with_errors)
        self.label_color_map = json.load(open(self.data_path / 'component_legend.json', 'r'))
        self.icon_label_color_map = json.load(open(self.data_path / 'icon_legend.json', 'r'))
        self.text_button_color_map = json.load(open(self.data_path / 'textButton_legend.json', 'r'))
        
        self.label_color_map['Icon'] = next(iter(self.icon_label_color_map.values()))
        self.label_color_map['Text Button'] = next(iter(self.text_button_color_map.values()))
        
        
    def __len__(self):
        return len(self.files)
    
    def _get_item(self, idx: int, only_data: Optional[bool] = None):
        if only_data is None:
            only_data = self._only_data
        
        with open(self.files[idx], 'r') as fp:
            json_file = json.load(fp)
            
        sample = {'data': json_file, 'filename': self.files[idx].stem}
        if not only_data:
            image = io.imread(self.files[idx].with_suffix('.png'))[:,: , :3] # Removing the Alpha channel
            sample['image'] = image            
            
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __getitem__(self, idx):
        return self._get_item(idx)


class RICOTripletsDataset(RICOSemanticAnnotationsDataset):
    VALID_TRIPLET_METRICS = {"iou", "ged"}

    def __init__(self, triplets: Union[Sequence[Dict[str, str]], str, Path], triplet_metric: str = "ged", **kwargs):
        if triplet_metric not in self.VALID_TRIPLET_METRICS:
            raise ValueError(
                f"Invalid value {triplet_metric} for argument `triplet_metric`. "
                f"Must be one of {self.VALID_TRIPLET_METRICS}"
            )

        super().__init__(**kwargs)
        if isinstance(triplets, (str, Path)):
            with open(triplets) as f:
                triplets = json.load(f)
        self.triplets = triplets
        self.triplet_metric = triplet_metric

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return {
            "anchor": self._get_item(int(self.triplets[idx]["anchor"])),
            "pos": self._get_item(int(self.triplets[idx][f"pos_{self.triplet_metric}"]["pair"]), only_data=True),
            "neg": self._get_item(int(self.triplets[idx][f"neg_{self.triplet_metric}"]["pair"]), only_data=True),
        }