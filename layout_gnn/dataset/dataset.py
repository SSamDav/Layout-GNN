import json
import zipfile
from pathlib import Path
from typing import Optional

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
        self._only_bool = only_data
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        zip_filename = self.data_path / 'semantic_annotations.zip'
        if not zip_filename.exists():
            client = storage.Client.create_anonymous_client()
            bucket = client.bucket(BUCKET_ID)
  
            blob = bucket.blob('rico_dataset_v0.1/semantic_annotations.zip')
            blob.download_to_filename(zip_filename)
        
        self._file_with_errors = json.load(open(self.data_path / 'file_with_error.json', 'r'))
        self.files = list(f for f in sorted((self.data_path / 'semantic_annotations').glob('*.json')) if f.stem not in self._file_with_errors)
        self.label_color_map = json.load(open(self.data_path / 'component_legend.json', 'r'))
        self.icon_label_color_map = json.load(open(self.data_path / 'icon_legend.json', 'r'))
        self.text_button_color_map = json.load(open(self.data_path / 'textButton_legend.json', 'r'))
        
        icon_rgb = icon_rgb = np.asarray([v['rgb'] for v in self.icon_label_color_map.values()]).mean(axis=0).astype(int)
        self.label_color_map['Icon'] = {
            'hex': '#%02x%02x%02x' % (icon_rgb[0], icon_rgb[1], icon_rgb[2]),
            'rgb': [icon_rgb[0], icon_rgb[1], icon_rgb[2]]
        }
        text_button_rgb = np.asarray([v['rgb'] for v in self.text_button_color_map.values()]).mean(axis=0).astype(int)
        self.label_color_map['Text Button'] = {
            'hex': '#%02x%02x%02x' % (text_button_rgb[0], text_button_rgb[1], text_button_rgb[2]),
            'rgb': [text_button_rgb[0], text_button_rgb[1], text_button_rgb[2]]
        }
        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        with open(self.files[idx], 'r') as fp:
            json_file = json.load(fp)
            
        sample = {'data': json_file, 'filename': self.files[idx].stem}
        if not self._only_bool:
            image = io.imread(self.files[idx].with_suffix('.png'))[:,: , :3] # Removing the Alpha channel
            sample['image'] = image            
            
        if self.transform:
            sample = self.transform(sample)
        
        return sample
