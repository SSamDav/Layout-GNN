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
    
    def __getitem__(self, idx):
        
        with open(self.files[idx], 'r') as fp:
            json_file = json.load(fp)
            
        sample = {'data': json_file, 'filename': self.files[idx].stem}
        if not self._only_data:
            image = io.imread(self.files[idx].with_suffix('.png'))[:,: , :3] # Removing the Alpha channel
            sample['image'] = image            
            
        if self.transform:
            sample = self.transform(sample)
        
        return sample
