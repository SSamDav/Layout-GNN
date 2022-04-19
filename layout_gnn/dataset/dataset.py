import json
import zipfile
from pathlib import Path

from skimage import io
from torch.utils.data import Dataset


class RICOSemanticAnnotationsDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None, only_data: bool = False):
        
        self.files = list((root_dir / 'semantic_annotations').glob('*.json'))
        self.root_dir = root_dir
        self.transform = transform
        self._only_bool = only_data
        
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
