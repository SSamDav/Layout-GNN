import json
import zipfile
from pathlib import Path

from skimage import io
from torch.utils.data import Dataset


class RICOSemanticAnnotationsDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        
        self.files = list((root_dir / 'semantic_annotations').glob('*.json'))
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        with open(self.files[idx], 'r') as fp:
            json_file = json.load(fp)
            
        image = io.imread(self.files[idx].with_suffix('.png'))[:,: , :3] # Removing the Alpha channel
        sample = {'data': json_file, 'image': image}
        if self.transform:
            sample = self.transform(sample)
        
        return sample
