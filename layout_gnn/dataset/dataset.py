import json
import zipfile
import requests
import shutil
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from google.cloud import storage
from skimage import io
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from joblib import Parallel, delayed, parallel_backend


# Local folders
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / 'data'
# Rico dataset
RICO_BUCKET_ID = 'crowdstf-rico-uiuc-4540'
RICO_BUCKET_PATH = 'rico_dataset_v0.1/semantic_annotations.zip'
# Enrico dataset
ENRICO_RESOURCES_BASE_URL = 'http://userinterfaces.aalto.fi/enrico/resources/'
ENRICO_HIERARCHIES_FILE = 'hierarchies.zip'
ENRICO_SEMANTIC_IMAGES_FILE = 'wireframes.zip'


class RICOSemanticAnnotationsDataset(Dataset):
    def __init__(
        self,
        root_dir: Optional[Path] = None,
        transform=None,
        only_data: bool = False,
        cache_items: bool = False,
    ):
        self.data_path = root_dir or DATA_PATH
        self.transform = transform
        self._only_data = only_data
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.label_color_map = json.load(open(self.data_path / 'component_legend.json', 'r'))
        self.icon_label_color_map = json.load(open(self.data_path / 'icon_legend.json', 'r'))
        self.text_button_color_map = json.load(open(self.data_path / 'textButton_legend.json', 'r'))
        
        self.label_color_map['Icon'] = next(iter(self.icon_label_color_map.values()))
        self.label_color_map['Text Button'] = next(iter(self.text_button_color_map.values()))

        self.files = self.get_files()

        # List that will store the cached items in memory
        self.data: Optional[List[Optional[Dict[str, Any]]]] = [None] * len(self.files) if cache_items else None
    
    def get_files(self) -> List[Path]:
        zip_filename = self.data_path / 'semantic_annotations.zip'
        if not zip_filename.exists():
            client = storage.Client.create_anonymous_client()
            bucket = client.bucket(RICO_BUCKET_ID)
  
            blob = bucket.blob('rico_dataset_v0.1/semantic_annotations.zip')
            blob.download_to_filename(zip_filename)
        
        extracted_folder = self.data_path / 'semantic_annotations'
        if not extracted_folder.exists():
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)

        file_with_errors = json.load(open(self.data_path / 'file_with_error.json', 'r'))
        return sorted(f for f in (extracted_folder).glob('*.json') if f.stem not in file_with_errors)

    def prepare(self, n_jobs: int = -1, max_size: Optional[int] = None):
        max_size = max_size or len(self.files)
        with parallel_backend('threading', n_jobs=n_jobs):
            self.data = Parallel()(delayed(self.load_sample)(file) for file in tqdm(self.files[:max_size]))
            
    def load_sample(self, file: Path) -> Dict[str, Any]:
        with open(file, 'r') as fp:
            json_file = json.load(fp)
            
        sample = {'data': json_file, 'filename': file.stem}
        if not self._only_data:
            image = io.imread(file.with_suffix('.png'))[:,: , :3] # Removing the Alpha channel
            sample['image'] = image 
            
        if self.transform:
            sample = self.transform(sample)
        
        return sample
            
    def _get_item(self, idx: int) -> Dict[str, Any]:
        if self.data is not None:
            if self.data[idx] is None:
                self.data[idx] = self.load_sample(self.files[idx])
            return self.data[idx]
        else:
            return self.load_sample(self.files[idx])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._get_item(idx)


class RICOTripletsDataset(RICOSemanticAnnotationsDataset):
    VALID_TRIPLET_METRICS = {"iou", "ged"}

    def __init__(
        self, 
        root_dir: Path, 
        triplets: Union[Sequence[Dict[str, str]], str, Path], 
        triplet_metric: str = "ged", **kwargs
    ):
        if triplet_metric not in self.VALID_TRIPLET_METRICS:
            raise ValueError(
                f"Invalid value {triplet_metric} for argument `triplet_metric`. "
                f"Must be one of {self.VALID_TRIPLET_METRICS}"
            )

        super().__init__(root_dir=root_dir, **kwargs)

        if isinstance(triplets, (str, Path)):
            path = root_dir / triplets
            with open(path) as f:
                triplets = json.load(f)
                
        self.triplets = list(filter(
            lambda x: "anchor" in x and f"pos_{triplet_metric}" in x and f"neg_{triplet_metric}" in x, 
            triplets,
        ))
        self.triplet_metric = triplet_metric
        self.file_name_to_idx = {k.stem: i for i, k in enumerate(self.files)}

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx = self.file_name_to_idx[self.triplets[idx]["anchor"]]
        pos_idx = self.file_name_to_idx[self.triplets[idx][f"pos_{self.triplet_metric}"]["pair"]]
        neg_idx = self.file_name_to_idx[self.triplets[idx][f"neg_{self.triplet_metric}"]["pair"]]
        
        return {
            "anchor": self._get_item(anchor_idx),
            "pos": self._get_item(pos_idx, only_data=True),
            "neg": self._get_item(neg_idx, only_data=True),
        }
        

class ENRICOSemanticAnnotationsDataset(RICOSemanticAnnotationsDataset):
    def __init__(
        self,
        root_dir: Optional[Path] = None,
        transform=None,
        only_data: bool = False,
        cache_items: bool = False,
    ) -> None:
        super().__init__(root_dir=root_dir, transform=transform, only_data=only_data, cache_items=cache_items)

        labels = pd.read_csv(self.data_path / 'design_topics.csv', dtype={'screen_id': str, 'topic': str})
        self.labels: Dict[str, str] = labels.set_index('screen_id').to_dict()['topic']

    def get_files(self) -> List[Path]:
        enrico_folder = self.data_path / 'enrico_semantic_annotations'
        enrico_folder.mkdir(exist_ok=True)
        
        # Download JSON Files
        zip_filename = enrico_folder / ENRICO_HIERARCHIES_FILE
        if not zip_filename.exists():
            req = requests.get(ENRICO_RESOURCES_BASE_URL + ENRICO_HIERARCHIES_FILE)
            with open(zip_filename, 'wb') as output_file:
                output_file.write(req.content)
        
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(enrico_folder)
                
            for file in (enrico_folder / 'hierarchies').glob('*.json'):
                file.rename(enrico_folder / file.name)
                
            shutil.rmtree(enrico_folder / 'hierarchies')

        # Download PNG Files
        zip_filename = enrico_folder / ENRICO_SEMANTIC_IMAGES_FILE
        if not zip_filename.exists():
            req = requests.get(ENRICO_RESOURCES_BASE_URL + ENRICO_SEMANTIC_IMAGES_FILE)
            with open(zip_filename, 'wb') as output_file:
                output_file.write(req.content)
        
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(enrico_folder)

            for file in (enrico_folder / 'wireframes').glob('*.png'):
                file.rename(enrico_folder / file.name)
                
            shutil.rmtree(enrico_folder / 'wireframes')

        issues = pd.read_csv(self.data_path / 'enrico_issues.csv', dtype=str)
        issue_files = issues[issues['source'].isin(['wireframe', 'hierarchy'])]['screen_id'].to_list()
        return sorted(f for f in (enrico_folder).glob('*.json') if f.stem not in issue_files)

    def load_sample(self, file: Path) -> Dict[str, Any]:
        sample = super().load_sample(file)
        sample["label"] = self.labels[sample["filename"]]
        return sample
