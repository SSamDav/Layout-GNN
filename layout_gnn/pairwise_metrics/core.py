from functools import cached_property
import multiprocessing as mp
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from layout_gnn.dataset.dataset import RICOSemanticAnnotationsDataset


class PairwiseMetricLoader:
    def __init__(self, dataset: RICOSemanticAnnotationsDataset) -> None:
        self.dataset = dataset

    @property
    def header(self) -> str:
        return "screen_id1,screen_id2,value\n"

    @cached_property
    def key_to_index(self) -> Dict[str, int]:
        return {file.stem: index for index, file in enumerate(self.dataset.files)}

    def get_matrix(
        self,
        values: Iterable[Tuple[str, str, float]],
    ) -> np.ndarray:
        matrix = np.zeros((len(self.dataset), len(self.dataset)))
        for k1, k2, v in values:
            i, j = self.key_to_index[k1], self.key_to_index[k2]
            matrix[i, j] = matrix[j, i] = v
        return matrix

    def iter_values_from_csv(self, filepath: Union[str, Path], verbose: int = 0) -> Iterator[Tuple[str, str, float]]:
        # NOTE: This method is ~15x faster than using pd.read_csv and then iterrows
        def parse_line(line: str) -> Tuple[str, str, float]:
            k1, k2, v = line.rstrip().split(",")
            return k1, k2, float(v)

        with open(filepath) as f:
            header = f.readline()
            if self.header != header:
                raise ValueError(f"Invalid header. Got '{header}', expected '{self.header}'.")

            yield from tqdm(map(parse_line, f), desc=f"Loading {filepath}", disable=(verbose == 0))

    def get_matrix_from_csv(self, filepath: Union[str, Path], verbose: int = 0):
        return self.get_matrix(self.iter_values_from_csv(filepath=filepath, verbose=verbose))


class PairwiseMetricCalculator(PairwiseMetricLoader):
    def __init__(
        self,
        dataset: RICOSemanticAnnotationsDataset,
        distance_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
    ) -> None:
        super().__init__(dataset=dataset)
        self.distance_fn = distance_fn

    def __call__(self, pair: Tuple[int, int]) -> Tuple[str, str, float]:
        i, j = pair
        sample1, sample2 = self.dataset[i], self.dataset[j]
        distance = self.distance_fn(sample1, sample2)
        return sample1["filename"], sample2["filename"], distance

    def iter_values(
        self,
        num_processes: Optional[int] = None,
        verbose: int = 0,
    ) -> Iterator[Tuple[str, str, float]]:
        if num_processes is None:
            num_processes = mp.cpu_count()

        n = len(self.dataset)
        pairs = combinations(range(n), 2)
        total = int(n*(n-1)/2)  # Number of combinations of n elements taken 2 at a time without repetition

        # Common code between single and multi process scenarios. The only thing that changes is the function used to
        # apply the transformation to the pairs.
        def get_iterator(map: Callable[[Callable, Iterable], Iterator]) -> Iterator:
            return tqdm(map(self, pairs), total=total, disable=(verbose == 0))

        if num_processes > 1:
            with mp.get_context("spawn").Pool(processes=num_processes) as pool:
                # Use the imap_unordered method
                yield from get_iterator(map=pool.imap_unordered)
        else:
            # Use the built-in map function
            yield from get_iterator(map=map)

    def compute_matrix(
        self,
        num_processes: Optional[int] = None,
        verbose: int = 0,
    ) -> np.ndarray:
        return self.get_matrix(self.iter_values(num_processes=num_processes, verbose=verbose))

    def write_values_to_csv(
        self,
        filepath: Union[str, Path],
        num_processes: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        with open(filepath, "w") as f:
            f.write(self.header)
            for k1, k2, v in self.iter_values(num_processes=num_processes, verbose=verbose):
                f.write(f"{k1},{k2},{v}\n")
