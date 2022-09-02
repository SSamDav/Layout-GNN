"""Reference: SimCLR, Chen et al., ICML 2020 (https://arxiv.org/abs/2002.05709)"""

from typing import Callable, Optional

import torch
import torch.nn as nn


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nn.functional.normalize(a) @ nn.functional.normalize(b).T


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss, as used in SimCLR.
    """
    def __init__(
        self,
        temperature: float = 0.5,
        similarity_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        **kwargs
    ) -> None:
        """
        Args:
            temperature (float, optional): Temperature parameter (1 is equivlent to regular cross entropy). Defaults to
                0.5 (see SimCLR Appendix B.9).
            similarity_fn (Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): Similarity
                function. Callable that receives two tensors with shape (N, D) and returns the pairwise similarity
                matrix of shape (N, N). If not provided, cosine similarity is used.
        """
        super().__init__()
        self.temperature = temperature
        self.similarity_fn = similarity_fn if similarity_fn is not None else cosine_similarity_matrix
        self.cross_entropy = nn.CrossEntropyLoss(**kwargs)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        similarity = self.similarity_fn(a, b)
        return self.cross_entropy(
            similarity / self.temperature,
            torch.arange(similarity.shape[0], device=similarity.device),
        )
