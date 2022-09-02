from typing import Optional, Type

import torch.nn as nn


def get_mlp(*features, num_layers: Optional[int] = None, activation_cls: Type[nn.Module] = nn.ReLU) -> nn.Sequential:
    if num_layers is not None:
        assert features, "At least in features must be provided."
        dif = num_layers + 1 - len(features)
        if dif > 0:
            # Repeat the last given number of features until the last layer.
            features += (features[-1], ) * dif
        elif dif < 0:
            # Use the first given numbers of features up to num layers. 
            features = features[:dif]

    assert len(features) >= 2, "At least in and out features must be provided"
    layers = [nn.Linear(*features[:2])]
    for i in range(1, len(features) - 1):
        layers.append(activation_cls())
        layers.append(nn.Linear(*features[i:i+2]))
    return nn.Sequential(*layers)