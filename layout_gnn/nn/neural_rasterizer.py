import torch
import torch.nn as nn
import numpy as np


class CNNNeuralRasterizer(nn.Module):
    """CNN Neural Rasterizer 
    This module is based on the decoder of HDCGAN generator.
    
    References
    ----------
    [1] Curtó, J. D., Zarza, I. C., de la Torre, F., King, I., and Lyu, M. R., 
    High-resolution Deep Convolutional Generative Adversarial Networks,
    arXiv e-prints, 2017.
    """    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        output_size: int,
        **kwargs,
    ):  
        """
        Args:
            input_dim (int): Dimention of the input vector.
            output_dim (int): Output dimention.
            hidden_dim (int): Hidden dimention.
            output_size (int): Output image size.
        """    
        super().__init__()    
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.number_layers = int(np.log2(self.output_size)) - 2
        
        layers, in_channel = [], self.input_dim
        for i in reversed(range(self.number_layers)):
            out_channel = self.hidden_dim * 2**i
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=4, 
                    stride=1 if len(layers) == 0 else 2, 
                    padding=0 if len(layers) == 0 else 1, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channel),
                nn.SELU(True),
            ])
            in_channel = out_channel

        layers.extend([
            nn.ConvTranspose2d(
                in_channels=in_channel,
                out_channels=self.output_dim,
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False),
            nn.Tanh()
        ])
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs the neural renderer given an input.

        Args:
            inputs (torch.Tensor): Latent vector. It should have the shape of [batch, input_dim, 1, 1].

        Returns:
            torch.Tensor: Neural render output. It will have the shape of [batch, output_dim, output_size, output_size].
        """        
        return self.layers(inputs)