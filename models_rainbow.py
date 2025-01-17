import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
           
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())
    

class Network(nn.Module):
    def __init__(
        self, 
        out_dim: int, 
        atom_size: int,
        non_visual_state_dim: int,
        support: torch.Tensor,
        std_noisy: float
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.obs_rest = (non_visual_state_dim>0)
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer_1 = nn.Sequential(
#             nn.Linear(in_dim, 128), 
#             nn.ReLU(),
              nn.Conv2d(3,32, kernel_size=8, stride=4),
              nn.ReLU(),
              nn.Conv2d(32,64, kernel_size=4, stride=2),
              nn.ReLU(),
              nn.Conv2d(64,64, kernel_size=3, stride=1),
              nn.ReLU()
        )
        
        self.feature_layer_2 = nn.Sequential( 
              nn.Linear(1024+non_visual_state_dim, 512),
              nn.ReLU(),
              nn.Linear(512, 128),
              nn.ReLU()
        )

        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128, std_noisy)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size, std_noisy)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128, std_noisy)
        self.value_layer = NoisyLinear(128, atom_size, std_noisy)

    def forward(self, x_visual: torch.Tensor, x_not_visual: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x_visual, x_not_visual)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self,  x_visual: torch.Tensor, x_not_visual: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        r = self.feature_layer_1(x_visual.permute(0,3,1,2))
        if self.obs_rest:
            feature = self.feature_layer_2(torch.cat([ r.view(r.shape[0], -1),  x_not_visual.view(-1,1) ], dim=1 ))
        else:
            feature = self.feature_layer_2(r.view(r.shape[0], -1))
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()