import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from torchvision import models

from typing import Dict

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict

from collections import OrderedDict


class ConvNeXtTinyFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor based on ConvNeXt-Tiny for use with reinforcement learning/imitation learning
    policies. This class adapts the ConvNeXt-Tiny model to accept any number of input channels and extracts features.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        fix_cnn: bool = False,
    ):
        # Get input channels from observation space
        input_channels = observation_space.shape[0]
        
        super().__init__(observation_space, features_dim)
 
        # Load ConvNeXt-Tiny and modify input layer to match the input channel count
        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        convnext.features[0][0] = nn.Conv2d(
            input_channels, 96, kernel_size=4, stride=4
        )
 
        # Optionally freeze feature extractor
        if fix_cnn:
            for param in convnext.features.parameters():
                param.requires_grad = False
 
        # Store ConvNeXt backbone
        self.backbone = convnext.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(768, features_dim)  # 768 is ConvNeXt-Tiny last feature dim
 
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Pass the observations through the ConvNeXt backbone and return the extracted features
        x = self.backbone(observations)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x 
    

class ExplorationBasicCombinedFeaturesExtractorGeneralChannel(BaseFeaturesExtractor):
    """
    Custom features extractor for Dict observation spaces used in the exploration task.
    The input corresponding to each dictionary key is fed through a separate features extractor 
    (whose architecture depends on the input shape).
    - vectors: flattened + fed to an MLP
    - matrices: fed to PretrainedCustomCNNGeneralChannel. Images of shape CxHxW (C can be any value).
    
    The output features are concatenated and fed to an MLP.
    Args:
        observation_space: Gym observation space (Box).
        cnn_output_dim (int): Dimension of cnn ouptut.
        normalized_image (bool): Indicates if image is normalized or not.
        fix_cnn (bool): Indicate if the CNN is frozen. 
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 128,                     
        normalized_image: bool = True,
        fix_cnn: bool = False,
        device='cuda:0',
    ) -> None:       
        # We do not know features-dim here before going over all the items, so put something there. Will be changed later on. 
        super().__init__(observation_space, features_dim=1)


        extractors: OrderedDict[str, nn.Module] = OrderedDict()
        # Proper features extractor
        num_concatenated_ftrs = 0

        # To preserve the order of observations, list the keys used. 
        # ordered_keys = ['depth_obs', 'map_obs', 'linear_velocity', 'angular_velocity', 'oneline_depth']
        # ordered_keys = ['map_obs', 'linear_velocity', 'angular_velocity', 'oneline_depth']
        # ordered_keys = ['map_obs', 'semantic_obs','linear_velocity', 'angular_velocity','oneline_depth']
        # ordered_keys = ['map_obs','semantic_obs','oneline_depth']
        ordered_keys = ['map_obs','oneline_depth', 'oneline_sems']
        # ordered_keys = ['map_obs', 'oneline_depth']

        # Specify which observation keys to process and in which order
        for key in ordered_keys:
            subspace = observation_space.spaces[key]
            
            # If observation is an image extract features first. Then concatenate all observations
            if is_image_space(subspace, normalized_image=normalized_image):                
                extractors[key] = ConvNeXtTinyFeatureExtractor(subspace, features_dim=cnn_output_dim, fix_cnn=fix_cnn)

                num_concatenated_ftrs += cnn_output_dim
            else:
                # For non-image observations, flatten the input
                extractors[key] = nn.Flatten()
                num_concatenated_ftrs += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = num_concatenated_ftrs

    def forward(self, observations: TensorDict) -> torch.Tensor:
        # Forward pass through extractor
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        
        return torch.cat(encoded_tensor_list, dim=1)
    

class SACNet(nn.Module):
    """
    A custom neural network for Soft Actor-Critic (SAC), incorporating a convolutional 
    feature extractor for image observations.

    Args:
        observation_space: Gym observation space (Box).
        act_dim (int): Dimension of the action space.
        device (torch.device): Computation device (CPU or CUDA).
        fix_cnn (bool): If True, freezes the convolutional layers in the feature extractor.
    """
    def __init__(self, observation_space, act_dim, device, fix_cnn=False):
        super().__init__()
        self.device = device
        hidden_dim=256

        # Feature extractor handles images and concatenates observations
        self.feature_extractor = ExplorationBasicCombinedFeaturesExtractorGeneralChannel(observation_space, device=self.device, fix_cnn=fix_cnn)
        features_dim = self.feature_extractor._features_dim

        # Actor network
        self.actor_fc = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 128),
            nn.ELU(),
            nn.Dropout(p=0.2),
        )
        self.actor_head = nn.Linear(128, act_dim) 

        # Learnable log standard deviation
        self.log_std = nn.Linear(128, act_dim)
        nn.init.xavier_uniform_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, 0)
        self.log_std.bias.data.zero_()

        # Critic network
        self.critic_fc = nn.Sequential(
            nn.Linear(features_dim+act_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 128),
            nn.ELU(),
            nn.Dropout(p=0.2),
        )
        self.critic_head = nn.Linear(128, 1)

        self.reset_parameters()

    
    def forward(self, observations, action=None):
        # Extract features from the observation
        net_in = observations
        features = self.feature_extractor(net_in)

        # Actor forward pass
        raw_action = self.actor_head(self.actor_fc(features))  # shape: [batch, 2]

        # Apply different bounds to output actions
        action_0 = (torch.tanh(raw_action[:, 0]) + 1) / 2         # maps to [0, 1]
        action_1 = torch.tanh(raw_action[:, 1])                # maps to [-1, 1]
        pi_mean = torch.stack([action_0, action_1], dim=1)

        # Compute log standard deviation and clip for stability
        raw_std = self.log_std(self.actor_fc(features))
        log_std = raw_std.clamp(min=-20, max=2)  # Clamping prevents extreme values
        pi_std = log_std.exp()
        if action is None:
            action = pi_mean  # use actor output for IL or inference

        # Critic forward pass
        critic_input = torch.cat([features, action], dim=-1)
        q_value = self.critic_head(self.critic_fc(critic_input))  # Single critic

        return q_value, pi_mean, pi_std


    def reset_parameters(self):
        # Initialize actor output layer weights with low gain
        nn.init.xavier_uniform_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias.data, 0)
