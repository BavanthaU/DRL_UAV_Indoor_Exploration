from typing import Dict
from gymnasium import spaces
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict


class ConvNeXtTinyImageFeatureExtractor(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        fix_cnn: bool = False,
    ):
        """
        Feature extractor using ConvNeXt-Tiny pretrained on ImageNet.
        Optionally adapts input channels and freezes weights.

        Args:
            observation_space: The image observation space (CxHxW).
            features_dim: Dimension of the output features.
            fix_cnn: If True, CNN backbone weights are frozen.
        """

        # Get input channels from observation space
        input_channels = observation_space.shape[0]
        super().__init__(observation_space, features_dim)
 
        # Load ConvNeXt-Tiny and modify input layer
        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        convnext.features[0][0] = nn.Conv2d(
            input_channels, 96, kernel_size=4, stride=4)
 
        # Optionally freeze CNN backbone
        if fix_cnn:
            for param in convnext.features.parameters():
                param.requires_grad = False
 
        # Save backbone and projection layers
        self.backbone = convnext.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(768, features_dim)  # 768 is ConvNeXt-Tiny last feature dim
 
    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.backbone(observations)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x   

class ExplorationBasicCombinedFeaturesExtractorGeneralChannel(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observation spaces in exploration task.

    - Image-like inputs: processed with ConvNeXt-Tiny backbone.
    - Vector inputs: flattened directly.
    - Outputs from each extractor are concatenated into a single feature vector.
    
    Args:
        observation_space (spaces.Dict): The observation space.
        cnn_output_dim (int): Number of features output by each CNN submodule.
        normalized_image (bool): Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions. Otherwise, it checks that it has expected 
        dtype (uint8) and bounds (values in [0, 255]). Defaults to True.
        fix_cnn (bool): If True, CNN backbone weights are frozen.
        """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 128,                     
        normalized_image: bool = True,
        fix_cnn: bool = False,
    ) -> None:
        
        # Temporarily set features_dim=1, updated after processing all subspaces
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        # Build feature extractors for each Dict key
        num_concatenated_ftrs = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = ConvNeXtTinyImageFeatureExtractor(subspace, features_dim=cnn_output_dim, fix_cnn=fix_cnn)
                num_concatenated_ftrs += cnn_output_dim
            else:
                # Treat 1D observation as vector input
                extractors[key] = nn.Flatten()
                num_concatenated_ftrs += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update features dim
        self._features_dim = num_concatenated_ftrs

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            
        return th.cat(encoded_tensor_list, dim=1)