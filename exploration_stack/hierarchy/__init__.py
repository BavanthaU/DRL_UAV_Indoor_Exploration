"""Learned hierarchy for VLM-PPO exploration."""

from .candidate_builder import CandidateBuilder, CandidateBuilderConfig
from .candidate_encoder import CandidateEncoderConfig, CandidateSetEncoder
from .credit_assignment import OptionCreditAccumulator, OptionCreditConfig
from .hierarchical_actor_critic import HierarchicalActorCritic, HierarchicalActorCriticConfig
from .option_manager import OptionManager, OptionManagerConfig
from .option_masking import build_option_mask
from .option_policy import OptionPolicyConfig, SemanticOptionPolicy
from .safety_shield import DepthLineSafetyShield, DepthLineSafetyShieldConfig
from .option_types import CandidateType, ExplorationOption, NUM_CANDIDATE_TYPES, NUM_OPTIONS, OPTION_NAMES

__all__ = [
    "CandidateBuilder",
    "CandidateBuilderConfig",
    "CandidateEncoderConfig",
    "CandidateSetEncoder",
    "CandidateType",
    "ExplorationOption",
    "HierarchicalActorCritic",
    "HierarchicalActorCriticConfig",
    "NUM_CANDIDATE_TYPES",
    "NUM_OPTIONS",
    "OPTION_NAMES",
    "OptionCreditAccumulator",
    "OptionCreditConfig",
    "OptionManager",
    "OptionManagerConfig",
    "OptionPolicyConfig",
    "SemanticOptionPolicy",
    "DepthLineSafetyShield",
    "DepthLineSafetyShieldConfig",
    "build_option_mask",
]
