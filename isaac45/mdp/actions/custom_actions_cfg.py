from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import custom_actions

@configclass
class KeyBoardDrivenActionCfg(ActionTermCfg):
    """Configuration for action term to drive the drone with the keyboard.

    See :class:`KeyBoardDriveAction` for more details.
    """

    class_type: type[ActionTerm] = custom_actions.KeyBoardDrivenAction

@configclass
class RLDrivenActionCfg(ActionTermCfg):
    """Configuration for the action term that sends 3D velocity setpoints from the RL algorithm to the low-level controller to drive the drone in 2D.

    See :class:`RLDrivenAction` for more details.
    """

    class_type: type[ActionTerm] = custom_actions.RLDrivenAction

@configclass
class RLDriven2DActionCfg(ActionTermCfg):
    """Configuration for the action term that sends 2D velocity setpoints from the RL algorithm to the low-level controller to drive the drone in 2D.

    See :class:`RLDriven2DAction` for more details.
    """

    class_type: type[ActionTerm] = custom_actions.RLDriven2DAction


@configclass
class RLDrivenDiscreteLimited5ValuesActionCfg(ActionTermCfg):
    """Configuration for the action term that sends discrete velocity commands from the RL algorithm to the low-level controller to drive the drone in 2D.

    See :class:`RLDrivenDiscreteLimited5ValuesAction` for more details.
    """

    class_type: type[ActionTerm] = custom_actions.RLDrivenDiscreteLimited5ValuesAction


@configclass
class IdealRLDrivenActionCfg(ActionTermCfg):
    """Configuration for the action term that transform discrete velocity commands from the RL algorithm to drone positions and orientations, and directly sets them into the simulator to move the drone in 2D.

    See :class:`IdealRLDrivenAction` for more details.
    """

    class_type: type[ActionTerm] = custom_actions.IdealRLDrivenAction



@configclass
class IdealRLDrivenAction2DCfg(ActionTermCfg):
    """Configuration for the action term that transform discrete velocity commands from the RL algorithm to drone positions and orientations, and directly sets them into the simulator to move the drone in 2D.

    See :class:`IdealRLDrivenAction` for more details.
    """

    class_type: type[ActionTerm] = custom_actions.IdealRLDrivenAction2D
