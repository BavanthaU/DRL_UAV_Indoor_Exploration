"""Depth-only hierarchical PPO/RND UAV exploration task for Isaac Lab."""

from .registration import TASK_ID, register_task

register_task()

__all__ = ["TASK_ID", "register_task"]
