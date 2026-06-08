"""Standalone depth-only hierarchical PPO + RND exploration baseline."""

from .runtime.registration import TASK_ID, register_task

__all__ = ["TASK_ID", "register_task"]
