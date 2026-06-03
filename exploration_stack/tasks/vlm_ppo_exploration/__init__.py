"""Standalone VLM-PPO UAV exploration task."""

from .registration import TASK_ID, register_task

register_task()

__all__ = ["TASK_ID", "register_task"]

