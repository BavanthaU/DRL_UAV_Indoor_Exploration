from __future__ import annotations


def rsl_rl_available() -> bool:
    try:
        import rsl_rl  # noqa: F401
    except ImportError:
        return False
    return True


class RslRlPPOTrainerAdapter:
    def __init__(self, *args, **kwargs):
        if not rsl_rl_available():
            raise RuntimeError("RSL-RL is not installed in this environment.")
        raise RuntimeError(
            "RSL-RL is available, but custom VLM dict observations and auxiliary "
            "heads are not yet integrated into an RSL-RL runner config. Use "
            "backend=torch_reference for the current trainable path."
        )

