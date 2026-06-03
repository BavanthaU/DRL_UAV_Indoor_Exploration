from __future__ import annotations


def skrl_available() -> bool:
    try:
        import skrl  # noqa: F401
    except ImportError:
        return False
    return True


class SkrlPPOTrainerAdapter:
    """SKRL integration marker for future drop-in backend selection.

    The current VLM actor-critic is a custom PyTorch module with dict image/map
    observations and auxiliary heads. The repository uses TorchPPOTrainerAdapter
    for this first trainable target; this class keeps backend selection explicit
    and fails with an actionable message instead of pretending SKRL was used.
    """

    def __init__(self, *args, **kwargs):
        if not skrl_available():
            raise RuntimeError("SKRL is not installed in this environment.")
        raise RuntimeError(
            "SKRL is installed, but this custom VLM actor-critic is not yet "
            "wired through SKRL's model API. Use backend=torch_reference for "
            "the current trainable VLM-PPO path."
        )

