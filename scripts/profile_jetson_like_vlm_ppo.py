#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import profile_vlm_ppo_explorer


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/vlm_ppo_explorer/jetson_mobileclip_deploy.yaml"])
    profile_vlm_ppo_explorer.main()
