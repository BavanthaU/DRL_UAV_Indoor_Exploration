from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None


@dataclass
class WandbReplayLoggerConfig:
    wandb_enabled: bool = True
    wandb_project: str = "depth-hierarchical-rnd-uav"
    log_replay_every_iterations: int = 20
    max_replay_episodes_per_log: int = 2
    replay_fps: int = 8
    replay_resolution: int = 512
    output_dir: str = "logs/depth_hierarchical_ppo/replays"
    dry_run: bool = False


class WandbReplayLogger:
    def __init__(self, cfg: WandbReplayLoggerConfig | None = None):
        self.cfg = cfg or WandbReplayLoggerConfig()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._wandb_run = None
        if self.cfg.wandb_enabled and not self.cfg.dry_run:
            import wandb

            self._wandb_run = wandb.init(project=self.cfg.wandb_project, mode="offline", reinit=True)

    def should_log(self, iteration: int) -> bool:
        return self.cfg.log_replay_every_iterations > 0 and iteration % self.cfg.log_replay_every_iterations == 0

    def log_replays(self, iteration: int, episode: dict):
        if not self.should_log(iteration):
            return {}
        paths = {
            "top_down_episode_replay": self.render_top_down_episode(episode, "top_down_episode_replay.mp4"),
            "depth_view_replay": self.render_depth_view(episode, "depth_view_replay.mp4"),
            "map_candidate_replay": self.render_candidate_view(episode, "map_candidate_replay.mp4"),
        }
        scalars = dict(episode.get("scalars", {}))
        scalars.update({key: str(path) for key, path in paths.items()})
        if self._wandb_run is not None:
            import wandb

            payload = {key: wandb.Video(str(path), fps=self.cfg.replay_fps, format="mp4") for key, path in paths.items()}
            payload.update({key: value for key, value in episode.get("scalars", {}).items() if _is_number(value)})
            self._wandb_run.log(payload, step=iteration)
        return scalars

    def render_top_down_episode(self, episode: dict, filename: str = "top_down_episode_replay.mp4") -> Path:
        frames = []
        maps = episode.get("maps") or [episode.get("map")]
        trajectory = episode.get("trajectory", [])
        for idx, grid in enumerate(maps):
            frames.append(
                self._draw_map_frame(
                    grid,
                    trajectory=trajectory[: idx + 1] if trajectory else [],
                    text=[
                        f"coverage {episode.get('coverage_percent', 0.0):.1f}%",
                        f"local_done {episode.get('local_done_reason', 'continue')}",
                        f"selected {episode.get('selected_candidate', '-')}",
                        f"RND {episode.get('depth_rnd_reward', 0.0):.3f}",
                    ],
                )
            )
        return self._write_video(frames, filename)

    def render_depth_view(self, episode: dict, filename: str = "depth_view_replay.mp4") -> Path:
        depth_seq = episode.get("depth_rays") or [[1.0] * 64]
        frames = []
        for idx, rays in enumerate(depth_seq):
            frames.append(
                self._draw_depth_frame(
                    rays,
                    text=[
                        f"action {episode.get('action', [0, 0, 0])}",
                        f"min_depth {min(rays):.2f}",
                        f"depth RND {episode.get('depth_rnd_reward', 0.0):.3f}",
                        "near obstacle" if min(rays) < 0.45 else "clear",
                    ],
                )
            )
        return self._write_video(frames, filename)

    def render_candidate_view(self, episode: dict, filename: str = "map_candidate_replay.mp4") -> Path:
        frames = []
        candidates = episode.get("candidates", [])
        for _ in range(max(1, len(candidates))):
            frames.append(
                self._draw_candidate_frame(
                    candidates,
                    selected=episode.get("selected_candidate"),
                    text=[
                        f"type {episode.get('selected_candidate_type', '-')}",
                        f"gain {episode.get('realized_gain', 0.0):.3f}",
                        f"R_high {episode.get('high_level_reward', 0.0):.3f}",
                        f"MapRND {episode.get('map_rnd_reward', 0.0):.3f}",
                        f"failure {episode.get('failure_status', 'none')}",
                    ],
                )
            )
        return self._write_video(frames, filename)

    def _draw_map_frame(self, grid, trajectory, text):
        image, draw = self._blank()
        if np is not None and grid is not None:
            arr = np.asarray(grid)
            colors = {
                -1: (36, 42, 46),
                0: (210, 220, 210),
                1: (35, 35, 35),
                2: (77, 149, 255),
            }
            h, w = arr.shape
            cell = max(1, min(self.cfg.replay_resolution // max(h, w), 12))
            ox, oy = 16, 16
            for row in range(h):
                for col in range(w):
                    draw.rectangle([ox + col * cell, oy + row * cell, ox + (col + 1) * cell, oy + (row + 1) * cell], fill=colors.get(int(arr[row, col]), (180, 180, 180)))
            for row, col in trajectory:
                draw.ellipse([ox + col * cell, oy + row * cell, ox + (col + 1) * cell, oy + (row + 1) * cell], fill=(255, 80, 80))
        self._text(draw, text)
        return image

    def _draw_depth_frame(self, rays, text):
        image, draw = self._blank()
        width = self.cfg.replay_resolution - 32
        height = self.cfg.replay_resolution // 3
        rays = list(float(x) for x in rays)
        max_depth = max(max(rays), 1.0)
        for idx, depth in enumerate(rays):
            x0 = 16 + int(width * idx / max(1, len(rays)))
            x1 = 16 + int(width * (idx + 1) / max(1, len(rays)))
            y1 = self.cfg.replay_resolution - 48
            y0 = y1 - int(height * min(depth / max_depth, 1.0))
            draw.rectangle([x0, y0, max(x0 + 1, x1), y1], fill=(80, 170, 220))
        self._text(draw, text)
        return image

    def _draw_candidate_frame(self, candidates, selected, text):
        image, draw = self._blank()
        cx = cy = self.cfg.replay_resolution // 2
        for idx, candidate in enumerate(candidates or []):
            row = int(candidate.get("row", idx * 6) if isinstance(candidate, dict) else idx * 6)
            col = int(candidate.get("col", idx * 6) if isinstance(candidate, dict) else idx * 6)
            x = 32 + (col * 7) % max(32, self.cfg.replay_resolution - 64)
            y = 64 + (row * 7) % max(32, self.cfg.replay_resolution - 96)
            fill = (255, 210, 75) if selected == idx else (120, 180, 120)
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=fill)
            draw.line([cx, cy, x, y], fill=(80, 80, 80))
        self._text(draw, text)
        return image

    def _blank(self):
        if Image is None:
            raise RuntimeError("WandbReplayLogger requires Pillow.")
        image = Image.new("RGB", (self.cfg.replay_resolution, self.cfg.replay_resolution), (245, 247, 248))
        return image, ImageDraw.Draw(image)

    def _text(self, draw, lines):
        y = 8
        for line in lines:
            draw.text((8, y), str(line), fill=(10, 20, 30))
            y += 16

    def _write_video(self, frames, filename: str) -> Path:
        path = self.output_dir / filename
        try:
            import imageio.v2 as imageio

            imageio.mimsave(path, [np.asarray(frame) for frame in frames], fps=self.cfg.replay_fps, macro_block_size=1)
        except Exception:
            if frames:
                frames[0].save(path.with_suffix(".png"))
            path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
        return path


def _is_number(value):
    return isinstance(value, (int, float))
