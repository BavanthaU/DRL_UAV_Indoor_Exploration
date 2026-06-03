from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_EXPLORATION_PROMPTS = [
    "a doorway leading to another room",
    "a corridor or hallway continuation",
    "an open navigable indoor space",
    "a wall or obstacle blocking the drone",
    "a narrow passage requiring careful flying",
    "a frontier between explored and unexplored space",
    "a place likely to reveal new area",
    "a risky area close to collision",
    "a repeated or already explored area",
    "a room exit",
    "a dead end",
    "a safe direction for a small drone",
]


@dataclass(frozen=True)
class PromptBank:
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_EXPLORATION_PROMPTS))

    @property
    def doorway_index(self) -> int:
        return self.prompts.index("a doorway leading to another room")

    @property
    def corridor_index(self) -> int:
        return self.prompts.index("a corridor or hallway continuation")

    @property
    def open_space_index(self) -> int:
        return self.prompts.index("an open navigable indoor space")

    @property
    def obstacle_index(self) -> int:
        return self.prompts.index("a wall or obstacle blocking the drone")

    @property
    def frontier_index(self) -> int:
        return self.prompts.index("a frontier between explored and unexplored space")

    @property
    def risky_index(self) -> int:
        return self.prompts.index("a risky area close to collision")

    def __len__(self) -> int:
        return len(self.prompts)

