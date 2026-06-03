from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_EXPLORATION_PROMPTS = [
    "a doorway leading to another area",
    "a corridor or hallway continuation",
    "an open navigable indoor space",
    "a wall or obstacle blocking the drone",
    "a narrow passage requiring careful flying",
    "a frontier between explored and unexplored space",
    "a place likely to reveal new area",
    "a risky area close to collision",
    "a repeated or already explored area",
    "an exit from the current area",
    "a dead end",
    "a safe direction for a small drone",
    "an area that should be explored next",
]


@dataclass(frozen=True)
class PromptBank:
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_EXPLORATION_PROMPTS))

    @property
    def doorway_index(self) -> int:
        return self.prompts.index("a doorway leading to another area")

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

    @property
    def repeated_index(self) -> int:
        return self.prompts.index("a repeated or already explored area")

    @property
    def dead_end_index(self) -> int:
        return self.prompts.index("a dead end")

    @property
    def explore_next_index(self) -> int:
        return self.prompts.index("an area that should be explored next")

    def __len__(self) -> int:
        return len(self.prompts)
