from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class PyBoySession:
    rom_path: Path
    window: str
    scale: int
    speed: int
    debug: bool = False

    pyboy: Any = None
    game_wrapper: Any = None

    def open(self) -> None:
        from pyboy import PyBoy

        self.pyboy = PyBoy(
            str(self.rom_path),
            window=self.window,
            scale=self.scale,
            debug=self.debug,
        )
        self.pyboy.set_emulation_speed(self.speed)
        self.game_wrapper = self.pyboy.game_wrapper

    def start_game(self, timer_div: int | None) -> None:
        if self.pyboy is None:
            self.open()
        self.game_wrapper.start_game(timer_div=timer_div)

    def reset_game(self, timer_div: int | None) -> None:
        self.game_wrapper.reset_game(timer_div=timer_div)

    def press(self, button_name: str) -> None:
        self.pyboy.button(button_name)

    def tick(self, ticks: int, render: bool) -> None:
        self.pyboy.tick(ticks, render)

    def game_area(self) -> np.ndarray:
        return np.asarray(self.game_wrapper.game_area(), dtype=np.int16)

    @property
    def score(self) -> int:
        return int(self.game_wrapper.score)

    @property
    def lines(self) -> int:
        return int(self.game_wrapper.lines)

    @property
    def level(self) -> int:
        return int(self.game_wrapper.level)

    def next_tetromino(self) -> str:
        return str(self.game_wrapper.next_tetromino())

    def game_over(self) -> bool:
        return bool(self.game_wrapper.game_over())

    def close(self) -> None:
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
            self.pyboy = None
            self.game_wrapper = None
