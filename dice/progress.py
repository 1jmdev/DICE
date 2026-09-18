"""Progress reporting helpers shared across the pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from tqdm import tqdm

ItemType = TypeVar("ItemType")


def progress(
    iterable: Iterable[ItemType],
    total: int | None = None,
    description: str | None = None,
    leave: bool = True,
    disable: bool | None = None,
) -> tqdm:
    """Wrap an iterable in a progress bar.

    Args:
        iterable: The sequence or generator to consume.
        total: Expected number of items, required for generators.
        description: Label shown to the left of the bar.
        leave: Keep the finished bar on screen.
        disable: Force the bar on or off; ``None`` disables it when the output
            stream is not a terminal.
    """
    return tqdm(
        iterable,
        total=total,
        desc=description,
        leave=leave,
        disable=disable,
    )
