"""Plot observed game prices and discounts over time with Matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_history(
    history: Iterable[dict[str, Any]],
    title: str,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Draw price and discount series; optionally save and/or show the figure."""
    rows = list(history)
    if not rows:
        raise ValueError("Cannot plot an empty price history")
    frame = pd.DataFrame.from_records(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date")

    figure, price_axis = plt.subplots(figsize=(10, 5))
    price_axis.plot(
        frame["date"],
        frame["price"],
        marker="o",
        color="#2864a5",
        label="Observed price",
    )
    price_axis.set_title(title)
    price_axis.set_xlabel("Observation date")
    price_axis.set_ylabel(f"Price ({frame['currency'].iloc[-1]})")
    price_axis.grid(True, alpha=0.25)

    discount_axis = price_axis.twinx()
    discount_axis.plot(
        frame["date"],
        frame["discount_percentage"],
        color="#d97706",
        linestyle="--",
        alpha=0.8,
        label="Discount",
    )
    discount_axis.set_ylabel("Discount (%)")
    discount_axis.set_ylim(0, 100)

    figure.autofmt_xdate()
    figure.tight_layout()
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    if show:
        plt.show()
    plt.close(figure)
