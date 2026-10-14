"""Thread-safe rolling window of recent served feature rows (input to drift detection)."""

import threading
from collections import deque
from typing import Deque, Dict

import pandas as pd


class PredictionBuffer:
    def __init__(self, maxlen: int) -> None:
        self._rows: Deque[Dict[str, float]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, row: Dict[str, float]) -> None:
        with self._lock:
            self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def to_frame(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(list(self._rows))

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()
