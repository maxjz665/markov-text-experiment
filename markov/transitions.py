"""Матрицы переходов первого порядка: количества N и вероятности P."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransitionModel:
    states: list[str]
    N: np.ndarray          # int64, N[i, j] — число переходов i→j
    unigrams: np.ndarray   # int64, число вхождений каждого состояния
    n_segments: int        # сегментов длиной ≥ 1

    @property
    def total_pairs(self) -> int:
        return int(self.N.sum())

    @property
    def row_totals(self) -> np.ndarray:
        return self.N.sum(axis=1)

    @property
    def P(self) -> np.ndarray:
        return probabilities(self.N)


def count_transitions(segments: list[str], states: list[str] | str) -> TransitionModel:
    """Считает переходы только между соседними символами внутри каждого сегмента.

    Символ вне ``states`` — ошибка: сегменты должны быть заранее очищены.
    """
    states = list(states)
    index = {s: k for k, s in enumerate(states)}
    n = len(states)
    N = np.zeros((n, n), dtype=np.int64)
    uni = np.zeros(n, dtype=np.int64)
    n_seg = 0
    for seg in segments:
        if not seg:
            continue
        n_seg += 1
        try:
            idx = [index[ch] for ch in seg]
        except KeyError as e:
            raise ValueError(f"Символ {e.args[0]!r} не входит в набор состояний") from None
        for k in idx:
            uni[k] += 1
        for a, b in zip(idx, idx[1:]):
            N[a, b] += 1
    expected = sum(max(len(s) - 1, 0) for s in segments)
    assert int(N.sum()) == expected, "total_counted_pairs != sum(N)"
    return TransitionModel(states=states, N=N, unigrams=uni, n_segments=n_seg)


def probabilities(N: np.ndarray) -> np.ndarray:
    """P[i, j] = N[i, j] / sum_j N[i, j]; строки без исходящих переходов — NaN."""
    N = np.asarray(N, dtype=float)
    rows = N.sum(axis=1, keepdims=True)
    P = np.full_like(N, np.nan)
    np.divide(N, rows, out=P, where=rows > 0)
    return P


def merge(models: list[TransitionModel]) -> TransitionModel:
    """Сумма моделей с одинаковым набором состояний (переходы между ними не создаются)."""
    if not models:
        raise ValueError("Пустой список моделей")
    states = models[0].states
    for m in models[1:]:
        if m.states != states:
            raise ValueError("Разные наборы состояний")
    return TransitionModel(
        states=list(states),
        N=sum(m.N for m in models),
        unigrams=sum(m.unigrams for m in models),
        n_segments=sum(m.n_segments for m in models),
    )
