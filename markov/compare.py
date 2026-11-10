"""Сравнение моделей авторов на общем наборе состояний."""
from __future__ import annotations

import random

import numpy as np

from .preprocess import PreprocessConfig, letter_segments
from .transitions import TransitionModel


def supported_rows(m: TransitionModel, threshold: int) -> np.ndarray:
    """Строки с числом исходящих переходов ≥ threshold (технический фильтр low_support)."""
    return m.row_totals >= threshold


def weighted_row_tv(a: TransitionModel, b: TransitionModel, threshold: int) -> float:
    """Взвешенное расстояние полной вариации между строками P_a и P_b.

    D = Σ_i w_i · ½ Σ_j |P_a(i,j) − P_b(i,j)| по строкам, достаточно
    обеспеченным в обеих моделях; w_i — средняя доля исходящих переходов
    из i (нормированная на учтённые строки). 0 — одинаковые модели, 1 — максимум.
    Описательная мера, не статистический критерий.
    """
    mask = supported_rows(a, threshold) & supported_rows(b, threshold)
    if not mask.any():
        return float("nan")
    wa = a.row_totals / a.total_pairs
    wb = b.row_totals / b.total_pairs
    w = ((wa + wb) / 2)[mask]
    w = w / w.sum()
    tv = 0.5 * np.abs(a.P[mask] - b.P[mask]).sum(axis=1)
    return float((w * tv).sum())


def top_transitions(m: TransitionModel, k: int, threshold: int) -> list[dict]:
    """k самых частых переходов (по N) с вероятностью и знаменателем строки."""
    order = np.argsort(-m.N, axis=None, kind="stable")[:k]
    rows = m.row_totals
    P = m.P
    out = []
    for flat in order:
        i, j = np.unravel_index(flat, m.N.shape)
        if m.N[i, j] == 0:
            break
        out.append({"from": m.states[i], "to": m.states[j], "count": int(m.N[i, j]),
                    "row_total": int(rows[i]), "probability": float(P[i, j]),
                    "share_of_all_pairs": float(m.N[i, j] / m.total_pairs),
                    "low_support": bool(rows[i] < threshold)})
    return out


def difference_table(a: TransitionModel, b: TransitionModel, threshold: int, k: int = 15) -> list[dict]:
    """Ячейки с наибольшим |P_a − P_b| среди строк, обеспеченных в обеих моделях."""
    mask = supported_rows(a, threshold) & supported_rows(b, threshold)
    D = a.P - b.P
    D[~mask, :] = np.nan
    flat = np.argsort(-np.nan_to_num(np.abs(D), nan=-1), axis=None, kind="stable")[:k]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, D.shape)
        if np.isnan(D[i, j]):
            break
        out.append({"from": a.states[i], "to": a.states[j],
                    "P_a": float(a.P[i, j]), "N_a": int(a.N[i, j]), "row_a": int(a.row_totals[i]),
                    "P_b": float(b.P[i, j]), "N_b": int(b.N[i, j]), "row_b": int(b.row_totals[i]),
                    "diff": float(D[i, j])})
    return out


def balanced_sample(docs: list[list[str]], letter_budget: int, chunk_tokens: int, seed: int,
                    preprocess: PreprocessConfig) -> list[str]:
    """Воспроизводимая выборка фрагментов по ``chunk_tokens`` подряд идущих токенов.

    Фрагменты всех текстов автора перемешиваются генератором с ``seed`` и
    берутся по очереди, пока число букв не достигнет ``letter_budget``.
    Возвращает буквенные сегменты (переходы между фрагментами не создаются).
    """
    chunks = [(d, s) for d, toks in enumerate(docs) for s in range(0, len(toks), chunk_tokens)]
    random.Random(seed).shuffle(chunks)
    segments: list[str] = []
    letters = 0
    for d, s in chunks:
        segs = letter_segments(" ".join(docs[d][s:s + chunk_tokens]), preprocess)
        segments.extend(segs)
        letters += sum(len(x) for x in segs)
        if letters >= letter_budget:
            return segments
    raise ValueError(f"Недостаточно текста: {letters} букв < {letter_budget}")


def split_texts(letters: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
    """Случайное разбиение текстов на две группы, близкие по числу букв (жадно после перемешивания)."""
    order = list(range(len(letters)))
    rng.shuffle(order)
    g1, g2, n1, n2 = [], [], 0, 0
    for i in order:
        if n1 <= n2:
            g1.append(i); n1 += letters[i]
        else:
            g2.append(i); n2 += letters[i]
    return g1, g2


def within_between(author_docs: dict[str, list[list[str]]], states: list[str], letters_each: int,
                   repeats: int, chunk_tokens: int, seed: int, threshold: int,
                   preprocess: PreprocessConfig) -> dict:
    """Внутриавторская и межавторская вариация на выборках равного объёма из РАЗНЫХ текстов.

    В повторе r тексты каждого автора случайно делятся на две непересекающиеся
    группы (seed + r); из каждой группы берётся выборка ≈letters_each букв.
    «Внутри» = D(A1, A2) — модели по разным произведениям одного автора;
    «между» = D(A1, B1). Объёмы равны, поэтому смещение расстояния из-за
    конечной выборки одинаково для обеих величин. Если у автора в повторе
    не удаётся набрать две группы нужного объёма, разбиение пересоздаётся
    (до 50 попыток), иначе повтор для автора пропускается.
    """
    from .transitions import count_transitions

    keys = list(author_docs)
    within = {k: [] for k in keys}
    between = {(a, b): [] for i, a in enumerate(keys) for b in keys[i + 1:]}
    letter_counts = {k: [sum(len(x) for x in letter_segments(" ".join(d), preprocess)) for d in docs]
                     for k, docs in author_docs.items()}
    skipped = {k: 0 for k in keys}
    for r in range(repeats):
        rng = random.Random(seed + r)
        models = {}
        for k in keys:
            for _ in range(50):
                g1, g2 = split_texts(letter_counts[k], rng)
                if min(sum(letter_counts[k][i] for i in g)
                       for g in (g1, g2)) >= letters_each:
                    break
            else:
                skipped[k] += 1
                continue
            s1 = balanced_sample([author_docs[k][i] for i in g1], letters_each, chunk_tokens, seed + r, preprocess)
            s2 = balanced_sample([author_docs[k][i] for i in g2], letters_each, chunk_tokens, seed + r, preprocess)
            models[k] = (count_transitions(s1, states), count_transitions(s2, states))
        for k in models:
            within[k].append(weighted_row_tv(models[k][0], models[k][1], threshold))
        for a, b in between:
            if a in models and b in models:
                between[(a, b)].append(weighted_row_tv(models[a][0], models[b][0], threshold))
    return {"within": within, "between": between, "skipped": skipped}


def log_likelihood(text_N: np.ndarray, author_N: np.ndarray, alpha: float) -> float:
    """Σ n_ij · log p̂_ij, p̂_ij = (N_ij + α) / (N_i + α·K) — по схеме Хмелёва (2000).

    Сглаживание применяется только здесь (пробная атрибуция) и не смешивается
    с описательными несглаженными матрицами.
    """
    K = author_N.shape[1]
    p = (author_N + alpha) / (author_N.sum(axis=1, keepdims=True) + alpha * K)
    return float((text_N * np.log(p)).sum())


def attribution_probe(text_models: dict[str, tuple[str, TransitionModel]],
                      author_models: dict[str, TransitionModel], alpha: float) -> list[dict]:
    """Leave-one-text-out: модель «своего» автора строится без проверяемого текста."""
    out = []
    for tid, (true_author, tm) in text_models.items():
        scores = {}
        for author, am in author_models.items():
            N = am.N - tm.N if author == true_author else am.N
            scores[author] = log_likelihood(tm.N, N, alpha) / max(tm.total_pairs, 1)
        ranked = sorted(scores, key=scores.get, reverse=True)
        out.append({"text_id": tid, "true_author": true_author, "predicted": ranked[0],
                    "correct": ranked[0] == true_author, "true_rank": ranked.index(true_author) + 1,
                    "pairs": tm.total_pairs,
                    "margin_per_pair": scores[ranked[0]] - scores[ranked[1]],
                    **{f"score_{a}": s for a, s in scores.items()}})
    return out
