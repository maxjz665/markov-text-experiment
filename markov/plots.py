"""Рисунки: тепловые карты, графы переходов, сравнение V/C.

Шрифт DejaVu Sans (входит в Matplotlib) содержит кириллицу и ѣ/і/ѳ.
Палитры: последовательная — один синий тон от светлого к тёмному;
расхождение — синий ↔ красный с нейтральным серым центром;
отсутствующие/ненадёжные строки — штриховка на фоне, а не цвет шкалы.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# Категориальные цвета в фиксированном порядке (проверены валидатором палитры)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#9e2020"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
})


def _hatch_rows(ax, rows: np.ndarray, n_cols: int) -> None:
    for i in np.flatnonzero(rows):
        ax.add_patch(Rectangle((-0.5, i - 0.5), n_cols, 1, fill=False, hatch="////",
                               edgecolor="#c3c2b7", linewidth=0))


def _letter_axes(ax, states: list[str], low: np.ndarray | None = None) -> None:
    n = len(states)
    ax.set_xticks(range(n), states)
    ax.set_yticks(range(n), states)
    ax.tick_params(length=0, labelsize=9)
    ax.xaxis.tick_top()
    ax.set_xlabel("следующая буква j", labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("текущая буква i")
    if low is not None:
        for lbl, flag in zip(ax.get_yticklabels(), low):
            if flag:
                lbl.set_color(MUTED)
                lbl.set_fontstyle("italic")
    for s in ax.spines.values():
        s.set_visible(False)


def letter_heatmap(P: np.ndarray, states: list[str], title: str, subtitle: str, path: Path,
                   vmax: float, low: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 9.6))
    data = np.ma.masked_invalid(P)
    im = ax.imshow(data, cmap=SEQ, norm=PowerNorm(0.5, vmin=0, vmax=vmax), interpolation="nearest")
    _hatch_rows(ax, np.isnan(P).all(axis=1), len(states))
    _letter_axes(ax, states, low)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_ticks([0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_label("P(j | i) — вероятность перехода (шкала квадратного корня, общая для всех авторов)")
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.06, ha="left", fontsize=14, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9, fontweight="normal", color=INK2, pad=34)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def counts_heatmap(N: np.ndarray, states: list[str], title: str, subtitle: str, path: Path, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 9.6))
    data = np.ma.masked_less_equal(N.astype(float), 0)
    im = ax.imshow(data, cmap=SEQ, norm=LogNorm(vmin=1, vmax=vmax), interpolation="nearest")
    _letter_axes(ax, states)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("N(i→j) — число переходов (лог. шкала; пусто = 0)")
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.06, ha="left", fontsize=14, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9, fontweight="normal", color=INK2, pad=34)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def difference_heatmap(D: np.ndarray, states: list[str], title: str, subtitle: str, path: Path,
                       vlim: float, masked_rows: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 9.6))
    D = D.copy()
    D[masked_rows, :] = np.nan
    im = ax.imshow(np.ma.masked_invalid(D), cmap=DIV, norm=TwoSlopeNorm(0, -vlim, vlim),
                   interpolation="nearest")
    _hatch_rows(ax, masked_rows, len(states))
    _letter_axes(ax, states, masked_rows)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
    cb.set_label("P_A(j|i) − P_B(j|i)")
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.06, ha="left", fontsize=14, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9, fontweight="normal", color=INK2, pad=34)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def vc_heatmap(N: np.ndarray, title: str, subtitle: str, path: Path) -> None:
    P = N / N.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(P, cmap=SEQ, vmin=0, vmax=1)
    labels = ["V (гласная)", "C (согласная)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_xlabel("следующий класс", labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("текущий класс")
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{P[i, j]:.3f}\nN = {N[i, j]:,}".replace(",", " "), ha="center", va="center",
                    fontsize=12, color="white" if P[i, j] > 0.55 else INK)
    fig.suptitle(title, x=0.04, ha="left", fontsize=13, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9, fontweight="normal", color=INK2, pad=34)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def top_graph(N: np.ndarray, states: list[str], k: int, title: str, subtitle: str, path: Path) -> None:
    """Граф k самых частых переходов: толщина дуги ~ P(j|i), подпись — P."""
    import networkx as nx

    P = N / np.where(N.sum(axis=1, keepdims=True) > 0, N.sum(axis=1, keepdims=True), 1)
    flat = np.argsort(-N, axis=None, kind="stable")[:k]
    G = nx.DiGraph()
    for f in flat:
        i, j = np.unravel_index(f, N.shape)
        G.add_edge(states[i], states[j], p=P[i, j], n=int(N[i, j]))
    nodes = [s for s in states if s in G]
    pos = {s: (math.cos(2 * math.pi * t / len(nodes) + math.pi / 2),
               math.sin(2 * math.pi * t / len(nodes) + math.pi / 2)) for t, s in enumerate(nodes)}
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_axis_off()
    widths = [1 + 7 * G[u][v]["p"] for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#6da7ec", alpha=0.85, arrows=True,
                           arrowstyle="-|>", arrowsize=16, node_size=900,
                           connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900, node_color=SURFACE, edgecolors="#184f95", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=15, font_family="DejaVu Sans", font_color=INK)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels={(u, v): f"{G[u][v]['p']:.2f}" for u, v in G.edges},
                                 font_size=8, font_color=INK2, label_pos=0.35, connectionstyle="arc3,rad=0.12",
                                 bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.9})
    fig.suptitle(title, x=0.04, ha="left", fontsize=14, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9, fontweight="normal", color=INK2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def vc_comparison(rows: list[dict], path: Path, title: str, subtitle: str) -> None:
    """Точечная диаграмма: 4 вероятности V/C и доля гласных по авторам.

    Точка — объединённая модель автора; серая линия — межквартильный размах
    тех же величин по отдельным текстам автора (если передан ключ ``iqr``).
    """
    metrics = [("P(V→V)", "P_VV"), ("P(V→C)", "P_VC"), ("P(C→V)", "P_CV"), ("P(C→C)", "P_CC"),
               ("доля V среди букв V/C", "V_share")]
    n_a = len(rows)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 0.55 * n_a + 2.2), sharey=True)
    ys = np.arange(n_a)[::-1]
    for ax, (label, key) in zip(axes, metrics):
        for k, (r, y) in enumerate(zip(rows, ys)):
            iqr = r.get("iqr", {}).get(key)
            if iqr is not None:
                ax.plot(iqr, [y, y], color="#c3c2b7", linewidth=4, solid_capstyle="round", zorder=1)
            ax.scatter([r[key]], [y], s=70, color=SERIES[k], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{r[key]:.3f}", (r[key], y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.margins(x=0.25, y=0.12)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(ys, [r["label"] for r in rows])
    axes[0].tick_params(axis="y", length=0)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.01, 0.93, subtitle, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def within_between_plot(wb: dict, labels: dict, path: Path, title: str, subtitle: str) -> None:
    """Для каждого автора: расстояние между двумя его выборками и до выборок других авторов."""
    keys = list(labels)
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(keys) + 2.2))
    ys = np.arange(len(keys))[::-1]
    for k, (a, y) in enumerate(zip(keys, ys)):
        others = []
        for (p, q), vals in wb["between"].items():
            if a in (p, q):
                others.append((np.mean(vals), labels[q if p == a else p]))
        xs = [o[0] for o in others]
        ax.plot([min(xs), max(xs)], [y, y], color=GRID, linewidth=1, zorder=0)
        ax.scatter(xs, [y] * len(xs), s=40, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.2, zorder=2,
                   label="до другого автора" if k == 0 else None)
        w = wb["within"][a]
        ax.plot([min(w), max(w)], [y, y], color=SERIES[k], linewidth=3, alpha=0.4, solid_capstyle="round", zorder=1)
        ax.scatter([np.mean(w)], [y], s=80, color=SERIES[k], edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="внутри автора (среднее; полоса — мин–макс по повторам)" if k == 0 else None)
        near = min(others)
        ax.annotate(f"ближайший: {near[1]}", (near[0], y), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=7, color=INK2)
    ax.set_yticks(ys, [labels[a] for a in keys])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("взвешенное расстояние полной вариации между буквенными моделями")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    ax.margins(y=0.1)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9, fontweight="normal", color=INK2, pad=26)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def distance_heatmap(M: np.ndarray, labels: list[str], title: str, subtitle: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = np.nanmax(M)
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.6 * vmax else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("взвешенное расстояние полной вариации")
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9, fontweight="normal", color=INK2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
