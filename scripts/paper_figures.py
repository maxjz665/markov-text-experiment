"""Рисунки для тезисов (ИТСиТ-2026): чёрно-белые, Times New Roman, без заголовков внутри.

Берёт готовые результаты markov.cli из results/ и пишет PNG 300 dpi в results/paper/.

    python scripts/paper_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "paper"
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": 300,
})


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    states = rows[0][1:]
    M = np.array([[float(x) if x else np.nan for x in r[1:]] for r in rows[1:]])
    return states, M


def fig_matrix() -> Path:
    states, P = read_matrix(RES / "transitions_letters_P_dostoevsky_fm.csv")
    fig, ax = plt.subplots(figsize=(12 * CM, 10.6 * CM))
    im = ax.imshow(P, cmap="Greys", norm=PowerNorm(0.5, vmin=0, vmax=1), interpolation="nearest")
    ax.set_xticks(range(len(states)), states, fontsize=7.5)
    ax.set_yticks(range(len(states)), states, fontsize=7.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("следующая буква j", labelpad=4)
    ax.set_ylabel("текущая буква i", labelpad=2)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_ticks([0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.set_yticklabels(["0", "0,01", "0,05", "0,1", "0,2", "0,4", "0,6", "0,8", "1"], fontsize=7.5)
    cb.set_label("P(j | i)", fontsize=9)
    cb.outline.set_linewidth(0.4)
    path = OUT / "fig1_matrix_dostoevsky_fm.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path


def fig_within_between() -> Path:
    order = ["Ф. М. Достоевский", "М. М. Достоевский", "А. А. Григорьев", "Н. Н. Страхов",
             "В. П. Мещерский", "П. К. Щебальский", "В. Ф. Пуцыкович"]
    within, between = {}, {a: [] for a in order}
    with open(RES / "within_between.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "внутри":
                within[r["author_a"]] = (float(r["mean"]), float(r["min"]), float(r["max"]))
            else:
                between[r["author_a"]].append(float(r["mean"]))
                between[r["author_b"]].append(float(r["mean"]))
    fig, ax = plt.subplots(figsize=(15 * CM, 6.2 * CM))
    for y, a in enumerate(order):
        m, lo, hi = within[a]
        ax.plot([lo, hi], [y, y], color="0.45", lw=2.2, solid_capstyle="butt", zorder=1)
        ax.plot(between[a], [y] * len(between[a]), "o", ms=4.5, mfc="white", mec="black", mew=0.8, zorder=2)
        ax.plot([m], [y], "o", ms=5, color="black", zorder=3)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlim(0.07, 0.15)
    ax.set_xticks(np.arange(0.07, 0.151, 0.01))
    ax.set_xticklabels([f"{x:.2f}".replace(".", ",") for x in np.arange(0.07, 0.151, 0.01)])
    ax.set_xlabel("расстояние D между буквенными моделями")
    ax.grid(axis="x", color="0.85", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h1, = ax.plot([], [], "o", ms=5, color="black")
    h2, = ax.plot([], [], color="0.45", lw=2.2)
    h3, = ax.plot([], [], "o", ms=4.5, mfc="white", mec="black", mew=0.8)
    ax.legend([h1, h2, h3], ["внутри автора, среднее", "внутри автора, мин.–макс.", "до другого автора"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8.5,
              handletextpad=0.4, columnspacing=1.5)
    path = OUT / "fig2_within_between.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (fig_matrix(), fig_within_between()):
        print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
