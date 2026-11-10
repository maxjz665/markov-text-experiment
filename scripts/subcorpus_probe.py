"""Пробная атрибуция по Хмелёву на подвыборках «один журнал, одни годы».

Дополнение к основному расчёту (markov.cli не меняется). Проверяется, как меняется
доля верных ответов, если сравнивать авторов внутри одного издания и если исключить
слова с заглавной буквы (имена собственные и начала предложений), как у Д. В. Хмелёва.

    python scripts/subcorpus_probe.py            # пишет results/subcorpus_probe.csv и .md
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from markov import compare  # noqa: E402
from markov.data import load_manifest, read_text  # noqa: E402
from markov.preprocess import PreprocessConfig, letter_segments  # noqa: E402
from markov.transitions import count_transitions, merge  # noqa: E402

# (название, журнал, годы, авторы, модели только по подвыборке?)
# Строки «модели по всем изданиям» — контроль: те же проверяемые тексты и те же кандидаты,
# но модели авторов строятся по всем их текстам. Разница показывает вклад ограничения изданием.
TRIO_V = ["dostoevsky_fm", "grigoryev", "strakhov"]
TRIO_G = ["dostoevsky_fm", "meshchersky", "putsykovich"]
SETS = [
    ("Все издания, 7 авторов", None, None, None, True),
    ("«Время» 1861–1863", "Время", (1861, 1863), TRIO_V, True),
    ("«Время» 1861–1863, модели по всем изданиям", "Время", (1861, 1863), TRIO_V, False),
    ("«Гражданин» 1872–1874", "Гражданин", (1872, 1874), TRIO_G, True),
    ("«Гражданин» 1872–1874, модели по всем изданиям", "Гражданин", (1872, 1874), TRIO_G, False),
]


def drop_capitalized(raw: str) -> str:
    return " ".join(t for t in raw.split() if not t[:1].isupper())


def main() -> int:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    pc = PreprocessConfig.from_dict(cfg)
    states = list(cfg["alphabet"])
    alpha = float(cfg["attribution_probe"]["smoothing_alpha"])
    min_letters = int(cfg["attribution_probe"].get("min_text_letters", 5000))
    labels = {a["key"]: a["label"] for a in cfg["authors"]}
    records = load_manifest(ROOT / "data" / "manifest.csv")

    rows, summary = [], []
    for mode, transform in (("все слова", None), ("без слов с заглавной", drop_capitalized)):
        models = {}
        for r in records:
            raw = read_text(r.path)
            if transform:
                raw = transform(raw)
            models[r.text_id] = (r, count_transitions(letter_segments(raw, pc), states))
        for name, magazine, years, authors, own_models in SETS:
            pool = {tid: (r, m) for tid, (r, m) in models.items()
                    if authors is None or r.author_key in authors}
            sel = {tid: (r, m) for tid, (r, m) in pool.items()
                   if (magazine is None or r.meta.get("magazine") == magazine)
                   and (years is None or years[0] <= int(r.meta.get("year") or 0) <= years[1])}
            train = sel if own_models else pool
            keys = sorted({r.author_key for r, _ in sel.values()}, key=list(labels).index)
            author_models = {k: merge([m for r, m in train.values() if r.author_key == k]) for k in keys}
            tests = {tid: (r.author_key, m) for tid, (r, m) in sel.items() if m.unigrams.sum() >= min_letters}
            res = compare.attribution_probe(tests, author_models, alpha)
            n_ok = sum(x["correct"] for x in res)
            counts = Counter(x["true_author"] for x in res)
            summary.append({
                "set": name, "mode": mode, "authors": ", ".join(labels[k] for k in keys),
                "texts_in_models": len(train), "texts_tested": len(res), "correct": n_ok,
                "share": round(n_ok / len(res), 3) if res else "",
                "majority_share": round(max(counts.values()) / len(res), 3) if res else "",
                "by_author": "; ".join(
                    f"{labels[k]} {sum(x['correct'] for x in res if x['true_author'] == k)}/{counts[k]}"
                    for k in keys),
            })
            for x in res:
                rows.append({"set": name, "mode": mode, "text_id": x["text_id"],
                             "true_author": labels[x["true_author"]], "predicted": labels[x["predicted"]],
                             "correct": x["correct"], "pairs": x["pairs"],
                             "title": sel[x["text_id"]][0].title[:80]})

    out = ROOT / "results"
    with open(out / "subcorpus_probe.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# Проба по Хмелёву на подвыборках", "",
             f"Leave-one-text-out, тексты ≥ {min_letters} букв, сглаживание α = {alpha}. "
             "Модели авторов строятся только по текстам подвыборки.", "",
             "| Подвыборка | Слова | Авторы | Текстов в моделях | Проверено | Верно | Доля | Доля самого частого автора | По авторам |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['set']} | {s['mode']} | {s['authors']} | {s['texts_in_models']} | {s['texts_tested']} | "
                     f"{s['correct']} | {s['share']} | {s['majority_share']} | {s['by_author']} |")
    (out / "subcorpus_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
