"""Запуск эксперимента одной командой.

    python -m markov.cli --manifest data/manifest.csv --config config.json --output results/
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from . import compare, plots, report
from .data import CorpusError, load_manifest, read_text
from .preprocess import PreprocessConfig, PreprocessLog, letter_segments, vc_segments
from .transitions import TransitionModel, count_transitions, merge

VC_STATES = ["V", "C"]


def fnum(n: int) -> str:
    """Целое с разделителем разрядов (узкий неразрывный пробел)."""
    return f"{int(n):,}".replace(",", " ")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_matrix(path: Path, M: np.ndarray, states: list[str], fmt: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["from\\to"] + states)
        for s, row in zip(states, M):
            if fmt == "d":
                w.writerow([s] + [str(int(v)) for v in row])
            else:
                w.writerow([s] + ["" if np.isnan(v) else format(v, fmt) for v in row.astype(float)])


def vc_row(m: TransitionModel) -> dict:
    N = m.N
    P = m.P
    v, c = m.unigrams
    return {"N_VV": int(N[0, 0]), "N_VC": int(N[0, 1]), "N_CV": int(N[1, 0]), "N_CC": int(N[1, 1]),
            "row_V": int(N[0].sum()), "row_C": int(N[1].sum()),
            "P_VV": float(P[0, 0]), "P_VC": float(P[0, 1]), "P_CV": float(P[1, 0]), "P_CC": float(P[1, 1]),
            "V_share": float(v / (v + c)), "vc_letters": int(v + c), "vc_pairs": m.total_pairs,
            "vc_segments": m.n_segments}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Марковские модели первого порядка по буквам и V/C")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--output", default="results/")
    args = ap.parse_args(argv)

    try:
        cfg_text = Path(args.config).read_text(encoding="utf-8")
        cfg = json.loads(cfg_text)
        pc = PreprocessConfig.from_dict(cfg)
        records = load_manifest(args.manifest)
    except (CorpusError, ValueError, NotImplementedError, FileNotFoundError) as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2

    out = Path(args.output)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    states = list(cfg["alphabet"])
    thr = int(cfg["low_support_threshold"])
    seed = int(cfg["seed"])
    labels = OrderedDict((a["key"], a["label"]) for a in cfg["authors"])

    # --- 1. Тексты -> сегменты -> модели по текстам -------------------------
    text_letter: dict[str, TransitionModel] = {}
    text_vc: dict[str, TransitionModel] = {}
    text_tokens: dict[str, list[str]] = {}
    logs: dict[str, PreprocessLog] = {k: PreprocessLog() for k in labels}
    by_author: dict[str, list] = {k: [] for k in labels}
    segment_counts: dict[str, Counter] = {k: Counter() for k in labels}  # для примеров слов в отчёте
    text_rows = []
    for r in records:
        if r.author_key not in labels:
            print(f"ОШИБКА: автор {r.author_key!r} (text_id={r.text_id}) отсутствует в config.json", file=sys.stderr)
            return 2
        raw = read_text(r.path)
        log = PreprocessLog()
        segs = letter_segments(raw, pc, log)
        if not segs:
            print(f"ОШИБКА: text_id={r.text_id}: после очистки не осталось букв", file=sys.stderr)
            return 2
        text_letter[r.text_id] = count_transitions(segs, states)
        text_vc[r.text_id] = count_transitions(vc_segments(segs, pc), VC_STATES)
        text_tokens[r.text_id] = raw.split()
        logs[r.author_key].merge(log)
        by_author[r.author_key].append(r)
        segment_counts[r.author_key].update(segs)
        uni = dict(zip(states, text_letter[r.text_id].unigrams.tolist()))
        text_rows.append({
            "text_id": r.text_id, "author": r.author, "year": r.meta.get("year", ""),
            "title": r.title[:90], "tokens": log.tokens, "letters": log.letters,
            "letter_pairs": text_letter[r.text_id].total_pairs, "vc_pairs": text_vc[r.text_id].total_pairs,
            "homoglyph_repaired_tokens": log.tokens_homoglyph_repaired,
            "stress_marks_removed": log.stress_marks_removed,
            "dropped_letters": sum(log.dropped_letters.values()),
            "dropped_other": sum(log.dropped_other.values()),
            **{f"per1000_{ch}": round(1000 * uni[ch] / log.letters, 2) for ch in "ѣіѳёъ"},
            **{f"Pvc_{a}{b}": round(float(text_vc[r.text_id].P[i, j]), 4)
               for i, a in enumerate("VC") for j, b in enumerate("VC")},
        })
    empty = [k for k, v in by_author.items() if not v]
    if empty:
        print(f"ОШИБКА: у авторов нет текстов в манифесте: {empty}", file=sys.stderr)
        return 2

    # --- 2. Модели авторов (полный объём) --------------------------------
    A_letter = {k: merge([text_letter[r.text_id] for r in v]) for k, v in by_author.items()}
    A_vc = {k: merge([text_vc[r.text_id] for r in v]) for k, v in by_author.items()}
    for k in labels:
        assert A_letter[k].total_pairs == int(A_letter[k].N.sum())
        assert A_letter[k].total_pairs == sum(text_letter[r.text_id].total_pairs for r in by_author[k])

    # --- 3. Сбалансированные выборки ------------------------------------
    bal_cfg = cfg["balanced"]
    budget = bal_cfg.get("letters_per_author") or min(logs[k].letters for k in labels)
    B_letter, B_vc = {}, {}
    if bal_cfg.get("enabled", True):
        for k in labels:
            docs = [text_tokens[r.text_id] for r in by_author[k]]
            segs = compare.balanced_sample(docs, budget, int(bal_cfg["chunk_tokens"]), seed, pc)
            B_letter[k] = count_transitions(segs, states)
            B_vc[k] = count_transitions(vc_segments(segs, pc), VC_STATES)

    # --- 4. Таблицы -------------------------------------------------------
    corpus_rows, metrics_rows, vc_rows, top_rows, unsupported_rows = [], [], [], [], []
    for k, lab in labels.items():
        recs = by_author[k]
        years = sorted({r.meta.get("year", "") for r in recs if r.meta.get("year")})
        mags = Counter(r.meta.get("magazine", "") for r in recs)
        lg, mL, mV = logs[k], A_letter[k], A_vc[k]
        corpus_rows.append({
            "author": lab, "author_key": k, "n_texts": len(recs),
            "years": f"{years[0]}–{years[-1]}" if len(years) > 1 else (years[0] if years else ""),
            "magazines": "; ".join(f"{m} ({n})" for m, n in mags.most_common()),
            "titles": " | ".join(r.title[:60] for r in recs),
            "tokens": lg.tokens, "letters": lg.letters, "letter_segments": mL.n_segments,
            "letter_pairs": mL.total_pairs, "vc_segments": mV.n_segments, "vc_pairs": mV.total_pairs,
            "orthography": cfg["preprocess"]["mode"] + " (дореформенная, ѣ/і/ѳ сохранены)",
            "homoglyph_repaired_tokens": lg.tokens_homoglyph_repaired,
            "stress_marks_removed": lg.stress_marks_removed,
            "dropped_letters": sum(lg.dropped_letters.values()),
            "dropped_other": sum(lg.dropped_other.values()),
            "low_support_rows": ", ".join(s for s, f in zip(states, ~compare.supported_rows(mL, thr)) if f),
        })
        for variant, ml, mv in (("full", mL, mV), ("balanced", B_letter.get(k), B_vc.get(k))):
            if ml is None:
                continue
            metrics_rows.append({"author": lab, "author_key": k, "variant": variant,
                                 "letters": int(ml.unigrams.sum()), "letter_segments": ml.n_segments,
                                 "letter_pairs": ml.total_pairs, "sum_N_letters": int(ml.N.sum()),
                                 "observed_rows": int((ml.row_totals > 0).sum()),
                                 "supported_rows": int(compare.supported_rows(ml, thr).sum()),
                                 **vc_row(mv)})
            vc_rows.append({"author": lab, "author_key": k, "variant": variant, **vc_row(mv)})
        for rank, t in enumerate(compare.top_transitions(mL, 10, thr), 1):
            top_rows.append({"author": lab, "rank": rank, **t})
        for ch, n in (lg.dropped_letters + lg.dropped_other).most_common():
            unsupported_rows.append({"author": lab, "char": ch, "codepoint": f"U+{ord(ch):04X}", "count": n,
                                     "kind": "буква вне алфавита" if ch.isalpha() else "не буква"})
        write_matrix(out / f"transitions_letters_N_{k}.csv", mL.N, states, "d")
        write_matrix(out / f"transitions_letters_P_{k}.csv", mL.P, states, ".6f")
        write_matrix(out / f"transitions_vc_N_{k}.csv", mV.N, VC_STATES, "d")
        write_matrix(out / f"transitions_vc_P_{k}.csv", mV.P, VC_STATES, ".6f")
        if k in B_letter:
            write_matrix(out / f"transitions_letters_N_{k}_balanced.csv", B_letter[k].N, states, "d")
            write_matrix(out / f"transitions_letters_P_{k}_balanced.csv", B_letter[k].P, states, ".6f")

    inventory = [{"char": s, "codepoint": f"U+{ord(s):04X}",
                  **{labels[k]: int(A_letter[k].unigrams[i]) for k in labels},
                  **{f"{labels[k]} на 1000": round(1000 * A_letter[k].unigrams[i] / A_letter[k].unigrams.sum(), 2)
                     for k in labels}} for i, s in enumerate(states)]

    # Расстояния и разности
    keys = list(labels)
    dist = {}
    for variant, models in (("full", A_letter), ("balanced", B_letter)):
        if not models:
            continue
        M = np.zeros((len(keys), len(keys)))
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                M[i, j] = compare.weighted_row_tv(models[a], models[b], thr) if i != j else 0.0
        dist[variant] = M
    dist_rows = [{"variant": v, "author_a": labels[a], "author_b": labels[b], "weighted_tv": round(float(M[i, j]), 5)}
                 for v, M in dist.items() for i, a in enumerate(keys) for j, b in enumerate(keys) if i < j]

    diff_tables = {}
    for a, b in cfg["difference_pairs"]:
        rows = compare.difference_table(A_letter[a], A_letter[b], thr, k=15)
        diff_tables[(a, b)] = rows
        write_csv(out / f"differences_{a}_{b}.csv", rows)

    write_csv(out / "corpus_table.csv", corpus_rows)
    write_csv(out / "metrics.csv", metrics_rows)
    write_csv(out / "vc_summary.csv", vc_rows)
    write_csv(out / "top10_letter_transitions.csv", top_rows)
    write_csv(out / "unsupported_chars.csv", unsupported_rows)
    write_csv(out / "char_inventory_by_author.csv", inventory)
    write_csv(out / "per_text_stats.csv", text_rows)
    write_csv(out / "distances_letters.csv", dist_rows)

    # --- 5a. Вариация внутри автора и между авторами (P4) ------------------
    wb_rows, wb = [], None
    wbc = cfg.get("within_between", {})
    if wbc.get("enabled"):
        wb = compare.within_between({k: [text_tokens[r.text_id] for r in by_author[k]] for k in keys}, states,
                                    int(wbc["letters_each"]), int(wbc["repeats"]), int(bal_cfg["chunk_tokens"]),
                                    seed, thr, pc)
        for k in keys:
            v = wb["within"][k]
            wb_rows.append({"kind": "внутри", "author_a": labels[k], "author_b": labels[k],
                            "mean": round(float(np.mean(v)), 5), "min": round(float(np.min(v)), 5),
                            "max": round(float(np.max(v)), 5), "repeats": len(v),
                            "skipped_repeats": wb["skipped"][k]})
        for (a, b), v in wb["between"].items():
            wb_rows.append({"kind": "между", "author_a": labels[a], "author_b": labels[b],
                            "mean": round(float(np.mean(v)), 5), "min": round(float(np.min(v)), 5),
                            "max": round(float(np.max(v)), 5), "repeats": len(v), "skipped_repeats": ""})
        write_csv(out / "within_between.csv", wb_rows)
        plots.within_between_plot(
            wb, labels, figs / "within_between.png",
            "Различаются ли авторы сильнее, чем две группы произведений одного автора?",
            f"Тексты автора делятся на 2 непересекающиеся группы; из каждой — выборка ≈{fnum(wbc['letters_each'])} букв; "
            f"{wbc['repeats']} повторов (seed {seed}…{seed + int(wbc['repeats']) - 1}); строки с ≥ {thr} переходами")

    # --- 5. Пробная атрибуция (P4) ---------------------------------------
    probe_rows = []
    pr = cfg.get("attribution_probe", {})
    if pr.get("enabled"):
        sel = {tid: (r.author_key, text_letter[tid]) for r in records for tid in [r.text_id]
               if text_letter[tid].unigrams.sum() >= pr["min_text_letters"]}
        probe_rows = compare.attribution_probe(sel, A_letter, float(pr["smoothing_alpha"]))
        for p in probe_rows:
            p["true_author"] = labels[p["true_author"]]
            p["predicted"] = labels[p["predicted"]]
        write_csv(out / "attribution_probe.csv",
                  [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()} for p in probe_rows])

    # --- 6. Рисунки ------------------------------------------------------
    vmax = float(np.nanmax([np.nanmax(A_letter[k].P) for k in keys]))
    nmax = float(max(A_letter[k].N.max() for k in keys))
    for k, lab in labels.items():
        m, mv = A_letter[k], A_vc[k]
        low = ~compare.supported_rows(m, thr)
        sub = (f"{len(by_author[k])} текстов · {fnum(m.unigrams.sum())} букв · {fnum(m.total_pairs)} пар внутри слов\n"
               f"курсив — строка с < {thr} исходящих переходов (low_support); штриховка — переходов нет")
        plots.letter_heatmap(m.P, states, f"{lab}: переходы буква → буква, P(j | i)", sub,
                             figs / f"letter_heatmap_{k}.png", vmax, low)
        plots.counts_heatmap(m.N, states, f"{lab}: число переходов N(i → j)", sub,
                             figs / f"letter_counts_{k}.png", nmax)
        plots.vc_heatmap(mv.N, f"{lab}: гласная / согласная",
                         f"{fnum(mv.total_pairs)} пар V/C; ъ и ь разрывают цепочку",
                         figs / f"vc_heatmap_{k}.png")
        plots.top_graph(m.N, states, int(cfg["top_graph_edges"]),
                        f"{lab}: {cfg['top_graph_edges']} самых частых переходов",
                        "Дуги выбраны по числу N(i→j); толщина и подпись — вероятность P(j | i)",
                        figs / f"letter_top_graph_{k}.png")
    vlim = 0.0
    for a, b in cfg["difference_pairs"]:
        mask = ~(compare.supported_rows(A_letter[a], thr) & compare.supported_rows(A_letter[b], thr))
        D = A_letter[a].P - A_letter[b].P
        vlim = max(vlim, float(np.nanmax(np.abs(D[~mask]))))
    vlim = min(vlim, 0.15)
    for a, b in cfg["difference_pairs"]:
        mask = ~(compare.supported_rows(A_letter[a], thr) & compare.supported_rows(A_letter[b], thr))
        plots.difference_heatmap(
            A_letter[a].P - A_letter[b].P, states,
            f"Разность P: {labels[a]} − {labels[b]}",
            f"красный — переход вероятнее у автора {labels[a]}, синий — у автора {labels[b]}; шкала обрезана на ±{vlim:.2f}\n"
            f"штриховка и курсив — строка с < {thr} переходами хотя бы у одного автора (не сравнивается)",
            figs / f"letter_difference_{a}_{b}.png", vlim, mask)
    min_letters = int(cfg.get("attribution_probe", {}).get("min_text_letters", 5000))
    iqr = {}
    for k in keys:
        per = [vc_row(text_vc[r.text_id]) for r in by_author[k]
               if text_letter[r.text_id].unigrams.sum() >= min_letters]
        if len(per) >= 3:
            iqr[k] = {m: [float(np.percentile([x[m] for x in per], 25)), float(np.percentile([x[m] for x in per], 75))]
                      for m in ("P_VV", "P_VC", "P_CV", "P_CC", "V_share")}
    for variant, suffix, models in (("полный объём", "", A_vc), ("равные выборки", "_balanced", B_vc)):
        if not models:
            continue
        rows = [{"label": labels[k], **vc_row(models[k]), **({"iqr": iqr[k]} if k in iqr else {})} for k in keys]
        note = ("точка — все отобранные тексты автора" if not suffix else
                f"точка — выборка ≈{fnum(budget)} букв на автора (фрагменты по {bal_cfg['chunk_tokens']} токенов, seed={seed})")
        plots.vc_comparison(rows, figs / f"vc_comparison{suffix}.png",
                            f"Переходы гласная/согласная по авторам ({variant})",
                            note + f"; серая полоса — межквартильный размах по отдельным текстам ≥ {fnum(min_letters)} букв "
                            "(у авторов с ≥ 3 такими текстами); ось X не от нуля")
    for variant, M in dist.items():
        plots.distance_heatmap(M, [labels[k] for k in keys],
                               "Различие буквенных моделей между авторами" +
                               (" (равные выборки)" if variant == "balanced" else " (полный объём)"),
                               f"взвешенное TV по строкам с ≥ {thr} переходами у обоих авторов",
                               figs / f"letter_distance_{variant}.png")

    # --- 7. Сведения о запуске и отчёт -----------------------------------
    import matplotlib
    import networkx
    run_info = {
        "config_sha256": hashlib.sha256(cfg_text.encode("utf-8")).hexdigest(),
        "manifest": str(args.manifest), "n_texts": len(records), "seed": seed,
        "balanced_letters_per_author": budget, "low_support_threshold": thr,
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "networkx": networkx.__version__,
    }
    (out / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    report.write_summary(out, cfg=cfg, labels=labels, states=states, by_author=by_author,
                         corpus_rows=corpus_rows, metrics_rows=metrics_rows, vc_rows=vc_rows,
                         A_letter=A_letter, B_letter=B_letter, dist=dist, diff_tables=diff_tables,
                         top_rows=top_rows, probe_rows=probe_rows, logs=logs, run_info=run_info,
                         wb_rows=wb_rows, iqr=iqr, segment_counts=segment_counts, wb=wb,
                         text_rows=text_rows)
    print(f"Готово: {len(records)} текстов, {len(keys)} авторов. Отчёт: {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
