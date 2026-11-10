"""Сборка интерактивной демонстрации results/demo.html из результатов эксперимента.

Запуск (после python -m markov.cli):
    python scripts/build_demo.py

Шаблон — demo/template.html; данные встраиваются в страницу как JSON,
поэтому страница открывается без сервера и без интернета (кроме шрифтов).
Результат — results/demo.html (для показа), results/demo_explained.html (личная копия
с пояснениями «Простыми словами / Зачем») и results/demo_text.html (то же без подписей
«Простыми словами» и «Зачем»). Все файлы открываются двойным щелчком,
все данные внутри, из сети подгружаются только шрифты Google Fonts (без сети — системные).
"""
from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from markov.data import load_manifest, read_text  # noqa: E402
from markov.preprocess import PreprocessConfig, letter_segments  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_matrix(path: Path) -> list[list[int]]:
    rows = read_csv(path)
    return [[int(r[c]) for c in list(r.keys())[1:]] for r in rows]


def pct(vals: list[float], q: float) -> float:
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> None:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    res = ROOT / "results"
    pc = PreprocessConfig.from_dict(cfg)
    states = list(cfg["alphabet"])
    sidx = {s: i for i, s in enumerate(states)}
    authors = cfg["authors"]
    keys = [a["key"] for a in authors]
    label = {a["key"]: a["label"] for a in authors}
    by_label = {a["label"]: a["key"] for a in authors}
    records = load_manifest(ROOT / "data" / "manifest.csv")

    # Слова (сегменты) по авторам: начала/концы слов и примеры слов для каждой клетки
    seg_counts = {k: Counter() for k in keys}
    for r in records:
        seg_counts[r.author_key].update(letter_segments(read_text(r.path), pc))

    data_authors = []
    corpus = {r["author_key"]: r for r in read_csv(res / "corpus_table.csv")}
    vc = {(r["author_key"], r["variant"]): r for r in read_csv(res / "vc_summary.csv")}
    per_text = read_csv(res / "per_text_stats.csv")
    min_letters = cfg["attribution_probe"]["min_text_letters"]
    for k in keys:
        N = read_matrix(res / f"transitions_letters_N_{k}.csv")
        starts, ends = [0] * len(states), [0] * len(states)
        cell_words: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for w, c in seg_counts[k].items():
            starts[sidx[w[0]]] += c
            ends[sidx[w[-1]]] += c
            seen = Counter(zip(w, w[1:]))
            for (a, b), m in seen.items():
                cell_words[(sidx[a], sidx[b])][w] += m * c
        words = {f"{i},{j}": [[w, c] for w, c in cnt.most_common(3)] for (i, j), cnt in cell_words.items()}
        c = corpus[k]
        texts = [t for t in per_text if t["author"] == label[k] and int(t["letters"]) >= min_letters]
        iqr = {}
        if len(texts) >= 3:
            for m in ("VV", "VC", "CV", "CC"):
                v = [float(t[f"Pvc_{m}"]) for t in texts]
                iqr[m] = [round(pct(v, 0.25), 4), round(pct(v, 0.75), 4)]
        data_authors.append({
            "key": k, "label": label[k], "n_texts": int(c["n_texts"]), "years": c["years"],
            "magazines": c["magazines"], "tokens": int(c["tokens"]), "letters": int(c["letters"]),
            "letter_pairs": int(c["letter_pairs"]), "vc_pairs": int(c["vc_pairs"]),
            "N": N, "starts": starts, "ends": ends, "words": words,
            "vc": {v: {m: int(vc[(k, v)][f"N_{m}"]) for m in ("VV", "VC", "CV", "CC")}
                   for v in ("full", "balanced")},
            "vc_iqr": iqr,
            "titles": [[r.text_id, r.meta.get("year", ""), r.title] for r in records if r.author_key == k],
        })

    wb = [{"kind": r["kind"], "a": by_label[r["author_a"]], "b": by_label[r["author_b"]],
           "mean": float(r["mean"]), "min": float(r["min"]), "max": float(r["max"])}
          for r in read_csv(res / "within_between.csv")]
    titles = {r.text_id: r.title for r in records}
    probe = [{"id": r["text_id"], "true": by_label[r["true_author"]], "pred": by_label[r["predicted"]],
              "ok": r["correct"] == "True", "title": titles.get(r["text_id"], "")}
             for r in read_csv(res / "attribution_probe.csv")]
    run = json.loads((res / "run_info.json").read_text(encoding="utf-8"))

    data = {
        "states": states, "vowels": cfg["vowels"], "consonants": cfg["consonants"],
        "historical": "ѣіѳ", "threshold": cfg["low_support_threshold"],
        "balanced_letters": run["balanced_letters_per_author"],
        "within_letters": cfg["within_between"]["letters_each"], "within_repeats": cfg["within_between"]["repeats"],
        "authors": data_authors, "within_between": wb, "probe": probe,
        "dump": cfg["source"]["dump_path"].split("/")[-1], "seed": cfg["seed"],
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tpl = (ROOT / "demo" / "template.html").read_text(encoding="utf-8")
    full = tpl.replace("/*__DATA__*/null", blob)
    # Три версии из одного шаблона:
    #   demo.html          — для показа, без пояснений;
    #   demo_explained.html — блоки «Простыми словами / Зачем» (между <!--EXPLAIN-->);
    #   demo_text.html     — то же, что demo_explained, но без подписей над абзацами.
    def strip(s: str, tag: str) -> str:
        return re.sub(rf"\s*<!--{tag}-->.*?<!--/{tag}-->", "", s, flags=re.S)

    def unwrap(s: str, tag: str) -> str:
        return s.replace(f"<!--{tag}-->", "").replace(f"<!--/{tag}-->", "")

    def retitle(s: str, suffix: str) -> str:
        return s.replace("<title>Буквы и авторы СМАЛТ</title>", f"<title>Буквы и авторы СМАЛТ · {suffix}</title>")

    explained = unwrap(full, "EXPLAIN")
    # demo_text.html — как demo_explained, но без подписей над абзацами, без плашки и со своими текстами
    text = explained.replace("<b>Простыми словами</b>", "").replace("<b>Зачем</b>", "")
    text = re.sub(r'\s*<div class="ex-banner">.*?</div>', "", text, flags=re.S)  # без плашки «Личная копия…»
    # свои, более простые формулировки пояснений — из demo/text_version.json (тема блока -> две колонки)
    alt = json.loads((ROOT / "demo" / "text_version.json").read_text(encoding="utf-8"))
    block = re.compile(r'(<aside class="explain"><div class="ex-topic">)(.*?)(</div><div class="ex-col"><p>)(.*?)'
                       r'(</p></div><div class="ex-col"><p>)(.*?)(</p></div></aside>)', re.S)
    used = set()

    def swap(m: re.Match) -> str:
        topic = m.group(2)
        if topic not in alt:
            raise SystemExit(f"demo/text_version.json: нет текста для блока «{topic}»")
        used.add(topic)
        left, right = alt[topic]
        return m.group(1) + topic + m.group(3) + left + m.group(5) + right + m.group(7)

    text = block.sub(swap, text)
    unused = set(alt) - used - {"_comment"}
    if unused:
        raise SystemExit(f"demo/text_version.json: темы не найдены на странице: {sorted(unused)}")
    outputs = {
        "demo.html": strip(full, "EXPLAIN"),
        "demo_explained.html": retitle(explained, "пояснения"),
        "demo_text.html": retitle(text, "текст"),
    }
    for name, page in outputs.items():
        (res / name).write_text(page, encoding="utf-8")
        print(f"results/{name}: {len(page.encode('utf-8')) / 1024:.0f} КБ")
    # Архив demo_text для отправки по почте: часть почтовых сервисов блокирует вложения .html.
    # Дата внутри архива фиксирована, чтобы пересборка без изменений давала тот же файл.
    with zipfile.ZipFile(res / "markov_smalt_demo.zip", "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("markov_smalt_demo.html", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, (res / "demo_text.html").read_bytes())  # ровно тот файл, что лежит в results/
    print("results/markov_smalt_demo.zip: demo_text.html для отправки")


if __name__ == "__main__":
    main()
