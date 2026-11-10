"""Однократное извлечение литературных таблиц из дампа СМАЛТ.

Запуск (из каталога markov-text-experiment):
    python scripts/extract_smalt.py --dump ../04_Данные_СМАЛТ/smalt.sql.20260115.070144.gz

Результат пишется в data/private/ (не публиковать):
    author.csv, author_types.csv, text.csv, magazine.csv,
    textlist_description.csv, textlist_items.csv,
    words/<TEXT_ID>.tsv  — токены текста со всеми индексами,
    extract_info.json    — контрольная сумма дампа и счётчики строк.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from markov.smalt_dump import iter_table_rows  # noqa: E402

META_TABLES = ["author", "author_types", "text", "magazine",
               "textlist_description", "textlist_items"]
WORD_COLS = ["ID_WORD", "TEXT_ID", "CHAPTER_INDEX", "PARAGRAPH_INDEX",
             "SENTENCE_INDEX", "WORD_INDEX", "WORDORDER", "WORDNO",
             "WORD_LENGTH", "DICTWORD_ID", "WORD"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default="data/private")
    args = ap.parse_args()

    dump = Path(args.dump)
    out = Path(args.out)
    (out / "words").mkdir(parents=True, exist_ok=True)

    meta_rows: dict[str, list[dict]] = {t: [] for t in META_TABLES}
    word_files: dict[int, tuple] = {}
    counts: Counter = Counter()
    try:
        for table, row in iter_table_rows(str(dump), set(META_TABLES) | {"word"}):
            counts[table] += 1
            if table != "word":
                meta_rows[table].append(row)
                continue
            tid = row["TEXT_ID"]
            if tid not in word_files:
                fh = open(out / "words" / f"{tid}.tsv", "w", encoding="utf-8", newline="")
                w = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                w.writerow(WORD_COLS)
                word_files[tid] = (fh, w)
            word_files[tid][1].writerow(["" if row[c] is None else row[c] for c in WORD_COLS])
    finally:
        for fh, _ in word_files.values():
            fh.close()

    for table, rows in meta_rows.items():
        if not rows:
            continue
        with open(out / f"{table}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    info = {
        "dump_file": dump.name,
        "dump_sha256": sha256(dump),
        "rows": dict(counts),
        "texts_with_words": len(word_files),
    }
    (out / "extract_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
