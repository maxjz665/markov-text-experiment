"""Формирование корпуса из извлечённых таблиц СМАЛТ.

Запуск (из каталога markov-text-experiment):
    python scripts/build_corpus.py --config config.json

Если data/private ещё нет, сначала выполняется scripts/extract_smalt.py.
Результат:
    data/raw/<text_id>.txt     — восстановленный текст (токены через пробел,
                                  абзацы с новой строки; пунктуации в СМАЛТ нет)
    data/manifest.csv          — отобранные тексты с происхождением
    data/excluded_texts.csv    — тексты выбранных авторов, не вошедшие в корпус, и причина
    data/selection_overview.csv — все тексты-эталоны СМАЛТ по авторам (для отчёта)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_int(v: str):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_tokens(path: Path, order: list[str]) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL))
    for r in rows:
        for k in order:
            if to_int(r[k]) is None:
                raise ValueError(f"{path.name}: пустой индекс {k} у ID_WORD={r['ID_WORD']}")
    rows.sort(key=lambda r: tuple(to_int(r[k]) for k in order))
    return rows


def reconstruct(rows: list[dict]) -> str:
    """Токены через пробел; новая строка при смене главы или абзаца."""
    lines, cur, prev = [], [], None
    for r in rows:
        key = (r["CHAPTER_INDEX"], r["PARAGRAPH_INDEX"])
        if prev is not None and key != prev:
            lines.append(" ".join(cur))
            cur = []
        cur.append(r["WORD"])
        prev = key
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines) + "\n"


def orthography_stats(text: str, tokens: list[str]) -> dict:
    s = unicodedata.normalize("NFC", text).lower()
    c = Counter(s)
    letters = sum(v for k, v in c.items() if k.isalpha())
    words = [t.lower() for t in tokens if any(ch.isalpha() for ch in t)]
    return {
        "letters_raw": letters,
        "yat_per_1000": round(1000 * c["ѣ"] / letters, 2) if letters else 0.0,
        "dotted_i_cyr_per_1000": round(1000 * c["і"] / letters, 2) if letters else 0.0,
        "dotted_i_latin_per_1000": round(1000 * c["i"] / letters, 2) if letters else 0.0,
        "fita_count": c["ѳ"] + c["ө"],
        "izhitsa_count": c["ѵ"],
        "final_hard_sign_share": round(sum(w.endswith("ъ") for w in words) / len(words), 3) if words else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    src = cfg["source"]
    priv = ROOT / src["private_dir"]
    raw_dir = ROOT / src["raw_dir"]

    if not (priv / "extract_info.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "extract_smalt.py"),
                        "--dump", str((ROOT / src["dump_path"]).resolve()), "--out", str(priv)],
                       check=True)
    info = json.loads((priv / "extract_info.json").read_text(encoding="utf-8"))
    if info["dump_sha256"] != src["dump_sha256"]:
        raise SystemExit(f"SHA-256 дампа {info['dump_sha256']} не совпадает с config.json")

    sel = cfg["selection"]
    authors = {a["author_id"]: a for a in cfg["authors"]}
    author_names = {to_int(r["ID"]): r for r in read_csv(priv / "author.csv")}
    author_types = {to_int(r["ID"]): r["NAME"] for r in read_csv(priv / "author_types.csv")}
    magazines = {to_int(r["ID"]): r["TITLE"] for r in read_csv(priv / "magazine.csv")}
    texts = read_csv(priv / "text.csv")
    excl_cfg = {int(k): v for k, v in cfg["exclude_texts"].items()}
    ocheck = cfg["orthography_check"]

    def is_reference(t: dict) -> bool:
        return all(to_int(t[f]) == sel[f] for f in ("AUTHOR_VERIFY", "STATUS", "CATEGORY", "INUSE1"))

    # Обзор всех эталонов (для отчёта о выборе авторов)
    overview: dict[int, dict] = {}
    for t in texts:
        if not is_reference(t):
            continue
        aid = to_int(t["AUTHOR_ID"])
        wf = priv / "words" / f"{t['ID']}.tsv"
        n_tok = sum(1 for _ in open(wf, encoding="utf-8")) - 1 if wf.exists() else 0
        o = overview.setdefault(aid, {"author_id": aid, "author_name": author_names.get(aid, {}).get("NAME", ""),
                                      "real_name": author_names.get(aid, {}).get("REAL_NAME", ""),
                                      "n_reference_texts": 0, "tokens": 0, "selected": aid in authors})
        o["n_reference_texts"] += 1
        o["tokens"] += n_tok

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest, excluded = [], []
    for t in texts:
        aid = to_int(t["AUTHOR_ID"])
        if aid not in authors:
            continue
        tid = to_int(t["ID"])
        base = {"text_id": tid, "author_key": authors[aid]["key"],
                "title": " ".join((t["TITLE"] or "").split())}
        if not is_reference(t):
            excluded.append({**base, "reason": "не проходит фильтр эталона: "
                             f"AUTHOR_VERIFY={t['AUTHOR_VERIFY']}, STATUS={t['STATUS']}, "
                             f"CATEGORY={t['CATEGORY']}, INUSE1={t['INUSE1']}"})
            continue
        if tid in excl_cfg:
            excluded.append({**base, "reason": excl_cfg[tid]})
            continue
        wf = priv / "words" / f"{tid}.tsv"
        if not wf.exists():
            excluded.append({**base, "reason": "нет токенов в таблице word"})
            continue
        rows = load_tokens(wf, sel["token_order"])
        text = reconstruct(rows)
        ost = orthography_stats(text, [r["WORD"] for r in rows])
        if ost["yat_per_1000"] < ocheck["min_yat_per_1000_letters"] and \
                ost["final_hard_sign_share"] >= ocheck["min_final_hard_sign_share"]:
            excluded.append({**base, "reason": f"несогласованная орфография: ъ на конце слов "
                             f"({ost['final_hard_sign_share']:.2f}), но ѣ={ost['yat_per_1000']}/1000 букв "
                             "(ѣ заменена на е при наборе)"})
            continue
        out_path = raw_dir / f"{tid}.txt"
        out_path.write_text(text, encoding="utf-8", newline="\n")
        at = to_int(t["AUTHOR_TYPE"])
        mid = to_int(t["MAGAZINE_ID"])
        manifest.append({
            "text_id": tid,
            "author": authors[aid]["label"],
            "author_key": authors[aid]["key"],
            "smalt_author_id": aid,
            "smalt_author_name": author_names[aid]["NAME"],
            "title": base["title"],
            "source_path": out_path.relative_to(ROOT).as_posix(),
            "source_reference": f"СМАЛТ, дамп {info['dump_file']} (SHA-256 {info['dump_sha256'][:12]}…), text.ID={tid}",
            "attribution_status": "Эталон (AUTHOR_VERIFY=1); статья проверена (STATUS=2)",
            "signed": {0: "анонимная", 1: "подписанная"}.get(to_int(t["TEXT_TYPE"]), "не указано"),
            "signature_type": author_types.get(at, "") if at else "",
            "genre": "",
            "year": (t["PUBLICATION_DATE"] or "")[:4],
            "publication_date": t["PUBLICATION_DATE"] or "",
            "magazine": magazines.get(mid, "") if mid else "",
            "magazine_no": t["MAGAZINE_NO"] or "",
            "orthography": "дореформенная (в исходном наборе)",
            "edition": "",
            "tokens": len(rows),
            **ost,
            "license_or_access_note": "Локальная копия дампа СМАЛТ (ПетрГУ); только для исследования, не публиковать",
            "note": "",
        })

    # Пометки, найденные при ручной проверке пересечений (см. README)
    notes = {299: "содержит цитату ≈229 слов из text.ID=298 того же автора (≈12 % 5-словных шинглов)",
             298: "фрагмент ≈229 слов процитирован в text.ID=299"}
    for m in manifest:
        m["note"] = notes.get(m["text_id"], "")

    manifest.sort(key=lambda m: (m["author_key"], m["publication_date"], m["text_id"]))
    data_dir = ROOT / "data"
    for name, rows in (("manifest.csv", manifest), ("excluded_texts.csv", excluded),
                       ("selection_overview.csv", sorted(overview.values(), key=lambda o: -o["tokens"]))):
        with open(data_dir / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    by_author = Counter(m["author_key"] for m in manifest)
    print(f"В корпус вошло {len(manifest)} текстов: {dict(by_author)}")
    print(f"Исключено текстов выбранных авторов: {len(excluded)} (см. data/excluded_texts.csv)")


if __name__ == "__main__":
    main()
