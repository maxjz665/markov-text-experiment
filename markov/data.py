"""Загрузка и проверка манифеста корпуса."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("text_id", "author", "title", "source_path", "source_reference", "attribution_status")


class CorpusError(Exception):
    """Ошибка данных, которую нужно показать пользователю."""


@dataclass
class TextRecord:
    text_id: str
    author: str
    author_key: str
    title: str
    path: Path
    meta: dict


def load_manifest(manifest_path: str | Path, base_dir: str | Path | None = None) -> list[TextRecord]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise CorpusError(f"Манифест не найден: {manifest_path}")
    base = Path(base_dir) if base_dir else manifest_path.resolve().parent.parent
    with open(manifest_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise CorpusError(f"В манифесте нет обязательных колонок: {missing}")
        rows = list(reader)
    if not rows:
        raise CorpusError("Манифест пуст")
    records, seen = [], set()
    for r in rows:
        tid = r["text_id"].strip()
        if tid in seen:
            raise CorpusError(f"Повторяющийся text_id={tid}")
        seen.add(tid)
        status = r["attribution_status"].strip()
        if not status or status.lower().startswith(("dubia", "не подтвержд", "unknown")):
            raise CorpusError(f"text_id={tid}: авторство не подтверждено ({status!r})")
        path = base / r["source_path"]
        if not path.exists():
            raise CorpusError(f"text_id={tid}: файл не найден {path}")
        records.append(TextRecord(
            text_id=tid, author=r["author"].strip(),
            author_key=(r.get("author_key") or r["author"]).strip(),
            title=r["title"].strip(), path=path, meta=r,
        ))
    return records


def read_text(path: Path) -> str:
    """Читает UTF-8 строго: повреждённая кодировка — явная ошибка."""
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        raise CorpusError(f"{path}: повреждённый UTF-8 ({e})") from None
