"""Безопасное потоковое чтение mysqldump СМАЛТ без выполнения SQL.

Дамп содержит DROP TABLE и служебные таблицы (sys_users, django_session,
auth_*). Здесь SQL никуда не импортируется: читаются только строки
``INSERT INTO `<table>` ... VALUES (...),(...);`` для таблиц из белого
списка, всё остальное пропускается без разбора.
"""
from __future__ import annotations

import gzip
import re
from typing import Iterator

# Литературные таблицы, которые разрешено читать.
ALLOWED_TABLES = frozenset({
    "author", "author_types", "text", "magazine",
    "textlist_description", "textlist_items", "word",
})

_INSERT_RE = re.compile(r"^INSERT INTO `([^`]+)` \(([^)]*)\) VALUES ")
_TOKEN_RE = re.compile(
    r"'((?:[^'\\]|\\.)*)'"          # 1: строка в одинарных кавычках
    r"|(NULL)"                      # 2: NULL
    r"|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"  # 3: число
    r"|([(),;])",                   # 4: разделители
    re.S,
)
_ESCAPES = {
    "0": "\0", "'": "'", '"': '"', "b": "\b", "n": "\n", "r": "\r",
    "t": "\t", "Z": "\x1a", "\\": "\\", "%": "\\%", "_": "\\_",
}
_ESC_RE = re.compile(r"\\(.)", re.S)


def unescape_mysql(s: str) -> str:
    """Снимает экранирование строкового литерала mysqldump."""
    if "\\" not in s:
        return s
    return _ESC_RE.sub(lambda m: _ESCAPES.get(m.group(1), m.group(1)), s)


def parse_values(values_sql: str) -> Iterator[list]:
    """Разбирает часть после VALUES: ``(1,'a',NULL),(2,'b',3);``."""
    row: list | None = None
    for m in _TOKEN_RE.finditer(values_sql):
        s, null, num, punct = m.groups()
        if punct is not None:
            if punct == "(":
                if row is not None:
                    raise ValueError("Вложенная скобка в VALUES")
                row = []
            elif punct == ")":
                if row is None:
                    raise ValueError("Лишняя закрывающая скобка в VALUES")
                yield row
                row = None
            continue
        if row is None:
            raise ValueError(f"Значение вне кортежа: {m.group(0)[:40]!r}")
        if s is not None:
            row.append(unescape_mysql(s))
        elif null is not None:
            row.append(None)
        else:
            row.append(float(num) if any(c in num for c in ".eE") else int(num))
    if row is not None:
        raise ValueError("Незакрытый кортеж в VALUES")


def iter_table_rows(dump_path: str, tables: set[str] | frozenset[str]) -> Iterator[tuple[str, dict]]:
    """Выдаёт (имя_таблицы, строка-словарь) для таблиц из ``tables``.

    Разрешены только таблицы из ALLOWED_TABLES; запрос служебных таблиц
    считается ошибкой.
    """
    forbidden = set(tables) - ALLOWED_TABLES
    if forbidden:
        raise PermissionError(f"Таблицы не входят в белый список: {sorted(forbidden)}")
    opener = gzip.open if str(dump_path).endswith(".gz") else open
    with opener(dump_path, "rt", encoding="utf-8", errors="strict", newline="\n") as fh:
        for line in fh:
            if not line.startswith("INSERT INTO `"):
                continue
            m = _INSERT_RE.match(line)
            if not m or m.group(1) not in tables:
                continue
            table = m.group(1)
            cols = [c.strip().strip("`") for c in m.group(2).split(",")]
            for values in parse_values(line[m.end():]):
                if len(values) != len(cols):
                    raise ValueError(f"{table}: {len(values)} значений при {len(cols)} колонках")
                yield table, dict(zip(cols, values))
