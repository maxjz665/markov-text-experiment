"""Предобработка: нормализация, разбиение на сегменты, классы V/C.

Правила одинаковы для всех авторов и задаются конфигурацией:
  * Unicode NFC, нижний регистр, удаление знаков ударения (U+0300/U+0301);
  * исправление латинских «двойников» кириллицы (i→і, o→о, …) только если
    после замены токен становится полностью кириллическим;
  * cross_word=false: переходы считаются только внутри токена; любой символ
    вне алфавита (пробел, цифра, дефис, латиница) разрывает цепочку;
  * для V/C ъ и ь не относятся ни к одному классу и тоже разрывают цепочку.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

_CYR = re.compile(r"[Ѐ-ӿ]")
_LATIN_LETTER = re.compile(r"[a-zÀ-ɏ]")


@dataclass
class PreprocessConfig:
    alphabet: str
    vowels: str
    consonants: str
    vc_breakers: str
    cross_word: bool = False
    keep_yo: bool = True
    strip_stress_marks: bool = True
    stress_marks: tuple[str, ...] = ("̀", "́")
    homoglyph_repair: bool = True
    homoglyphs: dict[str, str] = field(default_factory=dict)
    mode: str = "original"

    def __post_init__(self) -> None:
        if self.cross_word:
            raise NotImplementedError("cross_word=true не реализован в этом этапе")
        if self.mode != "original":
            raise NotImplementedError(f"Режим {self.mode!r} не реализован; используется 'original'")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("В алфавите есть повторяющиеся символы")
        classes = set(self.vowels) | set(self.consonants) | set(self.vc_breakers)
        if set(self.vowels) & set(self.consonants):
            raise ValueError("Буква отнесена и к гласным, и к согласным")
        if classes != set(self.alphabet):
            raise ValueError("Гласные ∪ согласные ∪ разделители V/C должны совпадать с алфавитом: "
                             f"лишние {sorted(classes - set(self.alphabet))}, "
                             f"без класса {sorted(set(self.alphabet) - classes)}")
        if not self.keep_yo and "ё" in self.alphabet:
            raise ValueError("keep_yo=false несовместим с 'ё' в алфавите")

    @classmethod
    def from_dict(cls, cfg: dict) -> "PreprocessConfig":
        p = cfg.get("preprocess", {})
        return cls(
            alphabet=cfg["alphabet"], vowels=cfg["vowels"], consonants=cfg["consonants"],
            vc_breakers=cfg["vc_breakers"],
            cross_word=p.get("cross_word", False), keep_yo=p.get("keep_yo", True),
            strip_stress_marks=p.get("strip_stress_marks", True),
            stress_marks=tuple(p.get("stress_marks", ("̀", "́"))),
            homoglyph_repair=p.get("homoglyph_repair", True),
            homoglyphs=dict(p.get("homoglyphs", {})), mode=p.get("mode", "original"),
        )


@dataclass
class PreprocessLog:
    """Счётчики, которые попадают в отчёт (по автору или по тексту)."""
    tokens: int = 0
    tokens_homoglyph_repaired: int = 0
    letters: int = 0                       # символы алфавита
    dropped_letters: Counter = field(default_factory=Counter)   # буквы вне алфавита
    dropped_other: Counter = field(default_factory=Counter)     # цифры, знаки и пр.
    stress_marks_removed: int = 0

    def merge(self, other: "PreprocessLog") -> None:
        self.tokens += other.tokens
        self.tokens_homoglyph_repaired += other.tokens_homoglyph_repaired
        self.letters += other.letters
        self.dropped_letters.update(other.dropped_letters)
        self.dropped_other.update(other.dropped_other)
        self.stress_marks_removed += other.stress_marks_removed


def normalize_text(text: str, cfg: PreprocessConfig, log: PreprocessLog | None = None) -> str:
    """NFC, нижний регистр, удаление знаков ударения."""
    s = unicodedata.normalize("NFC", text).lower()
    if cfg.strip_stress_marks:
        d = unicodedata.normalize("NFD", s)
        marks = set(cfg.stress_marks)
        if log is not None:
            log.stress_marks_removed += sum(ch in marks for ch in d)
        s = unicodedata.normalize("NFC", "".join(ch for ch in d if ch not in marks))
    return s


def repair_homoglyphs(token: str, cfg: PreprocessConfig) -> str:
    """Заменяет латинские двойники кириллицей, если токен станет чисто кириллическим."""
    if not cfg.homoglyph_repair or not _CYR.search(token):
        return token
    if not any(ch in cfg.homoglyphs for ch in token):
        return token
    fixed = "".join(cfg.homoglyphs.get(ch, ch) for ch in token)
    return token if _LATIN_LETTER.search(fixed) else fixed


def letter_segments(text: str, cfg: PreprocessConfig, log: PreprocessLog | None = None) -> list[str]:
    """Максимальные последовательности букв алфавита внутри токенов."""
    alphabet = set(cfg.alphabet)
    s = normalize_text(text, cfg, log)
    segments: list[str] = []
    for token in s.split():
        if log is not None:
            log.tokens += 1
        fixed = repair_homoglyphs(token, cfg)
        if log is not None and fixed != token:
            log.tokens_homoglyph_repaired += 1
        cur: list[str] = []
        for ch in fixed:
            if ch in alphabet:
                cur.append(ch)
                continue
            if log is not None:
                (log.dropped_letters if ch.isalpha() else log.dropped_other)[ch] += 1
            if cur:
                segments.append("".join(cur))
                cur = []
        if cur:
            segments.append("".join(cur))
    if log is not None:
        log.letters += sum(len(x) for x in segments)
    return segments


def vc_segments(segments: list[str], cfg: PreprocessConfig) -> list[str]:
    """Переводит буквенные сегменты в строки из 'V'/'C'; ъ/ь разрывают цепочку."""
    vowels, consonants = set(cfg.vowels), set(cfg.consonants)
    out: list[str] = []
    for seg in segments:
        cur: list[str] = []
        for ch in seg:
            if ch in vowels:
                cur.append("V")
            elif ch in consonants:
                cur.append("C")
            else:  # разделитель (ъ, ь)
                if cur:
                    out.append("".join(cur))
                    cur = []
        if cur:
            out.append("".join(cur))
    return out
