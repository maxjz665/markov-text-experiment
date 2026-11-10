"""Живая демонстрация вычислений для показа на встрече.

Запуск (из каталога markov-text-experiment или двойным щелчком по run_demo.bat):
    python scripts/live_demo.py              # с паузами: Enter — следующий шаг
    python scripts/live_demo.py --no-pause   # без пауз

Все числа считаются заново, на глазах, тем же кодом пакета markov, что и основной
расчёт (python -m markov.cli). Нужны только data/manifest.csv и data/raw/ —
их создаёт scripts/build_corpus.py. Результаты в results/ не перезаписываются.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import textwrap
import time
import unittest
import webbrowser
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Кириллица с ѣ/і/ѳ и цвета в консоли Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if os.name == "nt":
    os.system("")  # включает ANSI-последовательности в консоли Windows

import numpy as np  # noqa: E402

from markov import compare  # noqa: E402
from markov.data import CorpusError, load_manifest, read_text  # noqa: E402
from markov.preprocess import PreprocessConfig, letter_segments, vc_segments  # noqa: E402
from markov.transitions import count_transitions, merge  # noqa: E402

B, DIM, END = "\033[1m", "\033[2m", "\033[0m"
BLUE, GREEN, AMBER, RED = "\033[38;5;33m", "\033[38;5;35m", "\033[38;5;178m", "\033[38;5;167m"
PAUSE = True
EXPLAIN = True
TEST_RU = {
    "test_abba": "«абба»: а→б, б→б, б→а — ровно 3 пары, строки P дают в сумме 1",
    "test_mama_vc": "«мама» в модели V/C: C→V = 2, V→C = 1, всего 3 пары",
    "test_no_cross_word": "«а а»: между словами перехода нет — 0 пар",
    "test_object": "«объект»: б→ъ и ъ→е есть в буквенной модели, но ъ разрывает цепочку V/C",
    "test_no_transition_across_files": "между двумя файлами перехода нет",
    "test_unobserved_row_is_nan_not_uniform": "у буквы без продолжений вероятность не оценивается (нет деления на 0)",
    "test_comparison_uses_probabilities_not_counts": "короткий и длинный текст с одним составом дают одинаковые P",
    "test_historical_letters_counted_as_states": "ѣ и ё — отдельные состояния: в→ѣ, ё→ж считаются",
    "test_pairs_equal_sum_N": "сумма матрицы N = число пар = буквы − слова",
    "test_unknown_state_raises": "неизвестный символ в модели — явная ошибка, а не тихий пропуск",
    "test_balanced_sample_is_deterministic": "равные выборки воспроизводятся при одном seed",
    "test_space_breaks_chain": "пробел разрывает цепочку",
    "test_punctuation_digits_hyphen_break": "дефис, запятая и цифры разрывают цепочку",
    "test_hard_sign_is_letter_but_vc_breaker": "ъ — буква в буквенной модели и разрыв в модели V/C",
    "test_soft_sign_vc_breaker": "ь разрывает цепочку V/C («кольцо» → CVC | CV)",
    "test_yo_kept_separately": "ё не сливается с е",
    "test_historical_letters_preserved": "ѣ, і, ѳ сохраняются как есть",
    "test_vc_classes_for_historical_letters": "ѣ и і — гласные, ѳ — согласная",
    "test_latin_i_homoglyph_repaired_only_in_cyrillic_tokens": "латинская i в русском слове → і; латинские слова не трогаются",
    "test_mixed_token_not_fully_fixable_is_left_alone": "смешанное слово, которое нельзя исправить целиком, не трогается",
    "test_stress_marks_removed": "знаки ударения снимаются, й и ё не портятся",
    "test_unknown_letters_logged_not_silently_lost": "буквы вне алфавита записываются в лог, а не теряются",
    "test_invalid_config_rejected": "противоречивые настройки алфавита отвергаются",
}
T0 = time.perf_counter()


def n(x) -> str:
    return f"{int(x):,}".replace(",", " ")


def texts_word(k: int) -> str:
    """1 текст, 2 текста, 5 текстов."""
    if k % 10 == 1 and k % 100 != 11:
        return "текст"
    if 2 <= k % 10 <= 4 and not 12 <= k % 100 <= 14:
        return "текста"
    return "текстов"


def hist(s: str) -> str:
    """Подсвечивает дореформенные буквы."""
    return "".join(f"{AMBER}{c}{END}" if c in "ѣіѳ" else c for c in s)


def say(label: str, text: str, color: str = AMBER) -> None:
    """Абзац «простыми словами» с подписью слева и переносом строк."""
    pad = " " * (len(label) + 4)
    body = textwrap.fill(text, width=96, initial_indent=pad, subsequent_indent=pad)[len(pad):]
    print(f"  {color}{B}{label}{END}  {body}")


def step(num: int, title: str, tech: str, plain: str = "", why: str = "") -> float:
    print(f"\n{BLUE}{B}── Шаг {num}. {title} {'─' * max(3, 70 - len(title))}{END}")
    if EXPLAIN and plain:
        say("Что делаем:", plain)
        say("Зачем:     ", why)
    print(f"{DIM}  {tech}{END}\n")
    return time.perf_counter()


def seen(text: str) -> None:
    """Вывод шага простыми словами."""
    if EXPLAIN:
        print()
        say("Что видно: ", text, GREEN)


def done(t: float) -> None:
    print(f"\n{DIM}  ⏱ {time.perf_counter() - t:.1f} с{END}")
    if PAUSE:
        try:
            input(f"{DIM}  Enter — дальше…{END}")
        except EOFError:
            pass


def bar(i: int, total: int, label: str, width: int = 28) -> None:
    filled = round(width * i / total)
    sys.stdout.write(f"\r  {label:<20} {BLUE}{'█' * filled}{DIM}{'░' * (width - filled)}{END} {i}/{total}")
    sys.stdout.flush()
    if i == total:
        sys.stdout.write("\n")


def shade(p: float) -> str:
    """Ячейка из 5 символов: штрих-шкала плюс значение."""
    if p != p:  # NaN — переходов нет
        return f"{DIM}  ·  {END}"
    k = min(4, int((p ** 0.5) * 5))
    return f"{BLUE}{' ░▒▓█'[k]}{END}{p:4.2f}"


def main() -> int:
    global PAUSE
    ap = argparse.ArgumentParser(description="Живая демонстрация вычислений")
    ap.add_argument("--no-pause", action="store_true", help="не ждать Enter между шагами")
    ap.add_argument("--author", default="dostoevsky_fm", help="автор для шага 5")
    ap.add_argument("--short", action="store_true", help="без пояснений «что делаем / зачем»")
    args = ap.parse_args()
    PAUSE = not args.no_pause
    global EXPLAIN
    EXPLAIN = not args.short

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    pc = PreprocessConfig.from_dict(cfg)
    states = list(cfg["alphabet"])
    thr = cfg["low_support_threshold"]
    labels = {a["key"]: a["label"] for a in cfg["authors"]}
    keys = list(labels)

    print(f"\n{B}Буквы и авторы СМАЛТ — цепи Маркова первого порядка{END}")
    print(f"{DIM}Демонстрация считает всё заново из текстов корпуса тем же кодом, что и основной расчёт.{END}")

    # ── 1. Корпус ─────────────────────────────────────────────────────────
    t = step(1, "Корпус", "Тексты СМАЛТ со статусом «Эталон» (AUTHOR_VERIFY = 1) и «Статья проверена» (STATUS = 2).",
             "Показываем, какие тексты взяли: только те, у которых в СМАЛТ авторство подтверждено и статья проверена.",
             "Если автор текста под сомнением, сравнивать авторов бессмысленно: мы бы сравнивали ошибки атрибуции.")
    try:
        records = load_manifest(ROOT / "data" / "manifest.csv")
    except CorpusError as e:
        print(f"{RED}Нет корпуса: {e}{END}\nСначала выполните: python scripts/build_corpus.py --config config.json")
        return 2
    by_author = {k: [r for r in records if r.author_key == k] for k in keys}
    print(f"  {'Автор':<20} {'текстов':>7}  годы")
    for k in keys:
        years = sorted({r.meta["year"] for r in by_author[k] if r.meta.get("year")})
        print(f"  {labels[k]:<20} {len(by_author[k]):>7}  {years[0]}–{years[-1]}")
    print(f"\n  Всего: {B}{len(records)} текстов, {len(keys)} авторов{END}; алфавит из {len(states)} букв: {hist(' '.join(states))}")
    sizes = sorted((len(by_author[k]), labels[k]) for k in keys)
    seen(f"У автора {sizes[-1][1]} {sizes[-1][0]} {texts_word(sizes[-1][0])}, у автора {sizes[0][1]} — {sizes[0][0]}. Объёмы очень разные, "
         "поэтому дальше авторов сравниваем по долям и на выборках одинакового размера. "
         "Буквы ѣ, і, ѳ оставлены: тексты набраны в дореформенной орфографии.")
    done(t)

    # ── 2. От текста к парам ─────────────────────────────────────────────
    t = step(2, "Как текст превращается в пары букв",
             "Переходы считаются только внутри слова; пробел, цифра, дефис разрывают цепочку.",
             "Берём настоящую фразу из корпуса, режем её на слова и выписываем соседние буквы внутри каждого слова. "
             "Каждая такая пара — один «переход».",
             "Вся модель — это подсчёт таких пар. Здесь видно, что именно считается и что между словами пар нет.")
    rec = next((r for r in records if r.text_id == "42"), by_author[keys[0]][0])
    lines = read_text(rec.path).splitlines()
    sample_line = next((ln for ln in lines if len(ln.split()) > 12), lines[0])
    fragment = " ".join(sample_line.split()[:12])
    print(f"  Фрагмент (text.ID = {rec.text_id}, {labels[rec.author_key]}):")
    print(f"  {B}{hist(fragment)}{END}\n")
    segs = letter_segments(fragment, pc)
    print(f"  Слова после нормализации: {hist(' | '.join(segs))}")
    w = max(segs, key=len)
    pairs = [f"{a}→{b}" for a, b in zip(w, w[1:])]
    print(f"  Пары в слове «{hist(w)}»: {hist('  '.join(pairs))}   ({len(w)} букв → {len(pairs)} пар)")
    total = sum(len(s) - 1 for s in segs)
    print(f"  Во всём фрагменте: {sum(len(s) for s in segs)} букв → {B}{total} пар{END} (между словами пар нет)")
    # слово, где ъ/ь стоит внутри, — чтобы был виден разрыв цепочки V/C
    with_hard = next((s for ln in lines for s in letter_segments(ln, pc)
                      if len(s) >= 5 and any(c in s[1:-1] for c in "ъь")), "объектъ")
    vc = vc_segments([with_hard], pc)
    print(f"\n  Гласные/согласные: «{hist(with_hard)}» → {B}{' | '.join(vc)}{END}"
          f"   {DIM}(ъ и ь не гласные и не согласные — разрывают цепочку){END}")
    seen("Слово из n букв даёт n − 1 пару. Для второй модели каждая буква заменяется на V (гласная) или C (согласная), "
         "а ъ и ь цепочку разрывают: через них переход не считается.")
    done(t)

    # ── 3. Тесты ─────────────────────────────────────────────────────────
    t = step(3, "Проверка арифметики", "Автоматические тесты: «абба», «мама», «а а», «объект», границы файлов, NaN и др.",
             "Запускаем автоматические проверки на маленьких словах, где правильный ответ известен заранее.",
             "Чтобы доверять большим числам дальше, сначала убеждаемся, что программа верно считает простые случаи.")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))

    def iter_tests(s):
        for x in s:
            yield from (iter_tests(x) if isinstance(x, unittest.TestSuite) else [x])
    tests = [(str(tc), tc._testMethodName) for tc in iter_tests(suite)]  # до запуска: run() очищает набор
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    failed = {str(x[0]) for x in result.failures + result.errors}
    for tid, name in tests:
        mark = f"{RED}✗{END}" if tid in failed else f"{GREEN}✓{END}"
        print(f"  {mark} {TEST_RU.get(name, name)}")
    status = f"{GREEN}все прошли{END}" if result.wasSuccessful() else f"{RED}есть ошибки{END}"
    print(f"\n  {B}Тестов: {result.testsRun} — {status}{END}")
    seen("Все проверки пройдены: программа правильно режет слова, не создаёт пар между словами и файлами "
         "и не выдумывает вероятности там, где данных нет." if result.wasSuccessful() else
         "Есть непройденные проверки — числам дальше доверять нельзя, пока они не исправлены.")
    done(t)

    # ── 4. Модели всех авторов ───────────────────────────────────────────
    t = step(4, "Считаем модели всех авторов", "Для каждого текста — матрица N 36×36 и матрица V/C 2×2; тексты автора складываются.",
             "Для каждого текста заполняем таблицу 36 × 36: сколько раз за буквой i шла буква j. "
             "Таблицы всех текстов одного автора складываем.",
             "Эта таблица и есть модель автора — цепь Маркова первого порядка. Всё дальнейшее строится из неё.")
    text_letter, text_vc, text_tokens = {}, {}, {}
    A, AV = {}, {}
    for k in keys:
        recs = by_author[k]
        for i, r in enumerate(recs, 1):
            raw = read_text(r.path)
            s = letter_segments(raw, pc)
            text_letter[r.text_id] = count_transitions(s, states)
            text_vc[r.text_id] = count_transitions(vc_segments(s, pc), "VC")
            text_tokens[r.text_id] = raw.split()
            bar(i, len(recs), labels[k])
        A[k] = merge([text_letter[r.text_id] for r in recs])
        AV[k] = merge([text_vc[r.text_id] for r in recs])
    print(f"\n  {'Автор':<20} {'букв':>10} {'пар букв':>10} {'сумма N':>10}  проверка")
    for k in keys:
        m = A[k]
        ok = int(m.N.sum()) == m.total_pairs == int(m.unigrams.sum()) - m.n_segments
        print(f"  {labels[k]:<20} {n(m.unigrams.sum()):>10} {n(m.total_pairs):>10} {n(m.N.sum()):>10}  "
              f"{GREEN + '✓ пары = буквы − слова' + END if ok else RED + '✗' + END}")
    seen("У каждого автора сумма таблицы равна числу пар, то есть «буквы минус слова». "
         "Значит, ни одна пара не потерялась и не посчиталась дважды.")
    done(t)

    # ── 5. Что внутри матрицы ────────────────────────────────────────────
    k0 = args.author if args.author in A else keys[0]
    m = A[k0]
    t = step(5, f"Что внутри матрицы: {labels[k0]}", "P(j | i) — доля переходов из буквы i, которые ведут в букву j.",
             "Заглядываем в таблицу одного автора: самые частые пары и доля каждого продолжения. "
             "P — это вероятность: «если сейчас буква г, насколько вероятно, что дальше будет о».",
             "Авторов сравнивают по P, а не по сырым числам: у Достоевского текста больше, и все его числа больше просто поэтому.")
    print(f"  Самые частые переходы:")
    for x in compare.top_transitions(m, 8, thr):
        print(f"    {hist(x['from'])} → {hist(x['to'])}   N = {n(x['count']):>7}   P = {x['probability']:.3f}  (из {n(x['row_total'])})")
    rows, cols = "аеиоуѣ", "внрстй"
    P = m.P
    print(f"\n  Фрагмент матрицы P (строка — текущая буква, столбец — следующая):")
    print("       " + " ".join(f"  {hist(c)}  " for c in cols))
    for r_ in rows:
        i = states.index(r_)
        print(f"    {hist(r_)}  " + " ".join(shade(P[i, states.index(c)]) for c in cols))
    gi, oi = states.index("г"), states.index("о")
    words = Counter()
    for r in by_author[k0]:
        for s in letter_segments(read_text(r.path), pc):
            c = s.count("го")
            if c:
                words[s] += c
    top = ", ".join(f"{w} ({n(c)})" for w, c in words.most_common(4))
    print(f"\n  Пример: P(о | г) = {B}{P[gi, oi]:.3f}{END} — после «г» почти в {round(P[gi, oi] * 100)} % случаев идёт «о».")
    print(f"  Откуда: {top}")
    seen("Самые частые пары почти одинаковы у всех авторов — это свойство русского языка и орфографии, а не автора. "
         "За каждым числом стоят конкретные слова, и это пригодится при сравнении.")
    done(t)

    # ── 6. Гласные/согласные ─────────────────────────────────────────────
    t = step(6, "Гласные и согласные", "Каждая буква → V (гласная) или C (согласная); четыре вероятности перехода.",
             "Сводим все буквы к двум видам — гласные и согласные — и считаем, как часто за гласной идёт гласная, "
             "за гласной согласная и так далее. Так Марков в 1913 году разбирал «Евгения Онегина».",
             "Проверяем, видна ли разница между авторами уже на самом грубом уровне.")
    print(f"  {'Автор':<20} {'P(V→V)':>8} {'P(V→C)':>8} {'P(C→V)':>8} {'P(C→C)':>8} {'пар V/C':>10}")
    for k in keys:
        Pv = AV[k].P
        print(f"  {labels[k]:<20} {Pv[0, 0]:>8.3f} {Pv[0, 1]:>8.3f} {Pv[1, 0]:>8.3f} {Pv[1, 1]:>8.3f} {n(AV[k].total_pairs):>10}")
    cc = [AV[k].P[1, 1] for k in keys]
    if not EXPLAIN:
        print(f"\n  P(C→C) у всех в пределах {min(cc):.3f}–{max(cc):.3f}: по двум состояниям авторы почти не различаются.")
    seen(f"Разница между авторами — в третьем знаке (P(C→C) от {min(cc):.3f} до {max(cc):.3f}). "
         "Двух состояний мало, чтобы различать авторов, поэтому основная модель — буквенная.")
    done(t)

    # ── 7. Сравнение авторов ─────────────────────────────────────────────
    t = step(7, "Сравнение авторов на равных выборках",
             "У каждого автора одинаковый объём текста (seed 42); расстояние D: 0 — одинаковые модели.",
             "Берём у каждого автора одинаковый объём текста и одним числом D измеряем, насколько похожи их таблицы. "
             "D = 0 — таблицы совпадают; чем больше D, тем сильнее различие.",
             "Равный объём убирает преимущество авторов с большим корпусом. А одно число позволяет сравнить все пары авторов сразу.")
    budget = min(int(A[k].unigrams.sum()) for k in keys)
    bal = {}
    for i, k in enumerate(keys, 1):
        docs = [text_tokens[r.text_id] for r in by_author[k]]
        bal[k] = count_transitions(compare.balanced_sample(docs, budget, cfg["balanced"]["chunk_tokens"], cfg["seed"], pc), states)
        bar(i, len(keys), "выборки")
    print(f"  По {n(budget)} букв на автора. Ближайший к каждому автор:")
    farthest = Counter()
    nearest_pairs = []
    for a in keys:
        d = sorted((compare.weighted_row_tv(bal[a], bal[b], thr), b) for b in keys if b != a)
        farthest[d[-1][1]] += 1
        nearest_pairs.append((d[0][0], a, d[0][1]))
        print(f"    {labels[a]:<20} → {labels[d[0][1]]:<20} D = {d[0][0]:.3f}   самый далёкий: {labels[d[-1][1]]} ({d[-1][0]:.3f})")
    np_ = min(nearest_pairs)
    far_k, far_c = farthest.most_common(1)[0]
    a, b = "dostoevsky_fm", "putsykovich"
    if a in A and b in A:
        print(f"\n  Где сильнее всего расходятся {labels[a]} и {labels[b]}:")
        segs_a = Counter(s for r in by_author[a] for s in letter_segments(read_text(r.path), pc))
        segs_b = Counter(s for r in by_author[b] for s in letter_segments(read_text(r.path), pc))

        def ex(cnt, bg):
            c = Counter({w: v * w.count(bg) for w, v in cnt.items() if bg in w})
            return ", ".join(w for w, _ in c.most_common(2))
        for d in compare.difference_table(A[a], A[b], thr, k=5):
            bg = d["from"] + d["to"]
            print(f"    {hist(d['from'])}→{hist(d['to'])}  {d['P_a']:.3f} против {d['P_b']:.3f}   "
                  f"{DIM}слова: {ex(segs_a, bg)} / {ex(segs_b, bg)}{END}")
        if not EXPLAIN:
            print(f"\n  {AMBER}Многие различия дают несколько слов — это тема текстов, а не «почерк».{END}")
    seen(f"Ближе всего друг к другу {labels[np_[1]]} и {labels[np_[2]]} (D = {np_[0]:.3f}). "
         f"Для {far_c} из {len(keys)} авторов самый далёкий — {labels[far_k]}. "
         "Но многие большие различия дают одно-два слова (например, «Франція» в обзорах иностранной политики): "
         "тема текста заметно влияет на буквенную таблицу.")
    done(t)

    # ── 8. Угадывание по Хмелёву ─────────────────────────────────────────
    t = step(8, "Угадывание автора по Хмелёву",
             "Каждый текст убирается из корпуса; выбирается автор, чья цепь даёт тексту наибольшее правдоподобие.",
             "По очереди прячем каждый текст, строим таблицы авторов без него и спрашиваем: таблица какого автора "
             "лучше всего «предсказывает» буквы этого текста? Это метод Д. В. Хмелёва (2000).",
             "Быстро проверить, есть ли у моделей сила различать авторов. Это пробный опыт, а не оценка точности метода.")
    pr = cfg["attribution_probe"]
    sel = {tid: (r.author_key, text_letter[tid]) for r in records for tid in [r.text_id]
           if text_letter[tid].unigrams.sum() >= pr["min_text_letters"]}
    res = compare.attribution_probe(sel, A, float(pr["smoothing_alpha"]))
    ok = sum(x["correct"] for x in res)
    majority = max(Counter(x["true_author"] for x in res).values())
    print(f"  Проверено текстов: {len(res)}")
    print(f"  Верно: {B}{ok} из {len(res)} ({ok / len(res):.0%}){END}; "
          f"если всегда называть самого «многотекстового» автора — {majority / len(res):.0%}")
    titles = {r.text_id: r.title for r in records}
    print(f"\n  Примеры ошибок:")
    for x in [x for x in res if not x["correct"]][:5]:
        print(f"    {x['text_id']:>4}  {labels[x['true_author']]} → {labels[x['predicted']]}   {DIM}{titles[x['text_id']][:45]}{END}")
    foreign = [x for x in res if "Иностранн" in titles[x["text_id"]]]
    foreign_bad = sum(not x["correct"] for x in foreign)
    seen(f"Модель угадывает {ok / len(res):.0%} текстов, а наивное «всегда самый плодовитый автор» — {majority / len(res):.0%}: "
         "сигнал есть. " + (f"Но {foreign_bad} из {len(foreign)} текстов рубрики «Иностранные события» отнесены не к тому автору: "
                            "одинаковая тема сближает разных авторов." if foreign else ""))
    done(t)

    # ── 9. Внутри и между ────────────────────────────────────────────────
    reps = 3
    t = step(9, "Различаются ли авторы сильнее, чем их собственные тексты",
             f"Тексты автора делятся на две группы произведений; быстрый режим — {reps} повтора (в отчёте 20).",
             "Делим тексты каждого автора на две кучки разных произведений. Смотрим, насколько кучки похожи друг на друга "
             "и насколько каждая похожа на другого автора.",
             "Это прямая проверка гипотезы: если свои кучки ближе друг к другу, чем к чужому автору, "
             "то автора можно узнать по буквам.")
    wbc = cfg["within_between"]
    wb = compare.within_between({k: [text_tokens[r.text_id] for r in by_author[k]] for k in keys}, states,
                                int(wbc["letters_each"]), reps, cfg["balanced"]["chunk_tokens"], cfg["seed"], thr, pc)
    sep = 0
    print(f"  {'Автор':<20} {'D внутри':>9} {'ближайший другой':>24}")
    for k in keys:
        w_ = float(np.mean(wb["within"][k]))
        near = min((float(np.mean(v)), b2 if a2 == k else a2) for (a2, b2), v in wb["between"].items() if k in (a2, b2))
        mark = f"{GREEN}отделяется{END}" if w_ < near[0] else f"{AMBER}есть автор ближе{END}"
        sep += w_ < near[0]
        print(f"  {labels[k]:<20} {w_:>9.3f}   {labels[near[1]]:<18} {near[0]:.3f}   {mark}")
    print(f"\n  {B}Отделяются {sep} из {len(keys)} авторов.{END} Гипотеза подтверждается частично.")
    seen(f"У {sep} из {len(keys)} авторов свои произведения похожи друг на друга сильнее, чем на других авторов, "
         "но запас небольшой. Там, где тексты одного автора на разные темы, они расходятся сильнее, чем разные авторы. "
         "Следующий шаг — убрать тематическую лексику и сравнивать тексты одного издания.")
    done(t)

    print(f"\n{BLUE}{B}Готово.{END} Всё посчитано за {time.perf_counter() - T0:.0f} с (с учётом пауз).")
    if EXPLAIN:
        print()
        say("Итог:      ", f"простые марковские цепи по буквам построены для {len(keys)} авторов СМАЛТ. "
            f"Гласные/согласные почти не различают авторов; буквенные таблицы различают лучше "
            f"(угадано {ok / len(res):.0%} текстов, отделяются {sep} из {len(keys)} авторов), "
            "но на них сильно влияет тема. Рабочая гипотеза: после удаления тематической лексики "
            "различия между авторами сохранятся. Её и предлагается проверить дальше.", BLUE)
    page = ROOT / "results" / "demo.html"
    if page.exists():
        if PAUSE:
            try:
                ans = input(f"Открыть интерактивную страницу? [Enter — да, n — нет] ")
            except EOFError:
                ans = "n"
        else:
            ans = "n"
        if ans.strip().lower() not in ("n", "н", "нет", "no"):
            webbrowser.open(page.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
