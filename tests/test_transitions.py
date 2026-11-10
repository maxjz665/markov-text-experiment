import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from markov.preprocess import PreprocessConfig, letter_segments, vc_segments
from markov.transitions import count_transitions, merge, probabilities

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
PC = PreprocessConfig.from_dict(CFG)
ALPHABET = CFG["alphabet"]


def letters(text):
    return count_transitions(letter_segments(text, PC), ALPHABET)


def vc(text):
    return count_transitions(vc_segments(letter_segments(text, PC), PC), "VC")


def cell(m, a, b):
    return int(m.N[m.states.index(a), m.states.index(b)])


class TestTransitions(unittest.TestCase):
    # 1
    def test_abba(self):
        m = letters("абба")
        self.assertEqual((cell(m, "а", "б"), cell(m, "б", "б"), cell(m, "б", "а")), (1, 1, 1))
        self.assertEqual(m.total_pairs, 3)
        P = m.P
        observed = m.row_totals > 0
        np.testing.assert_allclose(np.nansum(P[observed], axis=1), 1.0)

    # 2
    def test_mama_vc(self):
        m = vc("мама")
        self.assertEqual(cell(m, "C", "V"), 2)
        self.assertEqual(cell(m, "V", "C"), 1)
        self.assertEqual(cell(m, "V", "V") + cell(m, "C", "C"), 0)
        self.assertEqual(m.total_pairs, 3)

    # 3
    def test_no_cross_word(self):
        m = letters("а а")
        self.assertEqual(m.total_pairs, 0)
        self.assertEqual(cell(m, "а", "а"), 0)

    # 4
    def test_object(self):
        m = letters("объект")
        self.assertEqual(cell(m, "б", "ъ"), 1)
        self.assertEqual(cell(m, "ъ", "е"), 1)
        v = vc("объект")
        # о б | е к т : V→C, V→C, C→C; нет C→V через ъ
        self.assertEqual(cell(v, "C", "V"), 0)
        self.assertEqual(cell(v, "V", "C"), 2)
        self.assertEqual(cell(v, "C", "C"), 1)
        self.assertEqual(v.total_pairs, 3)

    # 5
    def test_no_transition_across_files(self):
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = Path(d, "a.txt"), Path(d, "b.txt")
            p1.write_text("аб", encoding="utf-8")
            p2.write_text("ва", encoding="utf-8")
            models = [letters(p.read_text(encoding="utf-8")) for p in (p1, p2)]
        m = merge(models)
        self.assertEqual(m.total_pairs, 2)
        self.assertEqual(cell(m, "б", "в"), 0)  # конец первого файла -> начало второго

    # 6
    def test_unobserved_row_is_nan_not_uniform(self):
        m = letters("аб")
        P = m.P
        i_b = m.states.index("б")
        self.assertTrue(np.isnan(P[i_b]).all())
        i_a = m.states.index("а")
        self.assertEqual(P[i_a, m.states.index("б")], 1.0)
        P2 = probabilities(np.zeros((2, 2), dtype=int))
        self.assertTrue(np.isnan(P2).all())

    # 7
    def test_comparison_uses_probabilities_not_counts(self):
        short, long = letters("аб ав"), letters("аб ав " * 10)
        self.assertNotEqual(short.total_pairs, long.total_pairs)
        np.testing.assert_allclose(short.P, long.P, equal_nan=True)

    # 8 (см. также test_preprocess)
    def test_historical_letters_counted_as_states(self):
        m = letters("свѣтъ")
        self.assertEqual(cell(m, "в", "ѣ"), 1)
        self.assertEqual(cell(m, "т", "ъ"), 1)
        m2 = letters("ёж еж")
        self.assertEqual(cell(m2, "ё", "ж"), 1)
        self.assertEqual(cell(m2, "е", "ж"), 1)

    def test_pairs_equal_sum_N(self):
        text = "Въ началѣ было слово, и слово было у Бога."
        segs = letter_segments(text, PC)
        m = count_transitions(segs, ALPHABET)
        self.assertEqual(m.total_pairs, sum(len(s) - 1 for s in segs))
        self.assertEqual(int(m.unigrams.sum()), sum(len(s) for s in segs))

    def test_unknown_state_raises(self):
        with self.assertRaises(ValueError):
            count_transitions(["аz"], ALPHABET)


class TestReproducibility(unittest.TestCase):
    # 9
    def test_balanced_sample_is_deterministic(self):
        from markov.compare import balanced_sample
        docs = [[f"слово{k}" for k in range(1000)], [f"текст{k}" for k in range(500)]]
        a = balanced_sample(docs, letter_budget=2000, chunk_tokens=50, seed=42, preprocess=PC)
        b = balanced_sample(docs, letter_budget=2000, chunk_tokens=50, seed=42, preprocess=PC)
        self.assertEqual(a, b)
        self.assertGreaterEqual(sum(len(s) for s in a), 2000)


if __name__ == "__main__":
    unittest.main()
