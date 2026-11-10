import json
import unittest
from pathlib import Path

from markov.preprocess import PreprocessConfig, PreprocessLog, letter_segments, vc_segments

ROOT = Path(__file__).resolve().parents[1]


def cfg() -> PreprocessConfig:
    return PreprocessConfig.from_dict(json.loads((ROOT / "config.json").read_text(encoding="utf-8")))


class TestPreprocess(unittest.TestCase):
    def test_space_breaks_chain(self):
        self.assertEqual(letter_segments("а а", cfg()), ["а", "а"])

    def test_punctuation_digits_hyphen_break(self):
        self.assertEqual(letter_segments("кто-то, 1861г.", cfg()), ["кто", "то", "г"])

    def test_hard_sign_is_letter_but_vc_breaker(self):
        c = cfg()
        segs = letter_segments("объект", c)
        self.assertEqual(segs, ["объект"])
        self.assertEqual(vc_segments(segs, c), ["VC", "VCC"])

    def test_soft_sign_vc_breaker(self):
        c = cfg()
        self.assertEqual(vc_segments(letter_segments("кольцо", c), c), ["CVC", "CV"])

    def test_yo_kept_separately(self):
        self.assertEqual(letter_segments("Ёлка ещё", cfg()), ["ёлка", "ещё"])

    def test_historical_letters_preserved(self):
        self.assertEqual(letter_segments("Свѣтъ Россіи Ѳома", cfg()), ["свѣтъ", "россіи", "ѳома"])

    def test_vc_classes_for_historical_letters(self):
        c = cfg()
        self.assertEqual(vc_segments(["ѣі"], c), ["VV"])
        self.assertEqual(vc_segments(["ѳ"], c), ["C"])

    def test_latin_i_homoglyph_repaired_only_in_cyrillic_tokens(self):
        c = cfg()
        log = PreprocessLog()
        # «россiи» с латинской i -> кириллическая і; латинское слово не трогаем
        self.assertEqual(letter_segments("россiи ira", c, log), ["россіи"])
        self.assertEqual(log.tokens_homoglyph_repaired, 1)
        self.assertEqual(log.dropped_letters["i"], 1)
        self.assertEqual(log.dropped_letters["r"], 1)

    def test_mixed_token_not_fully_fixable_is_left_alone(self):
        # 'роssumus': кириллические «ро» + латиница; замена не делает токен кириллическим
        self.assertEqual(letter_segments("роssumus", cfg()), ["ро"])

    def test_stress_marks_removed(self):
        c = cfg()
        self.assertEqual(letter_segments("что̀ чтó", c), ["что", "что"])
        # й и ё не разрушаются удалением ударений
        self.assertEqual(letter_segments("й ё", c), ["й", "ё"])

    def test_unknown_letters_logged_not_silently_lost(self):
        c = cfg()
        log = PreprocessLog()
        segs = letter_segments("ѵпостась", c, log)  # ѵ не входит в алфавит конфигурации
        self.assertEqual(segs, ["постась"])
        self.assertEqual(log.dropped_letters["ѵ"], 1)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            PreprocessConfig(alphabet="аб", vowels="а", consonants="", vc_breakers="")
        with self.assertRaises(ValueError):
            PreprocessConfig(alphabet="аб", vowels="аб", consonants="б", vc_breakers="")


if __name__ == "__main__":
    unittest.main()
