"""Regression checks for preserving meaning while normalizing lyric artifacts."""

import ast
import math
from pathlib import Path
import re
from types import SimpleNamespace
import unicodedata
import unittest
from unittest import mock

import pandas as pd

from text_quality import audit_lyrics, normalize_lyrics


class NormalizeLyricsTests(unittest.TestCase):
    def test_accents_are_composed_without_transliteration_or_case_changes(self):
        source = "Cafe\u0301, e\u0301lan; Noël; Łódź; Ελληνικά; привет; مرحبا; 한글; 漢字"
        actual = normalize_lyrics(source)
        self.assertEqual(actual, unicodedata.normalize("NFC", source))
        self.assertTrue(unicodedata.is_normalized("NFC", actual))

    def test_linguistic_joiners_and_emoji_sequences_survive(self):
        source = "می\u200cروم क्\u200dष 🙋🏼\u200d♀️ 👩\u200d👩\u200d👧\u200d👦"
        self.assertEqual(normalize_lyrics(source), source)

    def test_curly_and_legacy_c1_punctuation_are_normalized(self):
        source = "“I’m”—‘ready’… \x93We\x92re\x94\x96fine\x85Next"
        self.assertEqual(normalize_lyrics(source), '"I\'m"-\'ready\'... "We\'re"-fine\nNext')

    def test_unicode_spaces_and_line_separators_preserve_verse_structure(self):
        source = " A\u00a0B\u2005C\u205fD\u3000E \r\n F\rG\u2028H\u2029I\n\nJ\tK "
        self.assertEqual(normalize_lyrics(source), "A B C D E\nF\nG\nH\nI\n\nJ K")
        self.assertEqual(normalize_lyrics(source, preserve_lines=False), "A B C D E F G H I J K")

    def test_removable_controls_do_not_hide_replacement_or_mojibake_markers(self):
        source = "\ufeffA\u00adB\u200e\u202a\u202c\u2060C\x00D\ufffcE \ufffd cafÃ©"
        self.assertEqual(normalize_lyrics(source), "ABC D E \ufffd cafÃ©")

    def test_mixed_script_repair_is_opt_in_and_preserves_real_cyrillic(self):
        prefix = "A long English introduction with many ordinary Latin letters. " * 4
        source = prefix + "thе mе WЕ е Е ее привет mε hелlo ееa еa"
        self.assertEqual(normalize_lyrics(source), source)
        expected = prefix + "the me WE е Е ее привет mε hелlo ееa ea"
        self.assertEqual(normalize_lyrics(source, repair_mixed_script=True), expected)

    def test_mixed_script_repair_leaves_non_latin_majority_documents_alone(self):
        source = "thе привет мир Ελληνικά مرحبا"
        self.assertEqual(normalize_lyrics(source, repair_mixed_script=True), source)

    def test_zero_width_space_recovers_scraped_latin_words_and_other_boundaries(self):
        self.assertEqual(normalize_lyrics("h\u200bomo T\u200bhe St\u200bizzy"), "homo The Stizzy")
        self.assertEqual(normalize_lyrics("\u200b漢\u200b字\u200b"), "漢 字")

    def test_missing_scalars_pass_through_and_invalid_values_raise(self):
        for missing in (None, pd.NA, pd.NaT):
            with self.subTest(missing=repr(missing)):
                self.assertIs(normalize_lyrics(missing), missing)
        self.assertTrue(math.isnan(normalize_lyrics(float("nan"))))
        for value in (12, False, [], {}, pd.Series(["a"])):
            with self.subTest(value=repr(value)):
                with self.assertRaises(TypeError):
                    normalize_lyrics(value)

    def test_normalization_is_idempotent_across_options(self):
        source = "\ufeffCafe\u0301\u2005‘thе’\x92s\r\n\u200b🙋🏼\u200d♀️\n\nمرحبا\x00\ufffd"
        for repair in (False, True):
            for lines in (False, True):
                with self.subTest(repair=repair, lines=lines):
                    kwargs = {"repair_mixed_script": repair, "preserve_lines": lines}
                    once = normalize_lyrics(source, **kwargs)
                    self.assertEqual(normalize_lyrics(once, **kwargs), once)


class AuditLyricsTests(unittest.TestCase):
    def test_overlapping_flags_character_counts_and_duplicate_index_are_preserved(self):
        source = pd.Series(
            ["thе’s\u2005fine\u200b\ufffd’", "café’’", None, ""],
            index=pd.Index(["duplicate", "duplicate", "missing", "empty"], name="song"),
        )
        original = source.copy(deep=True)
        rows, characters = audit_lyrics(source)
        pd.testing.assert_series_equal(source, original)
        pd.testing.assert_index_equal(rows.index, source.index)
        first = rows.iloc[0]
        for name in ("has_non_ascii", "has_unicode_punctuation", "has_unicode_whitespace",
                     "has_format_controls", "has_replacement", "has_non_latin_letters",
                     "has_cyrillic_letters"):
            with self.subTest(flag=name):
                self.assertTrue(first[name])
        self.assertFalse(first["latin8_compatible"])
        self.assertEqual(first["mixed_script_words"], 1)
        self.assertEqual(first["latin_letters"], 7)
        self.assertEqual(first["non_latin_letters"], 1)
        self.assertEqual(first["non_latin_letter_ratio"], 1 / 8)
        self.assertFalse(rows.iloc[2]["is_text"])
        self.assertTrue(pd.isna(rows.iloc[2]["latin8_compatible"]))
        self.assertTrue(rows.iloc[3]["is_text"])
        self.assertTrue(rows.iloc[3]["latin8_compatible"])
        self.assertEqual(rows.iloc[3]["non_latin_letter_ratio"], 0)
        curly = characters.set_index("character").loc["’"]
        self.assertEqual(curly["occurrences"], 4)
        self.assertEqual(curly["rows"], 2)
        self.assertEqual(curly["codepoint"], "U+2019")
        self.assertNotIn("t", characters["character"].tolist())

    def test_latin8_incompatibility_is_not_a_non_latin_language_flag(self):
        rows, _ = audit_lyrics(pd.Series(["Hello, world!", "café", "don’t", "한글", "\u0301"]))
        self.assertEqual(rows["latin8_compatible"].tolist(), [True, True, False, False, False])
        self.assertEqual(rows["has_non_latin_letters"].tolist(), [False, False, False, True, False])
        self.assertEqual(rows["has_other_script_letters"].tolist(), [False, False, False, True, False])

    def test_mojibake_is_flagged_without_automatic_redecoding(self):
        rows, _ = audit_lyrics(pd.Series(["cafÃ©", "â€™", "plain", "\ufffd", "\ufffc", "\x91word\x92"]))
        self.assertEqual(rows["mojibake_suspected"].tolist(), [True, True, False, False, False, False])
        self.assertTrue(rows.iloc[3]["has_replacement"])
        self.assertTrue(rows.iloc[4]["has_object_placeholder"])
        self.assertTrue(rows.iloc[5]["has_c1_controls"])

    def test_empty_and_all_missing_series_keep_stable_schemas(self):
        empty = pd.Series([], index=pd.Index([], name="song"), dtype="object")
        rows, characters = audit_lyrics(empty)
        self.assertEqual(len(rows), 0)
        self.assertEqual(len(characters), 0)
        pd.testing.assert_index_equal(rows.index, empty.index)
        self.assertIn("latin8_compatible", rows.columns)
        self.assertIn("occurrences", characters.columns)
        self.assertEqual(str(rows["is_text"].dtype), "boolean")
        rows, characters = audit_lyrics(pd.Series([None, pd.NA, float("nan")]))
        self.assertEqual(rows["is_text"].tolist(), [False, False, False])
        self.assertTrue(rows["latin8_compatible"].isna().all())
        self.assertTrue(characters.empty)

    def test_invalid_items_are_rejected(self):
        for value in (123, False, ["lyrics"]):
            with self.subTest(value=repr(value)):
                with self.assertRaises(TypeError):
                    audit_lyrics(pd.Series(["valid", value], dtype="object"))


class CleanLyricsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the real pure function without importing unrelated Genius API
        # or NLTK integrations, whose dependencies/resources may not be installed.
        source = Path(__file__).resolve().parents[1] / "utils.py"
        parsed = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("clean_lyrics", "tokenize_lyrics")]
        cls.word_tokenize = mock.Mock()
        namespace = {
            "re": re, "unicodedata": unicodedata, "normalize_lyrics": normalize_lyrics,
            "nltk": SimpleNamespace(tokenize=SimpleNamespace(word_tokenize=cls.word_tokenize)),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        cls.clean_lyrics = staticmethod(namespace["clean_lyrics"])
        cls.tokenize_lyrics = staticmethod(namespace["tokenize_lyrics"])

    def test_word_preparation_preserves_accents_contractions_and_separates_punctuation(self):
        source = "[Verse 1]\r\nI’m Café—love/hate (Ναι), 한국어 and мир!"
        self.assertEqual(self.clean_lyrics(source), "i'm café love hate ναι 한국어 and мир")

    def test_word_preparation_handles_missing_and_empty_text(self):
        self.assertIsNone(self.clean_lyrics(None))
        self.assertIs(self.clean_lyrics(pd.NA), pd.NA)
        self.assertEqual(self.clean_lyrics("[Chorus] — /\n"), "")

    def test_word_preparation_preserves_meaningful_linguistic_joiners(self):
        source = "می\u200cروم क्\u200dष"
        self.assertEqual(self.clean_lyrics(source), source)

    def test_tokenization_retains_possessives_and_only_expands_exact_contractions(self):
        self.word_tokenize.reset_mock()
        self.word_tokenize.return_value = [
            "john", "'s", "guitar", "i", "'m", "ready", "do", "n't", "o'malley",
        ]
        actual = self.tokenize_lyrics("John’s guitar; I’m ready. Don’t, O’Malley!")
        self.assertEqual(actual, ["john", "'s", "guitar", "i", "am", "ready", "do", "not", "o'malley"])
        self.word_tokenize.assert_called_once_with("john's guitar i'm ready don't o'malley")


if __name__ == "__main__":
    unittest.main()
