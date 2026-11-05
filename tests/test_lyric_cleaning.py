import itertools
import unittest

from tools.lyric_cleaning import (
    clean_brackets,
    clean_unbracketed_headers,
    deduplicate_lines,
    remove_punctuation,
    tokenize_text,
)


class BracketCleaningTests(unittest.TestCase):
    def test_bracketed_numbers_are_not_unknown_words(self):
        self.assertEqual(clean_brackets('Call me [808]'), 'Call me [808]')

    def test_all_disabled_options_preserve_raw_text_exactly(self):
        text = " \r\n[Verse 1: Beyoncé]\r\n[Produced by JAY-Z] [?] [Take me down] (echo) \r\n"
        self.assertEqual(clean_brackets(text, False, False, False, False, False, False), text)

    def test_section_and_performer_options_are_independent(self):
        expected = {
            (False, False): "[Verse 1: Beyoncé & JAY-Z]",
            (False, True): "[Verse 1]",
            (True, False): "[Beyoncé & JAY-Z]",
            (True, True): "",
        }
        for (sections, performers), result in expected.items():
            with self.subTest(sections=sections, performers=performers):
                self.assertEqual(clean_brackets(
                    "[Verse 1: Beyoncé & JAY-Z]",
                    remove_section_headers=sections,
                    remove_performer_names=performers,
                ), result)

    def test_each_metadata_flag_works_without_the_other_flags(self):
        labels = ["[Chorus]", "[Produced by Élodie]", "[???]"]
        for sections, production, unknown in itertools.product((False, True), repeat=3):
            with self.subTest(sections=sections, production=production, unknown=unknown):
                for label, remove in zip(labels, (sections, production, unknown)):
                    self.assertEqual(clean_brackets(
                        label,
                        remove_section_headers=sections,
                        remove_performer_names=False,
                        remove_production_credits=production,
                        remove_unknown_markers=unknown,
                    ), "" if remove else label)

    def test_all_flag_combinations_keep_only_the_requested_content(self):
        text = "[Verse 1: Beyoncé] [Produced by Élodie] [?] [Take me down] (echo)"
        for flags in itertools.product((False, True), repeat=6):
            sections, performers, production, unknown, symbols, sung = flags
            with self.subTest(flags=flags):
                result = clean_brackets(text, *flags)
                self.assertEqual("Verse 1" in result, not sections)
                self.assertEqual("Beyoncé" in result, not performers)
                self.assertEqual("Élodie" in result, not production)
                self.assertEqual("?" in result, not unknown)
                self.assertEqual("Take me down" in result, not sung)
                self.assertIn("(echo)", result)
                if symbols:
                    self.assertNotIn("[", result)
                    self.assertNotIn("]", result)

    def test_sung_words_unknown_names_and_parentheses_survive_defaults(self):
        text = "[Stressed] [Take me down] [Beyoncé] [RICHARD] (Take me down)"
        self.assertEqual(clean_brackets(text, remove_section_headers=True), text)

    def test_multiline_and_spoken_payloads_are_not_mistaken_for_metadata(self):
        texts = [
            "[RICHARD Spoken: actual content\nAnd another spoken line]",
            "[Spoken: actual sung text]",
            "[Spoken: I Want To Live]",
        ]
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(clean_brackets(text, remove_section_headers=True), text)
                self.assertEqual(clean_brackets(text, remove_sung_brackets=True), "")

    def test_explicit_sung_bracket_experiment_is_destructive(self):
        self.assertEqual(clean_brackets("[Take me down] (echo)", remove_sung_brackets=True), "(echo)")

    def test_symbol_removal_keeps_sung_contents_and_line_structure(self):
        text = "[Stressed]\n[Take me down] (echo)"
        self.assertEqual(clean_brackets(text, remove_bracket_symbols=True), "Stressed\nTake me down (echo)")
        self.assertEqual(clean_brackets("[Chorus]", remove_bracket_symbols=True), "Chorus")

    def test_removing_metadata_cannot_merge_words(self):
        self.assertEqual(clean_brackets("love[?]hate"), "love hate")
        self.assertEqual(clean_brackets("love[Chorus]hate", remove_section_headers=True), "love hate")

    def test_bracketed_payload_without_credit_shape_is_retained(self):
        self.assertEqual(clean_brackets("[Chorus: Walk it through the rain]", remove_section_headers=True),
                         "[Walk it through the rain]")


class UnbracketedHeaderTests(unittest.TestCase):
    def test_all_disabled_options_preserve_raw_lines(self):
        text = "  Chorus\r\nProduced by Beyoncé\r\nName: Sung words\r\n"
        self.assertEqual(clean_unbracketed_headers(text, False, False), text)

    def test_standalone_sections_are_removed_but_sentences_remain(self):
        text = "Chorus\nVerse1\nPre-Chorus 2\nI sing the chorus\nVerse 1 is over\nDrake: Keep this lyric"
        expected = "I sing the chorus\nVerse 1 is over\nDrake: Keep this lyric"
        self.assertEqual(clean_unbracketed_headers(text, remove_section_headers=True), expected)

    def test_inline_section_payload_is_retained_with_case_and_endings(self):
        text = "Chorus: Walk it through the Rain\r\nVerse 1: Do not delete me\r\nSpoken: Hello there"
        expected = "Walk it through the Rain\r\nDo not delete me\r\nHello there"
        self.assertEqual(clean_unbracketed_headers(text, remove_section_headers=True), expected)

    def test_production_credits_are_explicit_and_case_insensitive(self):
        text = "pRoDuCeD bY Beyoncé\nLYRICS BY Élodie\nMusic: 坂本龍一\nI was produced by trouble\nBeyoncé"
        self.assertEqual(clean_unbracketed_headers(text), "I was produced by trouble\nBeyoncé")

    def test_only_metadata_becomes_empty(self):
        text = "\nChorus\n \nVerse 1:\nProduced by Beyoncé\n\n"
        self.assertEqual(clean_unbracketed_headers(text, remove_section_headers=True), "")


class WordAndRepetitionTests(unittest.TestCase):
    def test_deduplicate_lines_preserves_first_spelling_and_blank_structure(self):
        text = "Écho!\n\n  e\u0301CHO! \nNew line\nNew   line\n"
        self.assertEqual(deduplicate_lines(text), "Écho!\n\nNew line\n")
        self.assertEqual(deduplicate_lines("love\nLove!\nlove"), "love\nLove!\n")

    def test_deduplication_keeps_distinct_casefolded_payloads(self):
        text = "Give me love\nGive me hate\n"
        self.assertEqual(deduplicate_lines(text), text)

    def test_unicode_tokens_keep_contractions_accents_and_linguistic_joiners(self):
        text = "I’m John's rock’n’roll café e\u0301lan мир 漢字 می\u200cروم क्\u200dष 2026 love/hate!"
        self.assertEqual(tokenize_text(text), [
            "I’m", "John's", "rock’n’roll", "café", "e\u0301lan", "мир", "漢字", "می\u200cروم", "क्\u200dष", "2026", "love", "hate",
        ])

    def test_punctuation_becomes_boundaries_without_changing_case_or_accents(self):
        text = "'Écho!' love/hate—WORLD\nI’m John's (мир), 漢字!"
        self.assertEqual(remove_punctuation(text), "Écho love hate WORLD\nI’m John's мир 漢字")

    def test_joiners_are_kept_only_inside_words_and_emoji_do_not_leave_tokens(self):
        text = "می\u200cروم क्\u200dष 🙋🏼\u200d♀️"
        self.assertEqual(tokenize_text(text), ["می\u200cروم", "क्\u200dष"])
        self.assertEqual(remove_punctuation(text), "می\u200cروم क्\u200dष")

    def test_empty_text_and_invalid_types(self):
        for function in (clean_brackets, clean_unbracketed_headers, deduplicate_lines, remove_punctuation):
            with self.subTest(function=function.__name__):
                self.assertEqual(function(""), "")
                with self.assertRaises(TypeError):
                    function(None)
        self.assertEqual(tokenize_text(""), [])
        with self.assertRaises(TypeError):
            tokenize_text(42)


if __name__ == "__main__":
    unittest.main()
