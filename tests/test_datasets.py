import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from tools.datasets import (
    build_submission,
    duplicate_keys,
    load_datasets,
    split_training_data,
    training_keep_mask,
)


class DatasetLoadingTests(unittest.TestCase):
    def write_inputs(self, directory):
        train = pd.DataFrame({
            "id": [1, 2], "lyrics": ["café", "guitar"], "tag": ["pop", "rock"]
        })
        test = pd.DataFrame({
            "id": [4, 3], "lyrics": ["new", "other"], "tag": ["fake", "fake"]
        })
        template = pd.DataFrame({"id": [3, 4], "tag": ["placeholder", "placeholder"]})
        for name, frame in (("train", train), ("test", test), ("sample_submission", template)):
            frame.to_csv(directory / f"{name}.csv", index=False)
        return train, test, template

    def test_loads_only_raw_inputs_and_never_uses_placeholder_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected, _, template = self.write_inputs(directory)
            pd.DataFrame({"unrelated": [1]}).to_csv(directory / "clean_train.csv", index=False)
            original = {path.name: path.read_bytes() for path in directory.iterdir()}

            train, test, loaded_template = load_datasets(directory)
            submission = build_submission(test["id"], ["rock", "pop"], loaded_template)

            assert_frame_equal(train, expected)
            assert_frame_equal(loaded_template, template)
            self.assertNotIn("tag", test)
            self.assertEqual(submission["tag"].tolist(), ["pop", "rock"])
            self.assertEqual(original, {path.name: path.read_bytes() for path in directory.iterdir()})

    def test_rejects_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_inputs(directory)
            (directory / "train.csv").write_bytes(b"id,lyrics,tag\n1,caf\xe9,pop\n")
            with self.assertRaises(UnicodeDecodeError):
                load_datasets(directory)

    def test_rejects_bad_ids_missing_labels_and_mismatched_template(self):
        changes = [
            ("train", "id", [1, 1]),
            ("test", "id", [None, 3]),
            ("sample_submission", "id", [3, 3]),
            ("sample_submission", "id", [3, 9]),
            ("train", "tag", ["pop", None]),
        ]
        for name, column, values in changes:
            with self.subTest(name=name, column=column, values=values):
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    self.write_inputs(directory)
                    path = directory / f"{name}.csv"
                    frame = pd.read_csv(path)
                    frame[column] = values
                    frame.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_datasets(directory)


class DuplicateGroupingTests(unittest.TestCase):
    def test_canonical_groups_handle_headers_punctuation_and_repeated_lines(self):
        frame = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "lyrics": [
                "[Chorus: Beyoncé]\nCAFÉ!\n[Take me down]\nCAFÉ!",
                "cafe\u0301\nTake me down",
                "[Take me somewhere else]",
                "[Spoken: actual content\nKeep this line]",
                "actual content\nKeep this line",
            ],
        }, index=[7, 7, 8, 8, 8])
        original = frame.copy(deep=True)

        keys = duplicate_keys(frame)

        self.assertEqual(keys.iloc[0], keys.iloc[1])
        self.assertNotEqual(keys.iloc[1], keys.iloc[2])
        self.assertIn("actual content", keys.iloc[3])
        self.assertIn("keep this line", keys.iloc[3])
        self.assertEqual(keys.index.tolist(), frame.index.tolist())
        assert_frame_equal(frame, original)

    def test_empty_keys_are_song_specific_and_stable_when_shuffled(self):
        frame = pd.DataFrame({"id": [1, 2, 3], "lyrics": [None, "[Instrumental]", "!!!"]})
        keys = duplicate_keys(frame)
        shuffled = duplicate_keys(frame.iloc[[2, 0, 1]])

        self.assertEqual(keys.nunique(), 3)
        self.assertEqual(dict(zip(frame["id"], keys)), dict(zip(frame.iloc[[2, 0, 1]]["id"], shuffled)))

    def test_split_keeps_duplicate_groups_together_and_preserves_all_rows(self):
        records = []
        for group in range(30):
            tag = "rock" if group % 2 else "pop"
            records.extend([
                {"id": group * 2, "lyrics": f"[Verse]\nSong {group}!\nSong {group}!", "tag": tag},
                {"id": group * 2 + 1, "lyrics": f"song {group}", "tag": tag},
            ])
        frame = pd.DataFrame(records, index=[9] * len(records))
        original = frame.copy(deep=True)

        training, validation = split_training_data(frame)
        repeated_training, repeated_validation = split_training_data(frame)

        self.assertFalse(set(duplicate_keys(training)) & set(duplicate_keys(validation)))
        self.assertEqual(set(training["id"]) | set(validation["id"]), set(frame["id"]))
        self.assertFalse(set(training["id"]) & set(validation["id"]))
        self.assertEqual(set(training["tag"]), {"rock", "pop"})
        self.assertEqual(set(validation["tag"]), {"rock", "pop"})
        self.assertAlmostEqual(len(validation) / len(frame), 0.2)
        assert_frame_equal(training, repeated_training)
        assert_frame_equal(validation, repeated_validation)
        training.iloc[0, training.columns.get_loc("lyrics")] = "changed"
        assert_frame_equal(frame, original)

    def test_split_reports_impossible_class_coverage(self):
        frame = pd.DataFrame({
            "id": list(range(8)),
            "lyrics": ["same rare song"] * 2 + [f"other song {i}" for i in range(6)],
            "tag": ["rare"] * 2 + ["common"] * 6,
        })
        with self.assertRaisesRegex(ValueError, "independent lyric groups"):
            split_training_data(frame)

    def test_training_mask_uses_positional_labels_and_preserves_conflicts_and_empty(self):
        frame = pd.DataFrame({
            "lyrics": [f"raw {i}" for i in range(7)],
            "lyrics_processed": ["same", "same", "same", "", "  ", None, "different"],
        }, index=[4, 4, 4, 4, 4, 4, 4])
        labels = pd.Series(["rock", "rock", "pop", "rock", "rock", "rock", "rock"],
                           index=[8, 7, 6, 5, 4, 3, 2])

        mask = training_keep_mask(frame, labels)

        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.tolist(), [True, False, True, True, True, True, True])
        self.assertTrue(training_keep_mask(frame, labels, remove_duplicates=False).all())
        with self.assertRaisesRegex(ValueError, "equal length"):
            training_keep_mask(frame, labels.iloc[:-1])


class SubmissionTests(unittest.TestCase):
    def test_output_follows_template_order_with_shuffled_test_ids(self):
        template = pd.DataFrame({
            "id": [11, 13, 12], "tag": ["fake", "fake", "fake"], "extra": [1, 2, 3]
        }, index=[8, 8, 2])
        original = template.copy(deep=True)

        result = build_submission([12, 11, 13], pd.Series(["rock", "pop", "rap"], index=[5, 9, 1]), template)

        self.assertEqual(result.columns.tolist(), ["id", "tag"])
        self.assertEqual(result["id"].tolist(), [11, 13, 12])
        self.assertEqual(result["tag"].tolist(), ["pop", "rap", "rock"])
        self.assertEqual(result.index.tolist(), [0, 1, 2])
        assert_frame_equal(template, original)

    def test_rejects_invalid_ids_shapes_missing_predictions_and_count_mismatches(self):
        template = pd.DataFrame({"id": [1, 2], "tag": ["placeholder", "placeholder"]})
        cases = [
            ([1, 1], ["rock", "pop"]),
            ([1, None], ["rock", "pop"]),
            ([1, 3], ["rock", "pop"]),
            ([1, 2], ["rock"]),
            ([1, 2], ["rock", None]),
            ([1, 2], [["rock"], ["pop"]]),
            ([1, 2], "rock"),
        ]
        for ids, predictions in cases:
            with self.subTest(ids=ids, predictions=predictions):
                with self.assertRaises(ValueError):
                    build_submission(ids, predictions, template)
        duplicate_template = template.assign(id=[1, 1])
        with self.assertRaises(ValueError):
            build_submission([1, 2], ["rock", "pop"], duplicate_template)


if __name__ == "__main__":
    unittest.main()
