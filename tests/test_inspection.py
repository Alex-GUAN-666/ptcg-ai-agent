"""Public tests use fabricated weights and inputs, never the competition model."""

import base64
import contextlib
import hashlib
import io
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from tools.inspect_submission import check_expected, inspect_archive, main, read_archive
from tools.smoke_submission import fixture, validate_selection


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ptcg_inspect_test_")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "example.zip"

    def make_archive(self, *, count=6, values=None, layout=None, checksum=None,
                     deck=None, extra_source="", compression=zipfile.ZIP_DEFLATED):
        values = values if values is not None else [float(i) / 10 for i in range(6)]
        payload = struct.pack("<" + "f" * len(values), *values)
        fields = {
            "_PARAMETER_FLOAT_COUNT": count,
            "_PARAMETER_SHA256": checksum or hashlib.sha256(payload).hexdigest(),
            "_PARAMETER_LAYOUT": layout if layout is not None else [("test.weight", (2, 3), 0, 6)],
            "_PARAMETER_B64_CHUNKS": (base64.b64encode(payload),),
        }
        source = "\n".join(f"{name} = {value!r}" for name, value in fields.items()) + "\n" + extra_source
        with zipfile.ZipFile(self.path, "w", compression=compression) as archive:
            archive.writestr("main.py", source)
            archive.writestr("deck.csv", deck if deck is not None else "1\n" * 60)
        return self.path

    def test_valid_synthetic_archive(self):
        report = inspect_archive(self.make_archive())
        self.assertEqual(report["parameters"]["stored_float32_values"], 6)
        self.assertEqual(report["deck"]["card_count"], 60)
        self.assertFalse(report["code_executed"])

    def test_source_is_not_executed(self):
        report = inspect_archive(self.make_archive(extra_source="raise RuntimeError('must not run')"))
        self.assertFalse(report["code_executed"])

    def test_checksum_failure(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            inspect_archive(self.make_archive(checksum="0" * 64))

    def test_byte_count_failure(self):
        with self.assertRaisesRegex(ValueError, "length"):
            inspect_archive(self.make_archive(count=7))

    def test_non_finite_weights(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            inspect_archive(self.make_archive(values=[0, 1, 2, 3, 4, float("nan")]))

    def test_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape/offset"):
            inspect_archive(self.make_archive(layout=[("w", (2, 2), 0, 6)]))

    def test_gap_in_layout(self):
        with self.assertRaisesRegex(ValueError, "offsets"):
            inspect_archive(self.make_archive(layout=[("w", (5,), 1, 6)]))

    def test_duplicate_tensor_name(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            inspect_archive(self.make_archive(layout=[("w", (3,), 0, 3), ("w", (3,), 3, 6)]))

    def test_deck_count_failure(self):
        with self.assertRaisesRegex(ValueError, "60"):
            inspect_archive(self.make_archive(deck="1\n" * 59))

    def test_expected_member_identity(self):
        report = inspect_archive(self.make_archive())
        changed = inspect_archive(self.make_archive(extra_source="# different source\n"))
        with self.assertRaisesRegex(ValueError, "main.py"):
            check_expected(changed, report)

    def test_recompression_preserves_member_identity(self):
        first = inspect_archive(self.make_archive(compression=zipfile.ZIP_STORED))
        second = inspect_archive(self.make_archive())
        self.assertNotEqual(first["archive_sha256"], second["archive_sha256"])
        check_expected(second, first)

    def test_reject_unexpected_member(self):
        with zipfile.ZipFile(self.make_archive(), "a") as archive:
            archive.writestr("unexpected.txt", "not permitted")
        with self.assertRaisesRegex(ValueError, "exactly"):
            read_archive(self.path)

    def test_reject_path_traversal(self):
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("../main.py", "")
            archive.writestr("deck.csv", "")
        with self.assertRaisesRegex(ValueError, "members"):
            read_archive(self.path)

    def test_reject_duplicate_member(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "w") as archive:
                archive.writestr("main.py", "")
                archive.writestr("main.py", "")
        with self.assertRaisesRegex(ValueError, "members"):
            read_archive(self.path)

    def test_reject_symlink(self):
        with zipfile.ZipFile(self.path, "w") as archive:
            entry = zipfile.ZipInfo("main.py")
            entry.create_system = 3
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(entry, "elsewhere.py")
            archive.writestr("deck.csv", "")
        with self.assertRaisesRegex(ValueError, "Links"):
            read_archive(self.path)

    def test_archive_size_limit(self):
        self.make_archive()
        with patch("tools.inspect_submission.MAX_ARCHIVE_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "size"):
                read_archive(self.path)

    def test_uncompressed_size_limit(self):
        self.make_archive()
        with patch("tools.inspect_submission.MAX_UNPACKED_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "Uncompressed"):
                read_archive(self.path)

    def test_duplicate_parameter_declaration_rejected(self):
        self.make_archive(extra_source="_PARAMETER_SHA256 = str('dynamic')")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            inspect_archive(self.path)

    def test_cli_missing_archive_fails_cleanly(self):
        with patch("sys.argv", ["inspect_submission", str(self.path)]):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(main(), 1)
            self.assertIn("Inspection failed", error.getvalue())


class SelectionContractTests(unittest.TestCase):
    def setUp(self):
        self.observation = fixture([{"type": 1}, {"type": 2}], 1, 1)

    def test_valid_selection(self):
        validate_selection([0], self.observation)

    def test_out_of_range_selection(self):
        with self.assertRaisesRegex(ValueError, "index out"):
            validate_selection([2], self.observation)

    def test_duplicate_selection(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_selection([0, 0], self.observation)

    def test_invalid_count(self):
        with self.assertRaisesRegex(ValueError, "count"):
            validate_selection([], self.observation)

    def test_boolean_not_an_index(self):
        with self.assertRaisesRegex(ValueError, "integers"):
            validate_selection([True], self.observation)


if __name__ == "__main__":
    unittest.main()
