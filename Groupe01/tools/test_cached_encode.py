import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "04_embeddings_index.ipynb"
NOTEBOOK_DATA = json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))
FUNCTION_CELL = next(
    cell
    for cell in NOTEBOOK_DATA["cells"]
    if cell["cell_type"] == "code" and "def cached_encode(" in "".join(cell["source"])
)
NAMESPACE = {"np": np}
exec("".join(FUNCTION_CELL["source"]), NAMESPACE)
cached_encode = NAMESPACE["cached_encode"]


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, missing):
        self.calls.append(list(missing))
        return np.asarray([[float(i + 1), 0.0] for i in missing], dtype=np.float32)


class CachedEncodeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.vector_path = self.root / "vectors.npy"
        self.id_path = self.root / "ids.npy"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_cache_loads_without_writes_or_encoding(self):
        np.save(self.vector_path, np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32))
        np.save(self.id_path, np.asarray([10, 20]))
        encoder = FakeEncoder()

        with mock.patch("numpy.save", wraps=np.save) as save, mock.patch(
            "os.replace", wraps=os.replace
        ) as replace:
            vectors, reused, computed = cached_encode(
                self.vector_path, self.id_path, np.asarray([10, 20]), encoder
            )

        self.assertEqual(encoder.calls, [])
        self.assertEqual(save.call_count, 0)
        self.assertEqual(replace.call_count, 0)
        self.assertEqual((reused, computed), (2, 0))
        np.testing.assert_array_equal(vectors, [[1.0, 0.0], [2.0, 0.0]])

    def test_partial_cache_encodes_only_missing_rows(self):
        np.save(self.vector_path, np.asarray([[10.0, 0.0], [30.0, 0.0]], dtype=np.float32))
        np.save(self.id_path, np.asarray([10, 30]))
        encoder = FakeEncoder()

        vectors, reused, computed = cached_encode(
            self.vector_path, self.id_path, np.asarray([10, 20, 30]), encoder
        )

        self.assertEqual(encoder.calls, [[1]])
        self.assertEqual((reused, computed), (2, 1))
        np.testing.assert_array_equal(vectors, [[10.0, 0.0], [2.0, 0.0], [30.0, 0.0]])
        np.testing.assert_array_equal(np.load(self.id_path), [10, 20, 30])
        np.testing.assert_array_equal(np.load(self.vector_path), vectors)

    def test_failed_write_preserves_existing_cache_files(self):
        np.save(self.vector_path, np.asarray([[7.0, 0.0]], dtype=np.float32))
        np.save(self.id_path, np.asarray([7]))
        old_vectors = self.vector_path.read_bytes()
        old_ids = self.id_path.read_bytes()
        encoder = FakeEncoder()
        real_replace = os.replace
        replace_calls = 0

        def fail_second_replace(source, destination):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 2:
                raise PermissionError("read-only")
            return real_replace(source, destination)

        with mock.patch("os.replace", side_effect=fail_second_replace):
            with self.assertRaisesRegex(PermissionError, "ids.npy.*probably read-only"):
                cached_encode(
                    self.vector_path, self.id_path, np.asarray([7, 8]), encoder
                )

        self.assertEqual(self.vector_path.read_bytes(), old_vectors)
        self.assertEqual(self.id_path.read_bytes(), old_ids)
        self.assertEqual(encoder.calls, [[1]])
        self.assertEqual(replace_calls, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
