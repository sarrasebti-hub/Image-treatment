"""Small synthetic COCO fixtures. These tests do not produce scientific COCO results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from PIL import Image

from src.data.audit import audit
from src.data.coco import CocoData
from src.data.splits import check_splits, create_splits, load_splits
from src.embeddings.encode import check_matrix, generate_embeddings, load_embeddings
from src.indexing.faiss_index import build_indexes, load_indexes, search
from src.preprocessing.metadata import clean_caption, load_metadata, load_rgb, prepare_metadata
from src.preprocessing.quality import quality_control
from src.utils.core import load_config, normalize_vectors, path, read_json, write_json


def fixture(root):
    cfg = copy.deepcopy(load_config())
    cfg["_root"] = str(root)
    cfg.update(kb_size=4, validation_size=2, test_size=2)
    for source, ids in (("train2017", range(1, 11)), ("val2017", range(101, 107))):
        images, captions, instances = [], [], []
        folder = path(cfg, "train_images" if source == "train2017" else "val_images")
        folder.mkdir(parents=True, exist_ok=True)
        for i in ids:
            filename = f"{i:012}.jpg"
            Image.new("RGB", (32 + i % 3, 24), (i % 256, (i * 7) % 256, (i * 13) % 256)).save(folder / filename)
            images.append({"id": i, "width": 32 + i % 3, "height": 24, "file_name": filename})
            for j in range(5):
                captions.append({"id": i * 10 + j, "image_id": i, "caption": f" A red square near a person outside, synthetic {i} caption {j}.  "})
            for j in range(i % 9):
                instances.append({"id": i * 100 + j, "image_id": i, "category_id": 1 + j % 2,
                                  "bbox": [j, j, 8, 8], "area": 64, "iscrowd": 0})
        write_json(path(cfg, "annotations") / f"captions_{source}.json", {"images": images, "annotations": captions})
        write_json(path(cfg, "annotations") / f"instances_{source}.json", {"images": images, "annotations": instances,
                   "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "chair"}]})
    return cfg


class FakeEncoder:
    """Deterministic adapter double; never used by production CLI."""
    def __init__(self, cfg):
        self.device = "cpu"
        self.resolved_revision = "SYNTHETIC_TEST_DOUBLE"
        self.truncated_captions = 0

    def images(self, images):
        return normalize_vectors([[*np.asarray(img)[0, 0].tolist(), 1] for img in images])

    def texts(self, texts):
        return normalize_vectors([[len(text), sum(map(ord, text)) % 43, 3, 1] for text in texts])


class Phase1Tests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parents[1] / "tmp" / "unit_tests"
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.test_root)
        self.cfg = fixture(Path(self.temporary.name))
        self.data = CocoData(self.cfg)
        create_splits(self.cfg, self.data)

    def tearDown(self):
        if not Path(self.temporary.name).resolve().is_relative_to(self.test_root.resolve()):
            raise ValueError("Refusing cleanup outside the unit-test workspace")
        self.temporary.cleanup()

    def test_splits_are_disjoint_reproducible_and_provisional(self):
        splits = load_splits(self.cfg)
        check_splits(splits)
        create_splits(self.cfg, self.data, force=True)
        self.assertEqual(splits, load_splits(self.cfg))
        self.assertEqual(read_json(path(self.cfg, "splits") / "split_manifest.json")["status"], "PROVISIONAL_NOT_TEACHER_SPLIT")
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_splits({"knowledge_base": [1], "test": [1]})
        with self.assertRaises(FileExistsError):
            create_splits(self.cfg, self.data)

    def test_teacher_ids_take_precedence_and_count_is_not_overridden(self):
        write_json(Path(self.cfg["_root"]) / self.cfg["splits"]["official_test_candidate"], [101, 103])
        create_splits(self.cfg, self.data, force=True)
        self.assertEqual(load_splits(self.cfg)["test"], [101, 103])
        self.cfg["test_size"] = 3
        with self.assertRaisesRegex(ValueError, "Official"):
            create_splits(self.cfg, self.data, force=True)

    def test_raw_caption_preserved_and_rgb_loader(self):
        prepare_metadata(self.cfg, self.data)
        row = load_metadata(self.cfg, "knowledge_base")[0]
        raw = self.data.references("train2017", row["image_id"])[0]["caption"]
        self.assertEqual(row["captions"][0]["caption_raw"], raw)
        self.assertEqual(row["captions"][0]["caption_clean"], clean_caption(raw))
        self.assertTrue(raw.startswith(" "))
        filename = Path(self.temporary.name) / "gray.png"
        Image.new("L", (5, 4)).save(filename)
        self.assertEqual(load_rgb(filename).mode, "RGB")

    def test_normalization_rejects_invalid_vectors(self):
        normalized = normalize_vectors([[3, 4]])
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1)
        for values in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]]):
            with self.assertRaises(ValueError):
                normalize_vectors(values)
        with self.assertRaisesRegex(ValueError, "dimension"):
            check_matrix(normalized, [{"image_id": 1}], 3)

    def test_retrieval_excludes_all_captions_of_query(self):
        vectors = normalize_vectors([[1, 0], [1, .01], [0, 1]])
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        rows = [{"image_id": 1, "caption_id": 10}, {"image_id": 1, "caption_id": 11}, {"image_id": 2, "caption_id": 20}]
        self.assertEqual([h["image_id"] for h in search(index, rows, [1, 0], 2, 1)], [2])

    def test_missing_image_and_missing_caption_fail_with_report(self):
        ids = load_splits(self.cfg)["knowledge_base"]
        record = self.data.captions["train2017"].imgs[ids[0]]
        self.data.image_path("train2017", record).unlink()
        self.data.captions["train2017"].imgToAnns[ids[1]].pop()
        with self.assertRaises(ValueError):
            audit(self.cfg, self.data)
        report = read_json(path(self.cfg, "reports") / "audit.json")
        self.assertFalse(report["passed"])
        self.assertTrue({"missing_image", "caption_count"}.issubset({i["kind"] for i in report["issues"]}))

    def test_pipeline_mapping_cache_and_qc(self):
        prepare_metadata(self.cfg, self.data)
        with patch("src.embeddings.encode.Encoder", FakeEncoder):
            generate_embeddings(self.cfg, self.data)
        arrays, rows, _ = load_embeddings(self.cfg)
        self.assertEqual(len(rows["image"]), self.cfg["kb_size"])
        self.assertEqual(len(rows["text"]), self.cfg["kb_size"] * self.cfg["expected_captions"])
        # Reuse must not instantiate an encoder or recompute vectors.
        with patch("src.embeddings.encode.Encoder", side_effect=AssertionError("recomputed")):
            self.assertEqual(generate_embeddings(self.cfg, self.data)["status"], "reused_verified")
        # Simulate an interrupted run after shards, before the final manifest.
        (path(self.cfg, "embeddings") / "manifest.json").unlink()
        with patch("src.embeddings.encode.Encoder", FakeEncoder), \
                patch.object(FakeEncoder, "images", side_effect=AssertionError("image checkpoint recomputed")), \
                patch.object(FakeEncoder, "texts", side_effect=AssertionError("text checkpoint recomputed")):
            generate_embeddings(self.cfg, self.data)
        build_indexes(self.cfg)
        load_indexes(self.cfg)
        self.assertTrue(quality_control(self.cfg, self.data)["passed"])
        self.cfg["embedding"]["batch_size"] += 1
        with self.assertRaisesRegex(ValueError, "stale"):
            load_embeddings(self.cfg)

    def test_tampered_mapping_is_detected(self):
        prepare_metadata(self.cfg, self.data)
        with patch("src.embeddings.encode.Encoder", FakeEncoder):
            generate_embeddings(self.cfg, self.data)
        filename = path(self.cfg, "embeddings") / "image_rows.json"
        rows = read_json(filename)
        rows.reverse()
        write_json(filename, rows)
        with self.assertRaisesRegex(ValueError, "changed"):
            load_embeddings(self.cfg)

    def test_cross_split_exact_duplicate_fails(self):
        splits = load_splits(self.cfg)
        kb = self.data.captions["train2017"].imgs[splits["knowledge_base"][0]]
        test = self.data.captions["val2017"].imgs[splits["test"][0]]
        import shutil
        shutil.copyfile(self.data.image_path("train2017", kb), self.data.image_path("val2017", test))
        with self.assertRaises(ValueError):
            audit(self.cfg, self.data)
        self.assertTrue(any(i["kind"] == "duplicate_file" for i in read_json(path(self.cfg, "reports") / "audit.json")["issues"]))

    def test_spatial_and_caption_provenance(self):
        from src.preprocessing.metadata import caption_attributes, spatial_relations
        attrs = caption_attributes(["A red square outside at night."])
        self.assertEqual(attrs["source"], "caption_heuristic")
        self.assertIn("night", attrs["values"]["context"])
        boxes = [{"id": 1, "bbox": [0, 0, 2, 2]}, {"id": 2, "bbox": [4, 4, 2, 2]}]
        counts = spatial_relations(boxes, .1)["counts"]
        self.assertEqual(counts["left_of"], 1)
        self.assertEqual(counts["above"], 1)

    def test_bulk_validation_and_manual_query_outputs(self):
        from src.indexing.retrieval import manual_retrieval, validation_retrieval
        prepare_metadata(self.cfg, self.data)
        with patch("src.embeddings.encode.Encoder", FakeEncoder):
            generate_embeddings(self.cfg, self.data)
        build_indexes(self.cfg)
        with patch("src.indexing.retrieval.frozen_encoder", return_value=FakeEncoder(self.cfg)):
            result = validation_retrieval(self.cfg, self.data, top_k=2)
            self.assertEqual(result["queries_count"], self.cfg["validation_size"])
            folder = Path(result["output"])
            self.assertTrue((folder / "results.csv").is_file())
            self.assertTrue((folder / "gallery.html").is_file())
            self.assertEqual(len(read_json(folder / "results.json")["queries"]), 2)
            self.assertTrue(quality_control(self.cfg, self.data)["passed"])
            query_id = load_splits(self.cfg)["knowledge_base"][0]
            manual = manual_retrieval(self.cfg, self.data, image_id=query_id, top_k=2)
            self.assertEqual(len(manual["results"]), 2)
            self.assertNotIn(query_id, [h["image_id"] for h in manual["results"]])
            filename = self.data.image_path("train2017", self.data.captions["train2017"].imgs[query_id])
            by_path = manual_retrieval(self.cfg, self.data, image_path=str(filename), top_k=3)
            self.assertNotIn(query_id, [h["image_id"] for h in by_path["results"]])
            with self.assertRaises(ValueError):
                manual_retrieval(self.cfg, self.data, image_id=query_id, image_path=str(filename))

    def test_degenerate_box_preserved_but_excluded_from_geometry(self):
        from src.preprocessing.metadata import spatial_relations
        from src.utils.geometry import bbox_status
        box = [296.65, 388.33, 1.03, 0.0]
        self.assertEqual(bbox_status(box), "degenerate")
        objects = [{"id": 918, "bbox": box}, {"id": 919, "bbox": [1, 1, 3, 3]}]
        relations = spatial_relations(objects, .1)
        self.assertEqual(relations["excluded_annotation_ids"], [918])
        self.assertEqual(relations["counts"].get("left_of", 0), 0)

    def test_atomic_json_retries_transient_windows_lock(self):
        filename = Path(self.cfg["_root"]) / "atomic.json"
        write_json(filename, {"old": True})
        original = Path.replace
        attempts = []

        def transient(source, destination):
            attempts.append(1)
            if len(attempts) == 1:
                self.assertEqual(read_json(filename), {"old": True})
                raise PermissionError("simulated Windows reader lock")
            return original(source, destination)

        with patch.object(Path, "replace", new=transient), patch("src.utils.core.time.sleep"):
            write_json(filename, {"new": True})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(read_json(filename), {"new": True})

    def test_extend_test_preserves_training_validation_and_original_test(self):
        from src.data.splits import extend_test
        old = load_splits(self.cfg)
        self.cfg["test_size"] = 5
        extend_test(self.cfg, self.data)
        new = load_splits(self.cfg)
        self.assertEqual(new["knowledge_base"], old["knowledge_base"])
        self.assertEqual(new["validation"], old["validation"])
        self.assertEqual(len(new["test"]), 5)
        self.assertTrue(set(old["test"]).issubset(new["test"]))
        self.assertEqual(read_json(path(self.cfg, "splits") / "comparison/test_100_ids.json"), old["test"])
        self.assertEqual(extend_test(self.cfg, self.data)["status"], "already_extended")

    def test_reference_confidence_does_not_become_attribute_confidence(self):
        prepare_metadata(self.cfg, self.data)
        row = load_metadata(self.cfg, "knowledge_base")[0]
        self.assertTrue(all(c["confidence"] == 1 and c["source"] == "coco_human_reference" for c in row["captions"]))
        self.assertNotIn("confidence", row["derived_attributes"])
        from src.embeddings.encode import input_identity
        changed = copy.deepcopy([row])
        before = input_identity(self.cfg, changed)
        changed[0]["derived_attributes"]["note"] = "analysis metadata only"
        self.assertEqual(input_identity(self.cfg, changed), before)
        changed[0]["captions"][0]["caption_clean"] += " different encoder input"
        self.assertNotEqual(input_identity(self.cfg, changed), before)

    def test_download_unsatisfiable_resume_restarts_full_response(self):
        from io import BytesIO
        from urllib.error import HTTPError
        from src.data.download import fetch
        filename = Path(self.cfg["_root"]) / "resumed.jpg"
        filename.with_suffix(".jpg.part").write_bytes(b"old partial")
        response = BytesIO(b"fresh complete body")
        response.status = 200
        response.headers = {"Content-Length": str(len(response.getvalue()))}
        with patch("src.data.download.urlopen", side_effect=[HTTPError("https://example.test", 416, "Range", {}, None), response]) as opener:
            fetch("https://example.test/file.jpg", filename, self.cfg)
        self.assertEqual(filename.read_bytes(), b"fresh complete body")
        self.assertEqual(opener.call_count, 2)
        self.assertIsNone(opener.call_args_list[1].args[0].get_header("Range"))


if __name__ == "__main__":
    unittest.main()
