from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("censyncmb_inference", ROOT / "scripts/run_inference.py")
inference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = inference
spec.loader.exec_module(inference)
torch.set_num_threads(1)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.affine = np.array([[-0.7, 0, 0, 11], [0, 0.8, 0, -8], [0, 0, 2, 3], [0, 0, 0, 1]])
        data = np.random.default_rng(7).normal(size=(12, 14, 16)).astype(np.float32)
        self.image = self.root / "swi.nii.gz"
        nib.save(nib.Nifti1Image(data, self.affine), self.image)
        self.record = {"sid": "subject01", "t2s": str(self.image)}

    def test_resampling_preserves_world_coordinates(self):
        prepared = inference.prepare_record(self.record, self.root)
        sample = inference.build_inference_transform()(prepared)
        expected = np.eye(4)
        expected[:3, 3] = [3.3, -8, 3]
        np.testing.assert_allclose(inference.affine_from_tensor(sample["t2s"]), expected, atol=1e-5)
        self.assertEqual(tuple(sample["t2s"].shape), (1, 9, 11, 31))
        for key in ("t1", "t2"):
            self.assertEqual(int(torch.count_nonzero(sample[key])), 0)

    def test_missing_spatial_metadata_is_rejected(self):
        with self.assertRaises(RuntimeError):
            inference.affine_from_tensor(np.zeros((3, 3, 3)))

    def test_diagonal_lesion_is_one_26_connected_component(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        for i in range(1, 6):
            mask[i, i, i] = 1
        kept = inference.remove_small_components(mask, 5)
        self.assertEqual(int(kept.sum()), 5)
        affine = np.eye(4)
        affine[:3, 3] = [10, -4, 8]
        lesions = inference.lesion_centroids(kept, affine)
        self.assertEqual(len(lesions), 1)
        self.assertEqual(lesions[0]["n_voxels"], 5)
        np.testing.assert_allclose([lesions[0][k] for k in ("x_ras_mm", "y_ras_mm", "z_ras_mm")], [13, -1, 11])

    def test_small_separate_component_is_removed(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[1:3, 1:3, 1:3] = 1
        mask[6, 6, 6] = 1
        self.assertEqual(int(inference.remove_small_components(mask, 5).sum()), 8)
        self.assertEqual(inference.lesion_centroids(np.zeros_like(mask), np.eye(4)), [])

    def test_explicit_bad_modality_path_is_not_zero_filled(self):
        with self.assertRaises(FileNotFoundError):
            inference.prepare_record(dict(self.record, t1=str(self.root / "typo.nii.gz")), self.root)

    def test_same_shape_but_different_affine_is_rejected(self):
        path = self.root / "misaligned.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((12, 14, 16)), np.eye(4)), path)
        with self.assertRaisesRegex(ValueError, "co-registered"):
            inference.prepare_record(dict(self.record, t1=str(path)), self.root)

    def load_manifest(self, records):
        manifest = self.root / "subjects.json"
        manifest.write_text(json.dumps(records))
        return inference.load_records(argparse.Namespace(manifest=manifest, sid=None, t1=None, t2=None, t2s=None))

    def test_manifest_paths_resolve_relative_to_manifest(self):
        records = self.load_manifest([{"sid": "one", "t2s": "swi.nii.gz"}])
        self.assertEqual(Path(records[0]["t2s"]), self.image.resolve())

    def test_duplicate_and_unsafe_subject_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load_manifest([self.record, self.record])
        with self.assertRaises(ValueError):
            self.load_manifest([dict(self.record, sid="../outside")])

    def test_empty_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load_manifest([])

    def test_cli_help(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_inference.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)

    @unittest.skipUnless(os.environ.get("CENSYNCMB_CHECKPOINT"), "set CENSYNCMB_CHECKPOINT to test real pretrained inference")
    def test_real_checkpoint_cli_keeps_saved_geometry(self):
        out = self.root / "predictions"
        command = [sys.executable, str(ROOT / "scripts/run_inference.py"), "--checkpoint", os.environ["CENSYNCMB_CHECKPOINT"],
                   "--t2s", str(self.image), "--sid", "synthetic", "--output-dir", str(out), "--device", "cpu",
                   "--roi-size", "32,32,32", "--sw-batch-size", "1", "--no-amp"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        probability = nib.load(out / "synthetic_censyncmb_prob.nii.gz")
        self.assertEqual(probability.header.get_xyzt_units()[0], "mm")
        expected = np.eye(4)
        expected[:3, 3] = [3.3, -8, 3]
        np.testing.assert_allclose(probability.affine, expected, atol=1e-5)
        values = probability.get_fdata()
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((0 <= values) & (values <= 1)).all())
        summary = json.loads((out / "censyncmb_inference_summary.json").read_text())
        with (out / "synthetic_censyncmb_centroids.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(summary["outputs"][0]["n_lesions"], len(rows))


if __name__ == "__main__":
    unittest.main()
