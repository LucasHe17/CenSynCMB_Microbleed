import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location("downloader", Path(__file__).resolve().parents[1] / "scripts/download_weights.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "weights").mkdir()
        self.payload = b"synthetic download fixture"
        manifest = {"filename": "model.pth", "bytes": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest(),
                    "download_url": "https://example.invalid/model.pth"}
        (self.root / "weights/manifest.json").write_text(json.dumps(manifest))
        self.root_patch = patch.object(downloader, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.argv_patch = patch("sys.argv", ["download_weights.py"])
        self.argv_patch.start()
        self.addCleanup(self.argv_patch.stop)

    def test_download_checksum_and_cached_file(self):
        with patch.object(downloader, "urlopen", return_value=io.BytesIO(self.payload)) as call:
            self.assertEqual(downloader.main(), 0)
            self.assertEqual(downloader.main(), 0)
            self.assertEqual(call.call_count, 1)

    def test_corrupt_download_is_not_installed(self):
        with patch.object(downloader, "urlopen", return_value=io.BytesIO(b"corrupt")):
            with self.assertRaises(ValueError):
                downloader.main()
        self.assertFalse((self.root / "weights/model.pth").exists())
        self.assertFalse(list((self.root / "weights").glob("*.partial")))

    def test_unpublished_release_has_clear_error(self):
        with patch.object(downloader, "urlopen", side_effect=HTTPError("url", 404, "Not Found", {}, None)):
            self.assertEqual(downloader.main(), 1)
        self.assertFalse(list((self.root / "weights").glob("*.partial")))


if __name__ == "__main__":
    unittest.main()
