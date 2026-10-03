#!/usr/bin/env python3
"""Download the release checkpoint and verify its recorded SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "weights")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "weights" / "manifest.json").read_text())
    destination = args.output_dir / manifest["filename"]
    if destination.exists():
        if sha256(destination) == manifest["sha256"]:
            print(f"Verified existing checkpoint: {destination}")
            return 0
        parser.error(f"Existing file has an unexpected checksum: {destination}. Move it aside before retrying.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output_dir, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            request = Request(manifest["download_url"], headers={"User-Agent": "CenSynCMB-weight-downloader"})
            with urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        except HTTPError as exc:
            stream.close()
            temporary.unlink(missing_ok=True)
            if exc.code == 404:
                print("The pretrained asset is not published yet. Check the repository's Releases page or contact the maintainer.", file=sys.stderr)
            else:
                print(f"Weight download failed: HTTP {exc.code}", file=sys.stderr)
            return 1
        except (URLError, OSError) as exc:
            stream.close()
            temporary.unlink(missing_ok=True)
            print(f"Weight download failed: {exc}", file=sys.stderr)
            return 1
    try:
        if temporary.stat().st_size != manifest["bytes"] or sha256(temporary) != manifest["sha256"]:
            raise ValueError("Checkpoint checksum or size mismatch; the downloaded file was not installed.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
