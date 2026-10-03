#!/usr/bin/env python3
"""Export tensors and inference configuration without optimizer or training paths."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmb_detection.runtime import build_model, runtime_config_from_checkpoint


def export_checkpoint(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    config = runtime_config_from_checkpoint(checkpoint)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    public = {
        "format_version": 1,
        "model_name": "CenSynCMB",
        "config": asdict(config),
        "model": {key: value.detach().cpu() for key, value in checkpoint["model"].items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(public, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
        for key, tensor in checkpoint["model"].items():
            if not torch.equal(tensor.cpu(), reloaded["model"][key]):
                raise RuntimeError(f"Tensor changed during export: {key}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "filename": destination.name,
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "model": "CenSynCMB",
        "format_version": 1,
        "config": asdict(config),
        "tensor_equality_verified": True,
    }
    destination.with_suffix(".sha256").write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.input, args.output), indent=2))
