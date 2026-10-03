#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
from scipy import ndimage as ndi
from torch.amp import autocast

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cmb_detection.runtime import (  # noqa: E402
    RuntimeConfig,
    build_model,
    pick_main_output,
    runtime_config_from_checkpoint,
    segmentation_logits,
    unpack_outputs,
)


def parse_tuple3(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three positive integers, e.g. 128,128,128") from exc
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("expected three positive integers, e.g. 128,128,128")
    return parts


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.manifest is None:
        if args.t2s is None:
            raise SystemExit("Either --manifest or --t2s is required.")
        records = [
            {
                "sid": args.sid or Path(args.t2s).name.replace(".nii.gz", "").replace(".nii", ""),
                "t1": str(args.t1) if args.t1 else None,
                "t2": str(args.t2) if args.t2 else None,
                "t2s": str(args.t2s),
            }
        ]

    else:
        if any(value is not None for value in (args.t1, args.t2, args.t2s, args.sid)):
            raise ValueError("Use either --manifest or single-subject arguments, not both.")
        payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        records = payload.get("subjects", payload.get("records")) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a non-empty list of subject records.")
    base = Path(args.manifest).resolve().parent if args.manifest else Path.cwd()
    validated = []
    seen = set()
    for record in records:
        if not isinstance(record, dict) or not record.get("t2s"):
            raise ValueError("Every record must be an object with a 't2s' image path.")
        record = dict(record)
        sid = str(record.get("sid") or Path(record["t2s"]).name.replace(".nii.gz", "").replace(".nii", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sid):
            raise ValueError("Subject IDs must start with a letter or digit and contain only letters, digits, _, - or .")
        if sid in seen:
            raise ValueError(f"Duplicate subject ID: {sid}")
        seen.add(sid)
        record["sid"] = sid
        for key in ("t1", "t2", "t2s"):
            if record.get(key):
                path = Path(record[key]).expanduser()
                record[key] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
        validated.append(record)
    return validated


def make_zero_like(reference_path: Path, output_path: Path) -> Path:
    ref = nib.load(str(reference_path))
    data = np.zeros(ref.shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, ref.affine, ref.header), str(output_path))
    return output_path


def prepare_record(record: dict[str, Any], temp_dir: Path) -> dict[str, Any]:
    if not record.get("t2s"):
        raise ValueError(f"Record {record!r} is missing required key 't2s'.")
    t2s = Path(record["t2s"])
    if not t2s.is_file():
        raise FileNotFoundError(t2s)
    reference = nib.load(str(t2s))
    if len(reference.shape) != 3:
        raise ValueError(f"T2*/SWI must be a 3D NIfTI image, got {reference.shape}.")
    prepared = dict(record)
    for key in ("t1", "t2"):
        value = prepared.get(key)
        if value:
            if not Path(value).is_file():
                raise FileNotFoundError(f"Explicit {key} image does not exist: {value}")
            other = nib.load(str(value))
            if other.shape != reference.shape or not np.allclose(other.affine, reference.affine, rtol=0, atol=1e-4):
                raise ValueError(f"{key} must be co-registered and resampled to the T2*/SWI grid (same shape and affine).")
            continue
        prepared[key] = str(make_zero_like(t2s, temp_dir / f"{prepared.get('sid', t2s.stem)}_{key}_zero.nii.gz"))
    return prepared


def build_inference_transform() -> Compose:
    image_keys = ["t1", "t2", "t2s"]
    nan_guard = Lambdad(
        keys=image_keys,
        # Torch operations preserve MONAI MetaTensor affine and transform history.
        func=lambda x: torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32),
    )
    return Compose(
        [
            LoadImaged(keys=image_keys),
            EnsureChannelFirstd(keys=image_keys),
            Orientationd(keys=image_keys, axcodes="RAS"),
            Spacingd(
                keys=image_keys,
                pixdim=(1.0, 1.0, 1.0),
                mode=("trilinear", "trilinear", "trilinear"),
                align_corners=False,
            ),
            nan_guard,
            ScaleIntensityRangePercentilesd(
                keys=image_keys, lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True
            ),
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
            EnsureTyped(keys=image_keys, dtype=np.float32),
        ]
    )


def config_from_checkpoint(ckpt: dict[str, Any], args: argparse.Namespace) -> RuntimeConfig:
    config = runtime_config_from_checkpoint(ckpt)
    if args.roi_size is not None:
        config.roi_size = args.roi_size
    if args.sw_batch_size is not None:
        config.sw_batch_size = int(args.sw_batch_size)
    if args.sw_overlap is not None:
        config.sw_overlap = float(args.sw_overlap)
    if args.no_amp:
        config.amp = False
    if config.sw_batch_size < 1 or not 0 <= config.sw_overlap < 1:
        raise ValueError("Sliding-window batch size must be positive and overlap must be in [0, 1).")
    if any(v <= 0 or v % 16 for v in config.roi_size):
        raise ValueError("For the released Attention U-Net, each ROI dimension must be a positive multiple of 16.")
    return config


def affine_from_tensor(image: Any) -> np.ndarray:
    meta = getattr(image, "meta", None)
    affine = meta.get("affine") if isinstance(meta, dict) else None
    if affine is None:
        raise RuntimeError("Preprocessed image is missing its spatial affine; refusing to write misaligned output.")
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Invalid preprocessed affine.")
    return affine


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1:
        return mask.astype(np.uint8)
    labels, _ = ndi.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=bool))
    keep = np.bincount(labels.ravel()) >= int(min_voxels)
    keep[0] = False
    return keep[labels].astype(np.uint8)


def lesion_centroids(mask: np.ndarray, affine: np.ndarray) -> list[dict[str, Any]]:
    """Return 26-connected lesions in zero-based model-grid voxels and RAS mm."""
    labels, count = ndi.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return []
    centers = ndi.center_of_mass(mask, labels, range(1, count + 1))
    sizes = np.bincount(labels.ravel())
    lesions = []
    for index, center in enumerate(centers, start=1):
        world = nib.affines.apply_affine(affine, center)
        lesions.append(dict(lesion_id=index, n_voxels=int(sizes[index]),
                            i=float(center[0]), j=float(center[1]), k=float(center[2]),
                            x_ras_mm=float(world[0]), y_ras_mm=float(world[1]), z_ras_mm=float(world[2])))
    return lesions


def predict_one(
    record: dict[str, Any],
    transform: Compose,
    model: torch.nn.Module,
    config: RuntimeConfig,
    device: torch.device,
    args: argparse.Namespace,
    temp_dir: Path,
) -> dict[str, Any]:
    prepared = prepare_record(record, temp_dir)
    sample = transform(prepared)
    inputs = torch.cat([sample["t2s"], sample["t1"], sample["t2"]], dim=0).unsqueeze(0).to(device)

    def predictor(x: torch.Tensor) -> torch.Tensor:
        output = model(x)
        return segmentation_logits(pick_main_output(unpack_outputs(output)))

    with torch.no_grad(), autocast("cuda", enabled=bool(config.amp) and device.type == "cuda"):
        logits = sliding_window_inference(
            inputs,
            roi_size=tuple(int(v) for v in config.roi_size),
            sw_batch_size=int(config.sw_batch_size),
            overlap=float(config.sw_overlap),
            predictor=predictor,
        )
    prob = torch.softmax(logits, dim=1)[0, 1].detach().cpu().float().numpy()
    mask = remove_small_components(prob >= float(args.threshold), int(args.min_pred_voxels))

    sid = str(prepared.get("sid") or Path(prepared["t2s"]).name.replace(".nii.gz", "").replace(".nii", ""))
    affine = affine_from_tensor(sample["t2s"])
    prob_path = args.output_dir / f"{sid}_censyncmb_prob.nii.gz"
    mask_path = args.output_dir / f"{sid}_censyncmb_mask_thr{str(args.threshold).replace('.', 'p')}.nii.gz"
    for data, path in ((prob.astype(np.float32), prob_path), (mask.astype(np.uint8), mask_path)):
        image = nib.Nifti1Image(data, affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, str(path))
    lesions = lesion_centroids(mask, affine)
    centroids_path = args.output_dir / f"{sid}_censyncmb_centroids.csv"
    with centroids_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["lesion_id", "n_voxels", "i", "j", "k", "x_ras_mm", "y_ras_mm", "z_ras_mm"])
        writer.writeheader()
        writer.writerows(lesions)
    return {
        "sid": sid,
        "t2s": prepared["t2s"],
        "probability_map": str(prob_path),
        "binary_mask": str(mask_path),
        "centroids": str(centroids_path),
        "n_lesions": len(lesions),
        "missing_modalities": [key for key in ("t1", "t2") if not record.get(key)],
        "threshold": float(args.threshold),
        "min_pred_voxels": int(args.min_pred_voxels),
        "n_pred_voxels": int(mask.sum()),
        "output_space": "RAS, 1.0 mm isotropic model-preprocessed grid",
        "affine": affine.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CenSynCMB inference on one subject or a JSON manifest.")
    parser.add_argument("--checkpoint", type=Path, default=PACKAGE_ROOT / "weights" / "censyncmb_v1.0.0.pth")
    parser.add_argument("--manifest", type=Path, help="JSON list, or object with 'subjects'/'records'.")
    parser.add_argument("--sid", help="Subject id for single-subject mode.")
    parser.add_argument("--t2s", type=Path, help="T2*/SWI image path for single-subject mode.")
    parser.add_argument("--t1", type=Path, help="Optional T1 path; zero-filled if omitted.")
    parser.add_argument("--t2", type=Path, help="Optional T2 path; zero-filled if omitted.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-pred-voxels", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--roi-size", type=parse_tuple3, default=None)
    parser.add_argument("--sw-batch-size", type=int, default=None)
    parser.add_argument("--sw-overlap", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1 or args.min_pred_voxels < 1:
        parser.error("--threshold must be in [0, 1]; --min-pred-voxels must be at least 1.")
    records = load_records(args)
    if not args.checkpoint.is_file():
        parser.error("Checkpoint not found. Download the pretrained weight from GitHub Releases; see weights/README.md.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Install a compatible PyTorch build or choose --device cpu.")
    if args.device != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = config_from_checkpoint(ckpt, args)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_inference_transform()
    with tempfile.TemporaryDirectory(prefix="censyncmb_zero_fill_") as temp_name:
        temp_dir = Path(temp_name)
        outputs = [predict_one(record, transform, model, config, device, args, temp_dir) for record in records]

    summary = {
        "model": "CenSynCMB",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "roi_size": list(config.roi_size),
        "sw_overlap": config.sw_overlap,
        "n_subjects": len(outputs),
        "outputs": outputs,
    }
    (args.output_dir / "censyncmb_inference_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
