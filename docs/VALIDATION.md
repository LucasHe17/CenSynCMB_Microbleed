# Release validation

## Environment

Checks were run on 3 October 2026 using Linux aarch64 and:

| Component | Tested version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.6.0+cu126 |
| MONAI | 1.5.0 |
| NiBabel | 5.4.2 |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |

The test process used CPU execution with one numerical-library thread.

## Checks performed

- Strict state-dictionary loading of the provided Attention U-Net checkpoint: **23,627,747 model parameters**.
- Checkpoint export: every model tensor remained exactly equal after reloading the exported file. Optimizer state and training-machine paths were omitted.
- **14 regression tests passed**, including a real-checkpoint command-line run on a small synthetic 3D image with an anisotropic, non-identity affine.
- Spatial checks covered RAS orientation, 1 mm resampling, world-coordinate preservation in saved NIfTI output and millimetre header units.
- Postprocessing checks covered 26-connected components, removal of small isolated components, lesion centroid coordinates and empty detections.
- Input checks covered missing explicitly supplied modalities, mismatched grids, relative manifest paths, duplicate/unsafe subject IDs and empty manifests.
- Download checks covered successful checksum verification, cached files, corrupt-download cleanup and an unpublished release asset.

The real-checkpoint smoke test uses a 32 × 32 × 32 inference window to keep it lightweight. This is an execution and geometry check; it does not estimate detection accuracy.

## Packaging corrections

The publication package corrects two issues in the supplied inference script:

1. NumPy conversion in the NaN guard discarded MONAI spatial metadata. Tensor-preserving operations now retain the preprocessed affine, and output writing rejects missing metadata.
2. Connected-component filtering used SciPy's default 6-connectivity. It now uses 26-connectivity, matching the manuscript's lesion definition.

The release also adds explicit input validation, CPU/CUDA selection, lesion centroid export and checksum-verified weight downloading. These changes can affect predictions relative to the original inference wrapper.

## Boundaries

No cohort-level performance evaluation, full-volume runtime benchmark or CUDA inference test was performed for this release. The paper's reported metrics were not regenerated. Training, synthesis, dataset splits and the full evaluation pipeline remain outside this repository.

Run the commands in the main README to repeat the included checks. Without `CENSYNCMB_CHECKPOINT`, the real-checkpoint test is skipped.
