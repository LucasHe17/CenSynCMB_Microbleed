# Pretrained checkpoint

Weights are distributed separately through [GitHub Releases](https://github.com/LucasHe17/CenSynCMB_Microbleed/releases).

From the repository root:

```bash
python scripts/download_weights.py
```

The script installs `censyncmb_v1.0.0.pth` here after checking its size and SHA-256 against [manifest.json](manifest.json). It also verifies an existing copy, so the same command works after a manual download.

If the v1.0.0 asset has not been published, the script reports that it is unavailable. Repository cloning alone does not include the weight file.

Maintainers can follow [Publishing pretrained weights](../docs/RELEASING.md).
