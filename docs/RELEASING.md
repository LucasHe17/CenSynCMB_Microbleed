# Publishing pretrained weights

The inference checkpoint is distributed as a **GitHub Release asset**. Keep model binaries out of ordinary Git commits so the source repository remains small.

## Files for v1.0.0

Upload these three files together, keeping their names unchanged:

| File | Content |
| --- | --- |
| `censyncmb_v1.0.0.pth` | Model tensors and inference configuration |
| `censyncmb_v1.0.0.sha256` | SHA-256 checksum |
| `censyncmb_v1.0.0.json` | Export metadata |

The prepared checkpoint is **94,599,306 bytes** and has this SHA-256:

```text
95d4d35f36aea631128e787ee18182db9c4b55fc8d22d7cd6c58a983c509c965
```

It contains the same model tensors as the original training checkpoint, with optimizer state and training-machine paths removed. Publish the exported file described here.

## Upload through the browser

1. Open this repository's [Releases page](https://github.com/LucasHe17/CenSynCMB_Microbleed/releases) and select **Draft a new release**.
2. Create the tag **`v1.0.0`**, targeting the tested `main` commit.
3. Use the title **CenSynCMB v1.0.0 — pretrained inference**. Describe the checkpoint, input requirements and inference-only scope. State the weight reuse license explicitly.
4. Attach all three files above in the release-assets area.
5. Select **Publish release** when the notes and assets are complete.
6. From a fresh clone, run `python scripts/download_weights.py`. A successful download must pass the recorded size and SHA-256 checks.

A saved draft is only visible to repository collaborators; public download links work after publication. The downloader expects this exact URL:

```text
https://github.com/LucasHe17/CenSynCMB_Microbleed/releases/download/v1.0.0/censyncmb_v1.0.0.pth
```

See [GitHub's release instructions](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) and [guidance on distributing large binaries](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Upload with GitHub CLI

Authenticate with `gh auth login`, then run the following from the directory containing the three assets and your completed `release-notes.md`:

```bash
gh release create v1.0.0 \
  censyncmb_v1.0.0.pth \
  censyncmb_v1.0.0.sha256 \
  censyncmb_v1.0.0.json \
  --repo LucasHe17/CenSynCMB_Microbleed \
  --target main \
  --title "CenSynCMB v1.0.0 — pretrained inference" \
  --notes-file release-notes.md \
  --draft
```

Review the draft, then publish it:

```bash
gh release edit v1.0.0 \
  --repo LucasHe17/CenSynCMB_Microbleed \
  --draft=false
```

## Export a future checkpoint

Install the repository dependencies, then:

```bash
python scripts/export_checkpoint.py \
  --input /path/to/training_checkpoint.pth \
  --output release-assets/censyncmb_v1.1.0.pth
```

The exporter loads with `weights_only=True`, verifies the model's state dictionary with strict loading, keeps model tensors and inference settings, and checks every tensor after saving. It writes a checkpoint, checksum and JSON metadata.

For a future version, update `weights/manifest.json`, the default checkpoint path, documentation and release tag together. Serialization can produce different file bytes even when tensors are equal, so always use the checksum of the exact exported file. Use a new version for changed weights and keep published versions reproducible.
