# Selected trained models

This directory contains the 47 model objects retained from the original
`logs/rsl_rl/` tree: 31 checkpoints and 16 ONNX policies, about 215 MiB in total.
Each run keeps its original experiment name, timestamp, checkpoint filename,
available configuration files, and recorded source changes.

`manifest.json` records every selected model's SHA-256 digest, size, original path,
and the source commit from which it was moved. Moving these existing LFS pointers
does not create new versions of the model data.

## Download and use a model

Install Git LFS, then download only the selected models:

```sh
git lfs install
git lfs pull --include="models/**"
```

Evaluate a checkpoint by passing its new path explicitly, for example:

```sh
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file models/g1_tracking/2026-08-05_15-32-52_getup1/model_3999.pt \
  --motion-file /path/to/the/matching/motion.npz
```

Use the matching task and motion data for the chosen model. Historical evaluation
metadata in `logs/` records the original paths at evaluation time; use
`source_path` in the catalog to map those paths to the new locations.
For training that resumes through the run-directory mechanism, copy the selected
run back into the ignored `logs/rsl_rl/<experiment>/<run>/` output directory first.
Keep new training output outside `models/`.

## Storage policy

| Content | Location |
| --- | --- |
| Source, configurations, small evaluation summaries | Git |
| Selected final/best checkpoints and deployment policies | `models/`, in LFS |
| Intermediate checkpoints and raw TensorBoard events | Ignored training output |
| Historical runs needed for reproducibility | Verified external archive |

New training runs still write to `logs/rsl_rl/`, which is ignored in full. Raw
TensorBoard event files must not be committed. Curated models have a 1 GiB total
limit, checked against their declared LFS object sizes rather than pointer sizes.

To select another model, copy only its final/best checkpoint or deployment policy
and relevant configuration into a run directory here, add it to the catalog with
its SHA-256, size, original path, and source commit, then stage the files and run:

```sh
uv run --no-project scripts/check_model_storage.py
```

The same check runs in the local pre-commit hook and the model-storage workflow.
It reads staged Git pointers, so it does not download model data or use LFS
bandwidth. Changing a selected model requires updating its catalog entry.

## Historical storage cleanup

This layout prevents new generated output from entering Git. Existing historical
LFS objects remain on GitHub until a separate history cleanup and server purge.
Before that cleanup, verify the external archive includes every historical object
and that its size and SHA-256 match the inventory. Keep a full Git-history backup
as well so the original paths and commits can be recovered.

After the unwanted pointers are removed from every relevant branch and tag, ask
GitHub Support about purging those objects while preserving the selected models.
Deleting current files, changing ignore rules, or running local `git lfs prune`
does not reclaim GitHub's remote storage. Keep the account's $0 LFS budget unless
an overage has been explicitly approved.
