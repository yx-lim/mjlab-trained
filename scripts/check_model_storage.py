"""Check the staged model catalog without downloading Git LFS objects."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import PurePosixPath

POINTER = re.compile(
  rb"version https://git-lfs.github.com/spec/v1\n"
  rb"oid sha256:([a-f0-9]{64})\nsize ([0-9]+)\n?"
)
MAX_MODEL_BYTES = 1024**3


def validate(files: dict[str, bytes], manifest: dict) -> list[str]:
  """Reject generated output, unlisted objects, and catalog drift."""
  errors: list[str] = []
  catalog: dict[str, dict] = {}
  for artifact in manifest["artifacts"]:
    path = artifact["path"]
    parts = PurePosixPath(path).parts
    if not parts or parts[0] != "models" or ".." in parts:
      errors.append(f"Invalid model path: {path}")
    if path in catalog:
      errors.append(f"Duplicate model: {path}")
    catalog[path] = artifact

  total = 0
  for path, content in files.items():
    parts = PurePosixPath(path).parts
    if path.startswith("logs/rsl_rl/") or any(
      part.startswith("events.out.tfevents.") for part in parts
    ):
      errors.append(f"Generated training output must stay outside Git: {path}")
    pointer = POINTER.fullmatch(content)
    managed = path.startswith("models/") and path.endswith((".pt", ".onnx"))
    if pointer is not None and path not in catalog:
      errors.append(f"LFS object is outside the curated catalog: {path}")
    if managed and path not in catalog:
      errors.append(f"Add the selected model to models/manifest.json: {path}")
    if path not in catalog:
      continue
    if pointer is None:
      errors.append(f"Catalog entry must be stored as a Git LFS pointer: {path}")
      continue
    oid, size = pointer.groups()
    total += int(size)
    expected = catalog[path]
    if oid.decode() != expected["sha256"] or int(size) != expected["size"]:
      errors.append(f"Model checksum or size differs from its catalog entry: {path}")
  for path in catalog.keys() - files.keys():
    errors.append(f"Catalog model is missing from the index: {path}")
  if total > MAX_MODEL_BYTES:
    errors.append("Curated models exceed the 1 GiB storage limit.")
  return errors


def git(*args: str, data: bytes | None = None) -> bytes:
  return subprocess.check_output(["git", *args], input=data)


def read_index() -> dict[str, bytes]:
  """Read small indexed blobs; avoid fetching or reading large binary content."""
  paths: dict[str, str] = {}
  for item in git("ls-files", "--stage", "-z").split(b"\0"):
    if not item:
      continue
    info, path = item.split(b"\t", 1)
    _, sha, stage = info.split()
    if stage != b"0":
      raise ValueError("Resolve merge conflicts before checking model storage.")
    paths[path.decode()] = sha.decode()
  shas = sorted(set(paths.values()))
  query = ("\n".join(shas) + "\n").encode()
  sizes = git("cat-file", "--batch-check=%(objectname) %(objectsize)", data=query)
  small = [
    line.split()[0] for line in sizes.splitlines() if int(line.split()[1]) < 1024
  ]
  blobs: dict[str, bytes] = {}
  if small:
    raw = git("cat-file", "--batch", data=b"\n".join(small) + b"\n")
    offset = 0
    for sha in small:
      end = raw.index(b"\n", offset)
      size = int(raw[offset:end].split()[2])
      blobs[sha.decode()] = raw[end + 1 : end + 1 + size]
      offset = end + 2 + size
  return {path: blobs.get(sha, b"") for path, sha in paths.items()}


def main() -> int:
  try:
    manifest = json.loads(git("show", ":models/manifest.json"))
    files = read_index()
    errors = validate(files, manifest)
    names = [item["path"] for item in manifest["artifacts"]]
    attributes = git(
      "check-attr", "--cached", "--stdin", "filter", data="\n".join(names).encode()
    ).decode()
    for line in attributes.splitlines():
      if not line.endswith(": filter: lfs"):
        errors.append(f"Model must use the LFS filter: {line}")
  except (KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
    print(f"Cannot validate the staged model catalog: {exc}", file=sys.stderr)
    return 1
  if errors:
    print("\n".join(errors), file=sys.stderr)
    return 1
  total = sum(item["size"] for item in manifest["artifacts"])
  print(f"Validated {len(names)} curated LFS models ({total / 1024**2:.2f} MiB).")
  return 0


if __name__ == "__main__":
  sys.exit(main())
