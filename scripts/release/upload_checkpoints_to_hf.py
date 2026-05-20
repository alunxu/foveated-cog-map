"""Upload all 5-condition canonical checkpoints to the HF model repo
``alunxu/spatial-memory-checkpoints``.

Run from RCP, where the checkpoints actually live.

Behavior
--------
For each of the 5 paper-canonical conditions, this script:
  1. Resolves the source directory on the RCP scratch volume.
  2. Verifies it contains the expected ckpt.{0..49}.pth + latest.pth.
  3. Uploads the directory to the HF model repo under ``{cond}/`` --
     using HfApi.upload_folder with ``delete_patterns=["{cond}/*"]`` so any
     stale files in that subfolder on the repo are removed before upload.

Then the script uploads the README at ``docs/release/HF_README.md`` to the
top-level of the repo as ``README.md`` (also deletes the previous one).

Why a single Python script (not git LFS)
----------------------------------------
HfApi.upload_folder handles LFS transparently and is resumable; cloning the
existing repo with full LFS pull would be wasteful given we are replacing
all content anyway.

Auth
----
Requires ``HF_TOKEN`` env var set with write access to the repo, OR a
prior ``huggingface-cli login``. If neither is set, the script aborts.

Usage (on RCP)
--------------
::

    cd /path/to/Project
    python scripts/release/upload_checkpoints_to_hf.py            # full upload
    python scripts/release/upload_checkpoints_to_hf.py --dry-run  # list only
    python scripts/release/upload_checkpoints_to_hf.py \
        --blind-dir /scratch/wxu/habitat_checkpoints_rcp/blind_gibson_seed0
        # if the auto-detected blind dir is wrong

If a single condition needs re-pushing only:
::

    python scripts/release/upload_checkpoints_to_hf.py --only coarse
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "alunxu/spatial-memory-checkpoints"
REPO_TYPE = "model"

# Canonical mapping: condition name (paper / repo subfolder) -> RCP source dir.
# 4 sighted conditions are at dh-probe-{1..4} (post-retrain, hp-consistent).
# Blind kept from Phase-2; default guess provided, can be overridden via CLI.
RCP_SCRATCH = "/scratch/wxu/habitat_checkpoints_rcp"

DEFAULT_SOURCES = {
    "blind":              f"{RCP_SCRATCH}/blind_gibson",            # override if needed
    "coarse":             f"{RCP_SCRATCH}/dh-probe-1",
    "foveated":           f"{RCP_SCRATCH}/dh-probe-2",
    "uniform":            f"{RCP_SCRATCH}/dh-probe-3",
    "foveated_logpolar":  f"{RCP_SCRATCH}/dh-probe-4",
}

ALLOW_PATTERNS = ["ckpt.*.pth", "latest.pth"]


def resolve_blind_dir(user_override: str | None) -> str:
    """If the user passed --blind-dir, use it. Otherwise try the default;
    if that does not exist, hint candidates and abort."""
    if user_override:
        return user_override
    default = DEFAULT_SOURCES["blind"]
    if Path(default).is_dir():
        return default
    parent = Path(RCP_SCRATCH)
    candidates = sorted(p.name for p in parent.glob("blind*"))
    msg = (
        f"Default blind dir {default!r} does not exist.\n"
        f"Candidates under {RCP_SCRATCH}:\n  "
        + "\n  ".join(candidates or ["(none found)"])
        + "\nPass --blind-dir <path> to override."
    )
    raise SystemExit(msg)


def verify_source(cond: str, src: Path) -> list[Path]:
    """Return the list of ckpt files we will upload from src for cond."""
    if not src.is_dir():
        raise SystemExit(f"[{cond}] source dir does not exist: {src}")
    files = sorted(src.glob("ckpt.*.pth"))
    latest = src / "latest.pth"
    if latest.exists():
        files.append(latest)
    if not files:
        raise SystemExit(f"[{cond}] no ckpt.*.pth files found in {src}")
    return files


def fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PiB"


def upload_condition(api, cond: str, src: Path, dry_run: bool) -> None:
    files = verify_source(cond, src)
    total = sum(f.stat().st_size for f in files)
    print(f"[{cond}] src={src}")
    print(f"[{cond}]   {len(files)} files, total {fmt_size(total)}")
    print(f"[{cond}]   first: {files[0].name}, last: {files[-1].name}")

    if dry_run:
        return

    api.upload_folder(
        folder_path=str(src),
        path_in_repo=f"{cond}/",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=ALLOW_PATTERNS,
        delete_patterns=[f"{cond}/*"],  # clear any stale files first
        commit_message=f"replace {cond} checkpoints (hp-consistent retrain)",
    )
    print(f"[{cond}]   uploaded.")


def upload_readme(api, repo_root: Path, dry_run: bool) -> None:
    readme_src = repo_root / "docs" / "release" / "HF_README.md"
    if not readme_src.exists():
        raise SystemExit(f"README source not found: {readme_src}")
    print(f"[readme] src={readme_src}, size={fmt_size(readme_src.stat().st_size)}")
    if dry_run:
        return
    api.upload_file(
        path_or_fileobj=str(readme_src),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="replace README with minimal load-instructions version",
    )
    print("[readme] uploaded.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only",
        choices=list(DEFAULT_SOURCES) + ["readme"],
        help="upload a single condition (or just the readme)",
    )
    ap.add_argument("--blind-dir", default=None,
                    help="override RCP source dir for the blind condition")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    # Resolve blind first (early bail if not found)
    sources = dict(DEFAULT_SOURCES)
    sources["blind"] = resolve_blind_dir(args.blind_dir)

    # Auth
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install huggingface_hub  # required for upload")
    api = HfApi(token=token)
    try:
        whoami = api.whoami()
        print(f"HF auth: logged in as {whoami.get('name', '?')}")
    except Exception as e:
        raise SystemExit(
            f"HF auth failed: {e}\n"
            "Set HF_TOKEN env var or run `huggingface-cli login` first."
        )

    if args.only == "readme":
        upload_readme(api, repo_root, args.dry_run)
        return 0

    targets = [args.only] if args.only else list(sources)
    for cond in targets:
        upload_condition(api, cond, Path(sources[cond]), args.dry_run)

    if not args.only:
        upload_readme(api, repo_root, args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
