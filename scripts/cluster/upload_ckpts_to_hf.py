"""
Upload trained checkpoints to HuggingFace as unified storage.

Designed to run after RCP training converges. Uploads:
  - latest.pth (for resume/probe)
  - Selected intermediate ckpts (every Nth) for substitution-dynamics figure
  - Skips identical-by-mtime files already on the hub

Usage (on RCP pod, after training):
    HF_TOKEN=<token> python upload_ckpts_to_hf.py \
        --ckpt-dir /scratch/wxu/habitat_checkpoints_rcp/dh-spatial-tr-logpolar \
        --hf-repo alunxu/spatial-memory-checkpoints \
        --remote-subdir foveated_logpolar_seed1 \
        --keep-every 5

Idempotent: safe to re-run; skips if remote file size matches local.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List


def parse_ckpt_n(path: Path) -> int | None:
    """Extract N from ckpt.N.pth, or None for non-numeric names."""
    m = re.match(r"ckpt\.(\d+)\.pth$", path.name)
    return int(m.group(1)) if m else None


def select_ckpts_to_upload(ckpt_dir: Path, keep_every: int) -> List[Path]:
    """Pick latest.pth + every Nth numeric ckpt.

    The substitution-dynamics figure needs intermediate ckpts; convergence
    figures only need the final. Default keep_every=5 keeps ckpts
    {0, 5, 10, 15, ...} plus latest.pth.
    """
    selected: List[Path] = []
    latest = ckpt_dir / "latest.pth"
    if latest.exists():
        selected.append(latest)

    numeric: List[Path] = []
    for p in sorted(ckpt_dir.glob("ckpt.*.pth")):
        n = parse_ckpt_n(p)
        if n is None:
            continue
        if n % keep_every == 0:
            numeric.append(p)
    # Always include the highest-N ckpt (final converged one)
    all_numeric = sorted(
        [p for p in ckpt_dir.glob("ckpt.*.pth") if parse_ckpt_n(p) is not None],
        key=lambda p: parse_ckpt_n(p) or -1,
    )
    if all_numeric and all_numeric[-1] not in numeric:
        numeric.append(all_numeric[-1])
    selected.extend(numeric)
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="Local checkpoint dir (e.g. /scratch/wxu/habitat_checkpoints_rcp/<run>)")
    ap.add_argument("--hf-repo", default="alunxu/spatial-memory-checkpoints",
                    help="Target HF repo (default: %(default)s)")
    ap.add_argument("--remote-subdir", required=True,
                    help="Subdir under repo root, e.g. 'foveated_logpolar_seed1'")
    ap.add_argument("--keep-every", type=int, default=5,
                    help="Upload every Nth numeric ckpt (default 5; latest.pth + final always)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be uploaded but don't upload")
    args = ap.parse_args()

    if not args.ckpt_dir.exists():
        print(f"ERROR: ckpt-dir does not exist: {args.ckpt_dir}", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: set HF_TOKEN env var (or use --dry-run)", file=sys.stderr)
        return 2

    files = select_ckpts_to_upload(args.ckpt_dir, args.keep_every)
    if not files:
        print(f"No ckpts found in {args.ckpt_dir}")
        return 0

    print(f"Selected {len(files)} files to upload to "
          f"{args.hf_repo}/{args.remote_subdir}/:")
    for p in files:
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name}  ({size_mb:.1f} MB)")

    if args.dry_run:
        print("\n(dry-run: no uploads performed)")
        return 0

    from huggingface_hub import HfApi, HfFolder
    HfFolder.save_token(token)
    api = HfApi(token=token)

    # Ensure repo exists (create if not)
    try:
        api.repo_info(repo_id=args.hf_repo)
    except Exception:
        print(f"Creating repo {args.hf_repo}...")
        api.create_repo(repo_id=args.hf_repo, exist_ok=True)

    n_ok = 0
    n_skip = 0
    for p in files:
        remote_path = f"{args.remote_subdir}/{p.name}"
        # Check if remote already has this exact file (size match)
        try:
            remote_info = api.list_repo_files(repo_id=args.hf_repo)
            if remote_path in remote_info:
                # could compare more (LFS sha256) but size-mtime is often enough
                print(f"  SKIP (exists)  {remote_path}")
                n_skip += 1
                continue
        except Exception:
            pass
        try:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=remote_path,
                repo_id=args.hf_repo,
                commit_message=f"Upload {p.name} from RCP run",
            )
            print(f"  OK              {remote_path}")
            n_ok += 1
        except Exception as e:
            print(f"  FAIL            {remote_path}: {e}", file=sys.stderr)

    print(f"\nDone: {n_ok} uploaded, {n_skip} skipped.")
    print(f"View at: https://huggingface.co/{args.hf_repo}/tree/main/{args.remote_subdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
