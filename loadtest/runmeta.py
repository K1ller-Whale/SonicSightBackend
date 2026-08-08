"""Automatic run metadata — a result without this is not citable (§10 rule)."""

import hashlib
import os
import platform
import subprocess
import sys
import time

from . import paths


def _cmd(args, cwd=None):
    try:
        out = subprocess.run(args, capture_output=True, text=True, cwd=cwd,
                             timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _git(repo):
    return {
        "head": _cmd(["git", "rev-parse", "HEAD"], cwd=repo),
        "branch": _cmd(["git", "branch", "--show-current"], cwd=repo),
        "dirty": bool(_cmd(["git", "status", "--porcelain"], cwd=repo)),
    }


def _file_meta(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"bytes": os.path.getsize(path), "sha256_16": h.hexdigest()[:16]}


def collect(scenario, params):
    mobile = os.path.join(paths.WORKSPACE_ROOT, "SonicSightMobile")
    ckpt_dir = os.path.join(paths.SRC_DIR, "ckpt")
    ckpts = {}
    if os.path.isdir(ckpt_dir):
        for name in sorted(os.listdir(ckpt_dir)):
            ckpts[name] = _file_meta(os.path.join(ckpt_dir, name))

    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenario": scenario,
        "params": params,
        "git": {"SonicSightBackend": _git(paths.BACKEND_ROOT),
                "SonicSightMobile": _git(mobile)},
        "checkpoints": ckpts,
        "host": {"platform": platform.platform(),
                 "machine": platform.machine(),
                 "node": platform.node(),
                 "python": sys.version.split()[0]},
        "gpu": _cmd(["nvidia-smi",
                     "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"]),
        "nfr_targets_yaml": _file_meta(paths.NFR_TARGETS_YAML),
    }
    video = params.get("video")
    if video:
        meta["reference_clip"] = {"path": video, **(_file_meta(video) or {})}
    try:
        import torch
        meta["torch"] = {"version": torch.__version__,
                         "cuda": getattr(torch.version, "cuda", None)}
    except Exception:
        meta["torch"] = None
    try:
        import grpc
        meta["grpcio"] = grpc.__version__
    except Exception:
        meta["grpcio"] = None
    return meta
