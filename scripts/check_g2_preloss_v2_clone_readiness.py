#!/usr/bin/env python3
"""Fail-closed repository check for the clone-reproducible G2 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/repro/g2_recurrent_visual_preloss_v2_clone_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-tracked",
        action="store_true",
        help="also require every runtime file to exist in the current Git index",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recipe_path = ROOT / manifest["recipe"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked: set[Path] = {MANIFEST, recipe_path}

    expected_recipe = {
        "recurrent_profile": "legacy_256x16",
        "camera_encoder_profile": "cnn5_160",
        "camera_encoder_channels": [32, 64, 96, 128, 160],
        "convolution_layers_per_encoder": 5,
        "convolution_layers_total": 20,
        "hidden_dim": 256,
        "sequence_length": 16,
        "burn_in_steps": 4,
        "sequence_stride": 12,
        "gradient_clip": 5.0,
        "batch_size": 64,
        "random_shift_pad_pixels": 2,
    }
    for key, expected in expected_recipe.items():
        if recipe.get("model", {}).get(key) != expected:
            failures.append(f"recipe.model.{key} differs")
    for key in ("cross_camera_contrastive_weight", "temporal_pose_residual_weight"):
        if recipe.get("loss", {}).get(key) != 0.0:
            failures.append(f"recipe.loss.{key} must be zero")
    initialization = recipe.get("initialization", {})
    if not all(initialization.get(key) is True for key in (
        "fresh_actor", "fresh_critics", "fresh_replay"
    )) or initialization.get("resume_checkpoint") is not False:
        failures.append("recipe is not a fully fresh initialization")

    for relative, expected_hash in manifest["required_files_sha256"].items():
        path = ROOT / relative
        checked.add(path)
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256(path) != expected_hash:
            failures.append(f"sha256:{relative}")
    for relative in manifest["required_runtime_files"]:
        path = ROOT / relative
        checked.add(path)
        if not path.is_file():
            failures.append(f"missing:{relative}")

    usd_root = ROOT / "source/geniesim/assets/robot/G2_omnipicker"
    robot_layer = (usd_root / "robot_fix.usda").read_text(encoding="utf-8")
    for relative in (
        "configuration/Cam_L.usd",
        "configuration/Cam_R.usd",
        "configuration/robot_base.usd",
        "configuration/robot_physics.usd",
        "configuration/robot_sensor.usd",
        "configuration/robot_robot.usd",
    ):
        if f"@./{relative}@" not in robot_layer or not (usd_root / relative).is_file():
            failures.append(f"usd-reference:{relative}")

    urdf = ROOT / "source/geniesim/assets/robot/curobo_robot/assets/robot/G2/G2_omnipicker_fixed_dual.urdf"
    try:
        ET.parse(urdf)
    except (ET.ParseError, OSError) as error:
        failures.append(f"urdf-parse:{error}")

    source_files: list[Path] = []
    for relative_root in manifest["required_source_roots"]:
        root = ROOT / relative_root
        if not root.is_dir():
            failures.append(f"missing-source-root:{relative_root}")
            continue
        source_files.extend(
            path for path in root.rglob("*.py") if "__pycache__" not in path.parts
        )
    checked.update(source_files)
    for path in source_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as error:
            failures.append(f"python-compile:{path.relative_to(ROOT)}:{error}")

    shell = ROOT / manifest["entrypoint"]
    shell_check = subprocess.run(
        ("bash", "-n", str(shell)), capture_output=True, text=True, check=False
    )
    if shell_check.returncode:
        failures.append(f"shell-syntax:{shell_check.stderr.strip()}")

    lfs_check = subprocess.run(
        ("git", "check-attr", "filter", "--", str(
            Path("source/geniesim/assets/robot/G2_omnipicker/configuration/robot_base.usd")
        )), cwd=ROOT, capture_output=True, text=True, check=False
    )
    if not lfs_check.stdout.rstrip().endswith(": lfs"):
        failures.append("git-lfs-attribute-missing")

    untracked: list[str] = []
    if args.require_tracked:
        for path in sorted(checked):
            relative = str(path.relative_to(ROOT))
            result = subprocess.run(
                ("git", "ls-files", "--error-unmatch", "--", relative),
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode:
                untracked.append(relative)
        if untracked:
            failures.append("required-files-not-in-git-index")

    result = {
        "schema": "geniesim_g2_clone_readiness_v1",
        "status": "CLONE_READY" if not failures else "CLONE_NOT_READY",
        "recipe": recipe["name"],
        "checked_file_count": len(checked),
        "asset_hash_count": len(manifest["required_files_sha256"]),
        "git_index_required": args.require_tracked,
        "untracked_required_files": untracked,
        "failures": failures,
        "external_requirements": manifest["external_not_in_git"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
