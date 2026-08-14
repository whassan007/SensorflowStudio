"""Real discovery / validation / health helpers for execution integrity."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_EXTS = {".txt", ".json", ".xml"}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def discover_images(
    source_path: str,
    *,
    max_scan: int = 100_000,
    dataset_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover driving-relevant images under a path; report corrupt/unreadable files."""
    from sensorflow.av_media_filter import partition_media_files

    root = _resolve(source_path)
    result: Dict[str, Any] = {
        "source_path": str(root),
        "exists": root.exists(),
        "is_dir": root.is_dir() if root.exists() else False,
        "images_discovered": 0,
        "images_readable": 0,
        "images_corrupt": 0,
        "images_excluded": 0,
        "excluded_by_reason": {},
        "excluded_samples": [],
        "annotations_discovered": 0,
        "annotation_objects": 0,
        "class_ids": [],
        "class_id_out_of_range": [],
        "errors": [],
        "warnings": [],
        "sample_images": [],
        "output_manifest": None,
    }
    if not root.exists():
        result["errors"].append(f"Path does not exist: {source_path}")
        return result

    raw_images: List[Path] = []
    labels: List[Path] = []
    if root.is_file():
        if root.suffix.lower() in IMAGE_EXTS:
            raw_images = [root]
        elif root.suffix.lower() in LABEL_EXTS:
            labels = [root]
        else:
            result["errors"].append(f"Not an image or label file: {source_path}")
            return result
    else:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf in IMAGE_EXTS:
                raw_images.append(p)
            elif suf in LABEL_EXTS:
                labels.append(p)
            if len(raw_images) + len(labels) >= max_scan:
                result["warnings"].append(f"Scan capped at {max_scan} files")
                break

    scan_root = root if root.is_dir() else root.parent
    images, excluded = partition_media_files(
        raw_images,
        root=scan_root,
        dataset_type=dataset_type,
        allow_videos=False,
    )
    result["images_excluded"] = len(excluded)
    result["excluded_by_reason"] = {
        reason: sum(1 for e in excluded if e.get("reason") == reason)
        for reason in sorted({e.get("reason", "excluded_path") for e in excluded})
    }
    result["excluded_samples"] = excluded[:50]
    if excluded:
        result["warnings"].append(
            f"{len(excluded)} file(s) excluded as non-driving media "
            f"({', '.join(f'{k}: {v}' for k, v in result['excluded_by_reason'].items())})"
        )
    labels = sorted(labels)
    result["images_discovered"] = len(images)
    result["annotations_discovered"] = len(labels)

    readable = 0
    corrupt = 0
    for img in images:
        try:
            size = img.stat().st_size
            if size <= 0:
                corrupt += 1
                result["errors"].append(f"Empty image: {img}")
                continue
            # Light decode probe without requiring PIL
            with open(img, "rb") as f:
                header = f.read(16)
            if not header:
                corrupt += 1
                result["errors"].append(f"Unreadable image: {img}")
                continue
            # JPEG/PNG magic checks when applicable
            suf = img.suffix.lower()
            if suf in {".jpg", ".jpeg"} and not (header.startswith(b"\xff\xd8")):
                corrupt += 1
                result["errors"].append(f"Corrupt JPEG header: {img}")
                continue
            if suf == ".png" and not header.startswith(b"\x89PNG"):
                corrupt += 1
                result["errors"].append(f"Corrupt PNG header: {img}")
                continue
            readable += 1
            if len(result["sample_images"]) < 12:
                result["sample_images"].append(str(img))
        except OSError as e:
            corrupt += 1
            result["errors"].append(f"OS error reading {img}: {e}")

    result["images_readable"] = readable
    result["images_corrupt"] = corrupt

    class_ids: List[int] = []
    objects = 0
    for lab in labels:
        try:
            text = lab.read_text(errors="replace")
        except OSError as e:
            result["warnings"].append(f"Could not read label {lab}: {e}")
            continue
        if lab.suffix.lower() == ".txt":
            for line in text.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                    class_ids.append(cid)
                    objects += 1
                except ValueError:
                    result["warnings"].append(f"Bad label line in {lab}: {line[:40]}")
        elif lab.suffix.lower() == ".json":
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    objects += len(data)
                elif isinstance(data, dict):
                    anns = data.get("annotations") or data.get("objects") or data.get("labels") or []
                    if isinstance(anns, list):
                        objects += len(anns)
                        for a in anns:
                            if isinstance(a, dict) and "category_id" in a:
                                class_ids.append(int(a["category_id"]))
                            elif isinstance(a, dict) and "class_id" in a:
                                class_ids.append(int(a["class_id"]))
            except json.JSONDecodeError:
                result["warnings"].append(f"Invalid JSON label: {lab}")

    result["annotation_objects"] = objects
    result["class_ids"] = sorted(set(class_ids))
    return result


def write_load_manifest(discovery: Dict[str, Any], dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "load_manifest.json"
    payload = {
        "source_path": discovery.get("source_path"),
        "images_discovered": discovery.get("images_discovered"),
        "images_readable": discovery.get("images_readable"),
        "images_corrupt": discovery.get("images_corrupt"),
        "images_excluded": discovery.get("images_excluded"),
        "excluded_by_reason": discovery.get("excluded_by_reason"),
        "excluded_samples": discovery.get("excluded_samples"),
        "annotations_discovered": discovery.get("annotations_discovered"),
        "annotation_objects": discovery.get("annotation_objects"),
        "class_ids": discovery.get("class_ids"),
        "sample_images": discovery.get("sample_images"),
    }
    path.write_text(json.dumps(payload, indent=2))
    discovery["output_manifest"] = str(path)
    return path


def load_and_preprocess(
    source_path: str,
    *,
    yaml_path: Optional[str] = None,
    dataset_type: str = "local",
) -> Dict[str, Any]:
    """Real Load & Preprocess: discovery + optional YAML semantic checks."""
    discovery = discover_images(source_path, dataset_type=dataset_type)
    yaml_report = None
    if yaml_path:
        ypath = Path(yaml_path)
        if not ypath.is_absolute():
            ypath = Path.cwd() / ypath
        if ypath.exists():
            yaml_report = validate_yaml_semantics(yaml_path)
        else:
            # Missing YAML during load is a warning, not a load failure.
            # Use POST /api/yaml/validate for explicit VALIDATION_FAILED.
            yaml_report = {
                "yaml_path": str(ypath),
                "exists": False,
                "status": "NOT_EXECUTED",
                "errors": [],
                "warnings": [f"YAML not found (skipped semantic validation): {yaml_path}"],
            }
            discovery.setdefault("warnings", []).append(yaml_report["warnings"][0])

    status = "SUCCEEDED"
    if not discovery["exists"] or discovery["images_discovered"] == 0:
        status = "FAILED"
    elif discovery["images_corrupt"] > 0 and discovery["images_readable"] > 0:
        status = "PARTIAL_SUCCESS"
    elif discovery["images_corrupt"] > 0 and discovery["images_readable"] == 0:
        status = "FAILED"

    # Existing YAML with semantic errors elevates SUCCEEDED → VALIDATION_FAILED only.
    if yaml_report and yaml_report.get("status") == "VALIDATION_FAILED":
        if status == "SUCCEEDED":
            status = "VALIDATION_FAILED"
        else:
            discovery.setdefault("warnings", []).extend(yaml_report.get("errors") or [])
            discovery.setdefault("warnings", []).append(
                "YAML validation failed (reported as warning because load status already non-success)"
            )

    out_dir = Path("runs/dataset_loads")
    manifest = write_load_manifest(discovery, out_dir)

    loaded_pct = None
    if discovery["images_discovered"]:
        loaded_pct = round(100.0 * discovery["images_readable"] / discovery["images_discovered"], 1)

    return {
        "status": status,
        "dataset_type": dataset_type,
        "discovery": discovery,
        "yaml_validation": yaml_report,
        "manifest_path": str(manifest),
        "metrics": {
            "images_discovered": discovery["images_discovered"],
            "images_readable": discovery["images_readable"],
            "images_corrupt": discovery["images_corrupt"],
            "images_excluded": discovery.get("images_excluded", 0),
            "excluded_by_reason": discovery.get("excluded_by_reason", {}),
            "annotations_discovered": discovery["annotations_discovered"],
            "annotation_objects": discovery["annotation_objects"],
            "loaded_pct_of_discovered": loaded_pct,
            "browsable": discovery["images_readable"] > 0,
        },
        "reconciliation": reconcile_counts(discovery, yaml_report),
    }


def validate_yaml_semantics(yaml_path: str) -> Dict[str, Any]:
    """Resolve YAML paths, verify dirs, count images/labels, class ID range."""
    path = _resolve(yaml_path)
    report: Dict[str, Any] = {
        "yaml_path": str(path),
        "exists": path.exists(),
        "status": "SUCCEEDED",
        "errors": [],
        "warnings": [],
        "paths": {},
        "class_count": 0,
        "image_counts": {},
        "label_counts": {},
        "class_id_out_of_range": [],
        "content_hash": None,
        "used_by_execution": False,
        "last_execution_id": None,
    }
    if not path.exists():
        report["status"] = "VALIDATION_FAILED"
        report["errors"].append(f"YAML not found: {yaml_path}")
        return report

    text = path.read_text(errors="replace")
    report["content_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    report["last_modified"] = path.stat().st_mtime

    # Minimal YAML parse without requiring PyYAML for names/path/train/val
    data: Dict[str, Any] = {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except Exception:
        # Fallback: regex-ish key extraction
        for line in text.splitlines():
            if ":" not in line or line.strip().startswith("#"):
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip("'\"")
            if key in ("path", "train", "val", "test"):
                data[key] = val

    root = Path(data.get("path") or ".")
    if not root.is_absolute():
        root = path.parent / root
    report["paths"]["dataset_root"] = str(root)
    if not root.exists():
        report["errors"].append(f"Dataset root does not exist: {root}")
        report["status"] = "VALIDATION_FAILED"

    names = data.get("names") or {}
    if isinstance(names, dict):
        report["class_count"] = len(names)
        max_id = max((int(k) for k in names.keys()), default=-1)
    elif isinstance(names, list):
        report["class_count"] = len(names)
        max_id = len(names) - 1
    else:
        max_id = -1
        report["warnings"].append("No class names found in YAML")

    for split in ("train", "val", "test"):
        rel = data.get(split)
        if not rel:
            continue
        split_path = Path(rel)
        if not split_path.is_absolute():
            split_path = root / split_path
        report["paths"][split] = str(split_path)
        if not split_path.exists():
            report["errors"].append(f"Split path missing: {split}={split_path}")
            report["status"] = "VALIDATION_FAILED"
            report["image_counts"][split] = 0
            continue
        imgs = [
            p
            for p in (split_path.rglob("*") if split_path.is_dir() else [split_path])
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        report["image_counts"][split] = len(imgs)
        # YOLO labels sibling
        label_dir = Path(str(split_path).replace("/images", "/labels").replace("\\images", "\\labels"))
        if label_dir.exists():
            labs = [p for p in label_dir.rglob("*.txt") if p.is_file()]
            report["label_counts"][split] = len(labs)
            out_of_range = []
            for lab in labs[:5000]:
                for line in lab.read_text(errors="replace").splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except ValueError:
                        continue
                    if max_id >= 0 and (cid < 0 or cid > max_id):
                        out_of_range.append({"file": str(lab), "class_id": cid})
            if out_of_range:
                report["class_id_out_of_range"].extend(out_of_range[:50])
                report["errors"].append(
                    f"{len(out_of_range)} class id(s) out of range 0..{max_id} in {split}"
                )
                report["status"] = "VALIDATION_FAILED"

    return report


def verify_script(script_name: str) -> Dict[str, Any]:
    """Exists + syntax + importable deps probe + dry-run capability flag."""
    path = Path(script_name)
    report: Dict[str, Any] = {
        "script": script_name,
        "exists": path.exists(),
        "syntax_valid": False,
        "deps": {},
        "dry_run_ok": False,
        "last_execution_id": None,
        "last_status": None,
        "message": "",
    }
    if not path.exists():
        report["message"] = f"Not found: {script_name}"
        return report
    try:
        src = path.read_text(errors="replace")
        ast.parse(src)
        report["syntax_valid"] = True
    except SyntaxError as e:
        report["message"] = f"Syntax error: {e}"
        return report

    for mod in ("ultralytics", "torch", "yaml", "numpy"):
        report["deps"][mod] = importlib.util.find_spec(mod) is not None

    # Dry-run: --help
    try:
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        report["dry_run_ok"] = proc.returncode == 0
        report["exit_code"] = proc.returncode
        if proc.returncode != 0:
            report["message"] = (proc.stderr or proc.stdout or "")[:500]
        else:
            report["message"] = "exists, syntax valid, --help ok"
    except Exception as e:
        report["message"] = f"dry-run failed: {e}"

    # Attach last ledger execution for this script operation
    try:
        from sensorflow.execution_ledger import list_executions

        op_map = {
            "train.py": "training",
            "infer.py": "inference",
            "autograder.py": "auto_grader",
        }
        op = op_map.get(path.name)
        if op:
            recent = list_executions(operation=op, limit=1)
            if recent:
                report["last_execution_id"] = recent[0].get("execution_id")
                report["last_status"] = recent[0].get("status")
    except Exception:
        pass
    return report


def backend_health() -> Dict[str, Any]:
    """Meaningful health: API up, FS writable, Python env, YOLO/torch, optional GPU."""
    runs = Path("runs")
    writable = False
    try:
        runs.mkdir(parents=True, exist_ok=True)
        probe = runs / ".health_write_probe"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        writable = True
    except Exception as e:
        writable = False
        fs_error = str(e)
    else:
        fs_error = None

    torch_ok = importlib.util.find_spec("torch") is not None
    yolo_ok = importlib.util.find_spec("ultralytics") is not None
    gpu = {"available": False, "name": None}
    if torch_ok:
        try:
            import torch

            gpu["available"] = bool(torch.cuda.is_available())
            if gpu["available"]:
                gpu["name"] = torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu["available"] = True
                gpu["name"] = "mps"
        except Exception:
            pass

    last_ok = None
    try:
        from sensorflow.execution_ledger import list_executions

        for e in list_executions(limit=50):
            if e.get("status") == "SUCCEEDED" and e.get("verified"):
                last_ok = e.get("execution_id")
                break
            if e.get("status") == "SUCCEEDED" and last_ok is None:
                last_ok = e.get("execution_id")
    except Exception:
        pass

    scripts = {
        "train.py": verify_script("train.py"),
        "infer.py": verify_script("infer.py"),
        "autograder.py": verify_script("autograder.py"),
    }

    return {
        "status": "ok" if writable else "degraded",
        "api": True,
        "filesystem_writable": writable,
        "filesystem_error": fs_error,
        "python": sys.version.split()[0],
        "torch_installed": torch_ok,
        "ultralytics_installed": yolo_ok,
        "gpu": gpu,
        "last_successful_execution_id": last_ok,
        "scripts": scripts,
    }


def reconcile_counts(
    discovery: Dict[str, Any],
    yaml_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare source discovery vs YAML split counts."""
    source_images = discovery.get("images_discovered") or 0
    yaml_images = 0
    if yaml_report:
        yaml_images = sum((yaml_report.get("image_counts") or {}).values())
    consistent = True
    notes = []
    if yaml_report and yaml_images and source_images and yaml_images != source_images:
        consistent = False
        notes.append(
            f"Source images ({source_images}) != YAML split sum ({yaml_images})"
        )
    if discovery.get("images_corrupt"):
        consistent = False
        notes.append(f"{discovery['images_corrupt']} corrupt image(s)")
    return {
        "state": "CONSISTENT" if consistent else "INCONSISTENT",
        "source_images": source_images,
        "yaml_images": yaml_images,
        "notes": notes,
    }


def artifact_info(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    info: Dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if p.exists() and p.is_file():
        info["size"] = p.stat().st_size
        info["sha256"] = None
        try:
            from sensorflow.execution_ledger import sha256_file

            info["sha256"] = sha256_file(p)
        except Exception:
            pass
    return info
