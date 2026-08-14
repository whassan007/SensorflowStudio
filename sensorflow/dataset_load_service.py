"""Load configured datasets into separate pipeline sequences for further processing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sensorflow.adapters.a2d2_adapter import A2D2Adapter
from sensorflow.adapters.alpamayo_adapter import AlpamayoAdapter, DEFAULT_ALPAMAYO_SAMPLES
from sensorflow.adapters.local_adapter import DEFAULT_MAX_FRAMES, LocalSequenceAdapter
from sensorflow.adapters.vendor_media import media_available, resolve_vendor_root
from sensorflow.adapters.waymo_adapter import WaymoAdapter
from sensorflow.dataset_fusion_engine import DatasetFusionEngine

ALL_VENDORS = ("local", "alpamayo", "waymo", "a2d2")


def _missing_hint(vendor: str, tried: Optional[str]) -> str:
    if vendor == "local":
        return (
            f"Local path missing or empty ({tried or 'data'}). "
            "Set Dataset Configuration → Images Path to a folder of frames or a video, "
            "then re-run Load all datasets."
        )
    return (
        f"{vendor} lake not found"
        + (f" at {tried}" if tried else "")
        + f". Set {vendor}_root / source_path to the dataset folder, or enable allow_stub "
        "for a small demo sequence (not a full AV lake)."
    )


class DatasetLoadService:
    """Register each vendor as its own homogeneous UnifiedSequence under runs/pipeline/."""

    def __init__(self, engine: Optional[DatasetFusionEngine] = None):
        self.engine = engine or DatasetFusionEngine()
        self.local = LocalSequenceAdapter()
        self.alpamayo = AlpamayoAdapter()
        self.waymo = WaymoAdapter()
        self.a2d2 = A2D2Adapter()

    def load_all(
        self,
        *,
        sequence_prefix: str = "seq_001",
        vendors: Optional[List[str]] = None,
        source_path: Optional[str] = None,
        waymo_root: Optional[str] = None,
        alpamayo_root: Optional[str] = None,
        a2d2_root: Optional[str] = None,
        allow_stub: bool = True,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        vendors_norm = [v.lower().strip() for v in (vendors or list(ALL_VENDORS)) if v]
        if not vendors_norm:
            vendors_norm = list(ALL_VENDORS)

        roots = {
            "local": source_path or "data",
            # AV lakes use dedicated roots only — do not silently reuse Local Images Path
            # as Waymo/Alpamayo/A2D2 (that would fake three vendors from one folder).
            "waymo": waymo_root,
            "alpamayo": alpamayo_root,
            "a2d2": a2d2_root,
        }

        results: List[Dict[str, Any]] = []
        for vendor in vendors_norm:
            seq_id = f"{sequence_prefix}_{vendor}"
            entry = self._load_one(
                vendor=vendor,
                sequence_id=seq_id,
                root_hint=roots.get(vendor),
                allow_stub=allow_stub,
                max_frames=max_frames,
            )
            results.append(entry)

        loaded = [r for r in results if r.get("status") == "ok"]
        not_executed = [r for r in results if r.get("status") == "NOT_EXECUTED"]
        stubs = [r for r in loaded if r.get("demo_stub")]

        ledger_path = Path("runs/pipeline") / f"{sequence_prefix}_load_ledger.json"
        ledger = {
            "sequence_prefix": sequence_prefix,
            "results": results,
            "loaded": len(loaded),
            "not_executed": len(not_executed),
            "stub_count": len(stubs),
            "all_real": bool(loaded) and not stubs and not not_executed,
        }
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=2))

        # Point active sequence at first successfully loaded vendor for downstream stages.
        active = loaded[0]["sequence_id"] if loaded else f"{sequence_prefix}_alpamayo"
        state_path = Path("runs/pipeline/state.json")
        state = {}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except Exception:
                state = {}
        state["_active_sequence"] = active
        state["_load_ledger"] = str(ledger_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))

        overall = "ok" if loaded else "NOT_EXECUTED"
        return {
            "status": overall,
            "sequence_prefix": sequence_prefix,
            "active_sequence_id": active if loaded else None,
            "ledger_path": str(ledger_path),
            "results": results,
            "loaded": len(loaded),
            "not_executed": len(not_executed),
            "stub_count": len(stubs),
            "message": self._summary_message(loaded, not_executed, stubs),
        }

    def _load_one(
        self,
        *,
        vendor: str,
        sequence_id: str,
        root_hint: Optional[str],
        allow_stub: bool,
        max_frames: Optional[int],
    ) -> Dict[str, Any]:
        try:
            if vendor == "local":
                return self._load_local(sequence_id, root_hint, max_frames)

            root = resolve_vendor_root(
                {"source_path": root_hint, "root": root_hint},
                "source_path",
                "root",
            )
            if root is not None and media_available(root):
                source = {
                    "source_path": str(root),
                    "demo_stub": False,
                    "max_frames": DEFAULT_MAX_FRAMES if max_frames is None else max_frames,
                }
                seq = self._adapter(vendor).load(source, sequence_id)
            elif allow_stub:
                seq = self._adapter(vendor).load(self._stub_source(vendor), sequence_id)
            else:
                return {
                    "status": "NOT_EXECUTED",
                    "vendor": vendor,
                    "sequence_id": sequence_id,
                    "frames": 0,
                    "demo_stub": False,
                    "reason": "missing_dataset_root",
                    "message": _missing_hint(vendor, root_hint),
                }

            seq = self.engine._stratify(seq)
            seq.sequence_id = sequence_id
            manifest = self.engine.save_manifest(seq)
            demo_stub = bool(seq.taxonomy_manifest.get("demo_stub"))
            return {
                "status": "ok",
                "vendor": seq.vendor,
                "sequence_id": sequence_id,
                "frames": len(seq.frames),
                "demo_stub": demo_stub,
                "manifest": str(manifest),
                "message": (
                    f"demo stub: {len(seq.frames)} frames (not a full {vendor} lake)"
                    if demo_stub
                    else f"loaded {len(seq.frames)} frames from {root_hint or 'configured root'}"
                ),
            }
        except FileNotFoundError as exc:
            return {
                "status": "NOT_EXECUTED",
                "vendor": vendor,
                "sequence_id": sequence_id,
                "frames": 0,
                "demo_stub": False,
                "reason": "missing_data",
                "message": str(exc),
            }
        except Exception as exc:
            return {
                "status": "NOT_EXECUTED",
                "vendor": vendor,
                "sequence_id": sequence_id,
                "frames": 0,
                "demo_stub": False,
                "reason": "error",
                "message": f"{vendor} load failed: {exc}",
            }

    def _load_local(
        self,
        sequence_id: str,
        root_hint: Optional[str],
        max_frames: Optional[int],
    ) -> Dict[str, Any]:
        local_src = {
            "source_path": root_hint or "data",
            "max_frames": DEFAULT_MAX_FRAMES if max_frames is None else max_frames,
        }
        seq = self.local.load(local_src, sequence_id)
        seq = self.engine._stratify(seq)
        seq.sequence_id = sequence_id
        manifest = self.engine.save_manifest(seq)
        return {
            "status": "ok",
            "vendor": "local",
            "sequence_id": sequence_id,
            "frames": len(seq.frames),
            "demo_stub": False,
            "manifest": str(manifest),
            "message": f"loaded {len(seq.frames)} local frames from {local_src['source_path']}",
        }

    def _adapter(self, vendor: str):
        return {
            "alpamayo": self.alpamayo,
            "waymo": self.waymo,
            "a2d2": self.a2d2,
        }[vendor]

    @staticmethod
    def _stub_source(vendor: str) -> Dict[str, Any]:
        if vendor == "alpamayo":
            src = dict(DEFAULT_ALPAMAYO_SAMPLES["physical_ai"])
            src["demo_stub"] = True
            return src
        return {"demo_stub": True}

    @staticmethod
    def _summary_message(loaded, not_executed, stubs) -> str:
        parts = [f"{len(loaded)} dataset(s) loaded into pipeline runs"]
        if stubs:
            parts.append(f"{len(stubs)} demo stub(s) — not full AV lakes")
        if not_executed:
            parts.append(f"{len(not_executed)} NOT_EXECUTED (missing path or empty)")
        return "; ".join(parts)
