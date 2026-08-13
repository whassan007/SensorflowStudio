"""Masklet-style temporal propagation with BEV association.

Extends the platform's :class:`sensorflow.temporal_tracker.TemporalTracker`
(Kalman + Hungarian) with the two ideas this feature needs, mapping to SAM 3's
masklet propagation:

- association happens in the stable metric BEV plane with a richer cost:
  BEV center distance + class-mismatch penalty + size-difference penalty
  (fused boxes have reliable LiDAR geometry, so size/class gating is safe);
- masklet propagation: a track whose object briefly disappears (e.g. a 2-3
  frame camera occlusion that LiDAR also partially loses) keeps its identity
  alive via the Kalman-predicted BEV position and *emits propagated boxes*
  for the gap frames (confidence-decayed, flagged ``propagated``), so brief
  dropouts are bridged without spawning new IDs and without fragmenting the
  track — the direct analogue of a masklet being carried through frames where
  the object is not re-detected.

The baseline camera labeler deliberately uses frame-to-frame greedy
association with no motion model and no gap tolerance (see engines.py), which
is what makes ID switches and fragmentation measurable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from sensorflow.schemas.unified_frame import Object3D
from sensorflow.temporal_tracker import TemporalTracker


class BEVMaskletTracker(TemporalTracker):
    """BEV Hungarian association + masklet propagation across dropouts."""

    def __init__(self, max_age: int = 6, distance_gate: float = 4.0,
                 propagate_max: int = 3, min_hits_to_propagate: int = 2,
                 class_penalty: float = 1.5, size_penalty: float = 0.3,
                 dt: float = 0.1,
                 bounds: Tuple[float, float, float, float] = (0.0, 80.0, -25.0, 25.0)):
        super().__init__(max_age=max_age, distance_gate=distance_gate)
        self.dt = dt
        self.propagate_max = propagate_max
        self.min_hits_to_propagate = min_hits_to_propagate
        self.class_penalty = class_penalty
        self.size_penalty = size_penalty
        self.bounds = bounds

    # -- association: BEV distance + class + size (replaces the parent cost)

    def _associate(self, proposals: List[Object3D]) -> List[Tuple[int, int]]:
        from scipy.optimize import linear_sum_assignment

        track_ids = list(self.tracks.keys())
        cost = np.full((len(track_ids), len(proposals)), 1e6)
        for ti, tid in enumerate(track_ids):
            track = self.tracks[tid]
            last_dims = track.history[-1]["bbox_3d"][3:6] if track.history else [4.5, 1.9, 1.6]
            for pi, prop in enumerate(proposals):
                dist = math.hypot(track.state[0] - prop.bbox_3d[0],
                                  track.state[1] - prop.bbox_3d[1])
                if dist >= self.distance_gate:
                    continue
                c = dist
                if track.class_name != prop.class_name:
                    c += self.class_penalty
                c += self.size_penalty * (abs(last_dims[0] - prop.bbox_3d[3])
                                          + abs(last_dims[1] - prop.bbox_3d[4]))
                cost[ti, pi] = c

        row_ind, col_ind = linear_sum_assignment(cost)
        return [(track_ids[r], c) for r, c in zip(row_ind, col_ind)
                if cost[r, c] < 1e6]

    # -- update with masklet propagation

    def update(self, frame_id: str, proposals: List[Object3D]) -> List[Object3D]:
        out = super().update(frame_id, proposals)
        self.last_propagated: set = set()

        # The parent Kalman observes position only, so its velocity estimate
        # converges too slowly to bridge gaps for fast objects. Blend in a
        # finite-difference velocity from the last two *real* measurements
        # (SORT-style), which the constant-velocity prediction then carries
        # through occlusion gaps.
        for track in self.tracks.values():
            if track.missed_frames != 0:
                continue  # not matched this frame
            real_idx = [i for i, h in enumerate(track.history) if not h.get("propagated")]
            if len(real_idx) >= 2:
                i1, i0 = real_idx[-1], real_idx[-2]
                span = max(i1 - i0, 1)  # frames elapsed (propagated entries fill gaps)
                dx = track.history[i1]["bbox_3d"][0] - track.history[i0]["bbox_3d"][0]
                dy = track.history[i1]["bbox_3d"][1] - track.history[i0]["bbox_3d"][1]
                v_meas = np.array([dx, dy]) / (span * self.dt)
                track.state[2:4] = 0.5 * track.state[2:4] + 0.5 * v_meas

        # Track-level class voting: identity persistence lets high-confidence
        # camera semantics (or the modal LiDAR template guess) stabilize the
        # per-frame class of every box the track emits.
        if not hasattr(self, "_class_votes"):
            self._class_votes: Dict[int, Dict[str, float]] = {}
        for prop in out:
            votes = self._class_votes.setdefault(prop.track_id, {})
            votes[prop.class_name] = votes.get(prop.class_name, 0.0) + prop.confidence
            best = max(votes, key=votes.get)
            prop.class_name = best
            if prop.track_id in self.tracks:
                self.tracks[prop.track_id].class_name = best

        for track in self.tracks.values():
            if track.missed_frames == 0 or not track.history:
                continue  # matched (or spawned) this frame
            if not (1 <= track.missed_frames <= self.propagate_max):
                continue
            real_hits = sum(1 for h in track.history if not h.get("propagated"))
            if real_hits < self.min_hits_to_propagate:
                continue
            px, py = float(track.state[0]), float(track.state[1])
            x0, x1, y0, y1 = self.bounds
            if not (x0 <= px < x1 and y0 <= py < y1):
                continue  # object likely left the field of view: do not hallucinate
            last_box = track.history[-1]["bbox_3d"]
            bbox = [px, py, last_box[2], last_box[3], last_box[4], last_box[5], last_box[6]]
            conf = round(track.confidence * (0.8 ** track.missed_frames), 4)
            track.history.append({
                "frame_id": frame_id, "bbox_3d": bbox,
                "confidence": conf, "propagated": True,
            })
            prop = Object3D(bbox_3d=bbox, class_name=track.class_name,
                            confidence=conf, track_id=track.track_id)
            self.last_propagated.add(track.track_id)
            out.append(prop)
        return out


def tracks_to_dicts(per_frame: Dict[str, List[Dict]]) -> List[Dict]:
    """Group per-frame labeled boxes (with track_id) into MOT-shaped tracks."""
    tracks: Dict[object, Dict] = {}
    for frame_id in sorted(per_frame.keys()):
        for box in per_frame[frame_id]:
            tid = box.get("track_id")
            if tid is None:
                continue
            entry = tracks.setdefault(tid, {"track_id": tid,
                                            "class_name": box.get("class_name", "unknown"),
                                            "frames": []})
            entry["frames"].append({"frame_id": frame_id, "bbox_3d": box["bbox_3d"]})
    return list(tracks.values())
