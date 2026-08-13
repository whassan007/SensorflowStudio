"""Temporal tracking with Kalman filter and Hungarian association."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sensorflow.schemas.unified_frame import Object3D


@dataclass
class KalmanTrack:
    track_id: int
    state: np.ndarray  # [x, y, vx, vy]
    covariance: np.ndarray
    missed_frames: int = 0
    history: List[Dict] = field(default_factory=list)
    class_name: str = "unknown"
    confidence: float = 0.0

    def predict(self, dt: float = 0.1):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = np.eye(4) * 0.1
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, measurement: np.ndarray):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * 0.5
        y = measurement - H @ self.state
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ H) @ self.covariance
        self.missed_frames = 0


class TemporalTracker:
    """Kalman + Hungarian tracker for 3D bounding box sequences."""

    def __init__(
        self,
        max_age: int = 30,
        distance_gate: float = 3.0,
        velocity_penalty: float = 2.0,
    ):
        self.max_age = max_age
        self.distance_gate = distance_gate
        self.velocity_penalty = velocity_penalty
        self.tracks: Dict[int, KalmanTrack] = {}
        self.next_id = 1
        self.id_swap_log: List[Dict] = []

    def update(self, frame_id: str, proposals: List[Object3D]) -> List[Object3D]:
        for track in self.tracks.values():
            track.predict()
            track.missed_frames += 1

        if proposals and self.tracks:
            assignments = self._associate(proposals)
            matched_proposals = set()
            for track_id, prop_idx in assignments:
                prop = proposals[prop_idx]
                track = self.tracks[track_id]
                measurement = np.array([prop.bbox_3d[0], prop.bbox_3d[1]])
                track.update(measurement)
                track.class_name = prop.class_name
                track.confidence = prop.confidence
                track.history.append({
                    "frame_id": frame_id,
                    "bbox_3d": prop.bbox_3d,
                    "confidence": prop.confidence,
                })
                matched_proposals.add(prop_idx)

            for i, prop in enumerate(proposals):
                if i not in matched_proposals:
                    self._spawn_track(frame_id, prop)
        elif proposals:
            for prop in proposals:
                self._spawn_track(frame_id, prop)

        dead = [tid for tid, t in self.tracks.items() if t.missed_frames > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        result = []
        for track in self.tracks.values():
            if track.history and track.history[-1]["frame_id"] == frame_id:
                prop = Object3D(
                    bbox_3d=track.history[-1]["bbox_3d"],
                    class_name=track.class_name,
                    confidence=track.confidence,
                    track_id=track.track_id,
                )
                result.append(prop)
        return result

    def _associate(self, proposals: List[Object3D]) -> List[Tuple[int, int]]:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return self._greedy_associate(proposals)

        track_ids = list(self.tracks.keys())
        cost = np.full((len(track_ids), len(proposals)), 1e6)

        for ti, tid in enumerate(track_ids):
            track = self.tracks[tid]
            for pi, prop in enumerate(proposals):
                dist = math.sqrt(
                    (track.state[0] - prop.bbox_3d[0]) ** 2 +
                    (track.state[1] - prop.bbox_3d[1]) ** 2
                )
                # Yaw penalty: compare against the track's last *observed* yaw
                # (the Kalman state is [x, y, vx, vy] and carries no yaw).
                # The difference is wrapped to [-pi, pi] so 359 deg vs 1 deg
                # is a 2-degree disagreement, not a 358-degree one.
                if track.history:
                    last_yaw = track.history[-1]["bbox_3d"][6]
                    raw = last_yaw - prop.bbox_3d[6]
                    yaw_diff = abs(math.atan2(math.sin(raw), math.cos(raw)))
                else:
                    yaw_diff = 0.0
                vel_penalty = self.velocity_penalty * yaw_diff
                cost[ti, pi] = dist + vel_penalty

        row_ind, col_ind = linear_sum_assignment(cost)
        assignments = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.distance_gate:
                assignments.append((track_ids[r], c))
        return assignments

    def _greedy_associate(self, proposals: List[Object3D]) -> List[Tuple[int, int]]:
        assignments = []
        used = set()
        for tid, track in self.tracks.items():
            best_pi, best_dist = None, float("inf")
            for pi, prop in enumerate(proposals):
                if pi in used:
                    continue
                dist = math.sqrt(
                    (track.state[0] - prop.bbox_3d[0]) ** 2 +
                    (track.state[1] - prop.bbox_3d[1]) ** 2
                )
                if dist < best_dist and dist < self.distance_gate:
                    best_dist = dist
                    best_pi = pi
            if best_pi is not None:
                assignments.append((tid, best_pi))
                used.add(best_pi)
        return assignments

    def _spawn_track(self, frame_id: str, prop: Object3D):
        state = np.array([prop.bbox_3d[0], prop.bbox_3d[1], 0.0, 0.0])
        track = KalmanTrack(
            track_id=self.next_id,
            state=state,
            covariance=np.eye(4),
            class_name=prop.class_name,
            confidence=prop.confidence,
        )
        track.history.append({
            "frame_id": frame_id,
            "bbox_3d": prop.bbox_3d,
            "confidence": prop.confidence,
        })
        self.tracks[self.next_id] = track
        self.next_id += 1

    def run_sequence(
        self,
        proposals_by_frame: Dict[str, List[Object3D]],
        output_path: Path,
    ) -> List[Dict]:
        for frame_id in sorted(proposals_by_frame.keys()):
            self.update(frame_id, proposals_by_frame[frame_id])

        tracks_output = []
        for tid, track in self.tracks.items():
            if track.history:
                tracks_output.append({
                    "track_id": tid,
                    "class_name": track.class_name,
                    "frames": track.history,
                })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(tracks_output, f, indent=2)
        return tracks_output

    @staticmethod
    def load_proposals(proposals_dir: Path) -> Dict[str, List[Object3D]]:
        result = {}
        for p in sorted(proposals_dir.glob("*.json")):
            frame_id = p.stem
            with open(p) as f:
                data = json.load(f)
            result[frame_id] = [Object3D.model_validate(d) for d in data]
        return result
