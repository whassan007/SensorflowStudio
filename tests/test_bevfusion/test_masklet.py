"""Masklet propagation: identity bridges planted occlusion gaps in BEV."""

from sensorflow.bevfusion.engines import FrameToFrameTracker
from sensorflow.bevfusion.masklet import BEVMaskletTracker, tracks_to_dicts
from sensorflow.schemas.unified_frame import Object3D


def _det(x, y, cls="vehicle", conf=0.9):
    return Object3D(bbox_3d=[x, y, 0.8, 4.5, 1.9, 1.6, 0.0],
                    class_name=cls, confidence=conf)


def _stream_with_gap(gap_frames=(5, 6), n=11):
    """Object moving +1.2 m/frame in x; detections missing during the gap."""
    stream = []
    for fi in range(n):
        dets = [] if fi in gap_frames else [_det(10.0 + 1.2 * fi, 3.0)]
        stream.append((f"f{fi:03d}", dets))
    return stream


def test_masklet_bridges_two_frame_occlusion_without_id_switch():
    tracker = BEVMaskletTracker()
    ids_seen = set()
    propagated_frames = []
    for frame_id, dets in _stream_with_gap():
        out = tracker.update(frame_id, dets)
        assert len(out) == 1, f"expected exactly one box at {frame_id}"
        ids_seen.add(out[0].track_id)
        if out[0].track_id in tracker.last_propagated:
            propagated_frames.append(frame_id)
    assert len(ids_seen) == 1, "identity must be carried through the gap"
    assert propagated_frames == ["f005", "f006"], \
        "gap frames must be filled by propagated masklet boxes"


def test_propagated_box_follows_predicted_bev_position():
    tracker = BEVMaskletTracker()
    last_real_x = None
    for frame_id, dets in _stream_with_gap(gap_frames=(5, 6), n=8):
        out = tracker.update(frame_id, dets)
        if frame_id == "f004":
            last_real_x = out[0].bbox_3d[0]
        if frame_id == "f006":
            # The Kalman-predicted position must keep moving forward, not
            # freeze at the last observed position.
            assert out[0].bbox_3d[0] > last_real_x + 0.5


def test_baseline_frame_to_frame_tracker_switches_id_after_gap():
    tracker = FrameToFrameTracker(gate=3.0)
    ids_seen = set()
    for frame_id, dets in _stream_with_gap():
        boxes = [{"bbox_3d": d.bbox_3d, "class_name": d.class_name,
                  "confidence": d.confidence} for d in dets]
        for b in tracker.update(boxes):
            ids_seen.add(b["track_id"])
    assert len(ids_seen) >= 2, "the naive baseline must fragment across the gap"


def test_track_class_voting_stabilizes_semantics():
    """One noisy LiDAR-template class flip must not change the track's label."""
    tracker = BEVMaskletTracker()
    classes = ["pedestrian", "pedestrian", "cyclist", "pedestrian"]
    emitted = []
    for fi, cls in enumerate(classes):
        out = tracker.update(f"f{fi:03d}", [_det(10.0 + fi, 0.0, cls=cls)])
        emitted.append(out[0].class_name)
    assert emitted[2] == "pedestrian", "majority vote must override the flip"
    assert emitted[-1] == "pedestrian"


def test_tracks_to_dicts_groups_by_track_id():
    per_frame = {
        "f000": [{"track_id": 1, "class_name": "vehicle", "bbox_3d": [1, 0, 0, 4, 2, 1.5, 0]}],
        "f001": [{"track_id": 1, "class_name": "vehicle", "bbox_3d": [2, 0, 0, 4, 2, 1.5, 0]},
                 {"track_id": 2, "class_name": "pedestrian", "bbox_3d": [8, 3, 0, 0.7, 0.7, 1.7, 0]}],
    }
    tracks = {t["track_id"]: t for t in tracks_to_dicts(per_frame)}
    assert len(tracks[1]["frames"]) == 2
    assert len(tracks[2]["frames"]) == 1
