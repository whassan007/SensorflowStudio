"""Test temporal tracker."""

from sensorflow.schemas.unified_frame import Object3D
from sensorflow.temporal_tracker import TemporalTracker


def _make_proposal(x, y, frame_offset=0):
    return Object3D(
        bbox_3d=[x, y, 0.5, 4.0, 1.8, 1.5, 0.0],
        class_name="car",
        confidence=0.9,
    )


def test_track_persistence():
    tracker = TemporalTracker(max_age=5)
    p1 = _make_proposal(10.0, 2.0)
    result1 = tracker.update("frame_0000", [p1])
    assert len(result1) == 1
    track_id = result1[0].track_id

    p2 = _make_proposal(10.5, 2.1)
    result2 = tracker.update("frame_0001", [p2])
    assert result2[0].track_id == track_id


def test_occlusion_recovery():
    tracker = TemporalTracker(max_age=3)
    tracker.update("frame_0000", [_make_proposal(10.0, 2.0)])
    tid = list(tracker.tracks.keys())[0]

    tracker.update("frame_0001", [])
    tracker.update("frame_0002", [])

    result = tracker.update("frame_0003", [_make_proposal(11.0, 2.0)])
    assert len(result) >= 1


def test_run_sequence(tmp_path):
    tracker = TemporalTracker()
    proposals = {
        "frame_0000": [_make_proposal(10.0, 2.0)],
        "frame_0001": [_make_proposal(10.5, 2.1)],
        "frame_0002": [_make_proposal(11.0, 2.0)],
    }
    output = tmp_path / "tracks.json"
    tracks = tracker.run_sequence(proposals, output)
    assert output.exists()
    assert len(tracks) >= 1
