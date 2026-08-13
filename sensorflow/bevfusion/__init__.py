"""BEV-Fusion perception engine (inspired by BEVFusion) with masklet-style
temporal propagation (inspired by SAM 3).

What this package does
----------------------
Fuses camera and LiDAR evidence in a unified bird's-eye-view (BEV) plane to
produce better auto-labels than a single-sensor (camera-primary) labeler, and
demonstrates the improvement with the platform's own evaluation machinery on
synthetic scenes with known ground truth.

Concept mapping (paper -> this implementation)
----------------------------------------------
BEVFusion:
    - "Lift camera features to BEV via calibrated geometry"
        -> :mod:`geometry`: pinhole projection with a 6-DoF camera-to-ego
           extrinsic; monocular detections are back-projected along the viewing
           ray into the BEV plane with an anisotropic (along-ray-elongated)
           position covariance.
    - "Rasterize LiDAR features into a BEV grid"
        -> :mod:`fusion`: LiDAR detections are splatted into a metric BEV
           occupancy/feature grid with tight isotropic covariance.
    - "Fuse the modality feature maps in the shared BEV space"
        -> :mod:`fusion.fuse_maps`: per-cell fusion of existence evidence
           (noisy-OR), class evidence (reliability-weighted histograms; camera
           weighted higher), and geometry evidence (inverse-variance weighted
           accumulators; LiDAR dominates position where present).
    - "Decode fused BEV features into 3D boxes"
        -> :mod:`fusion.decode_bev`: peak extraction + ray-aware NMS +
           continuous weighted decoding of center/dims/yaw/class.

SAM 3 masklets:
    - "Propagate each object's identity through time, surviving brief
      disappearances without re-identification"
        -> :mod:`masklet.BEVMaskletTracker`: association in the stable BEV
           plane (Hungarian on BEV distance + class + size, extending
           :mod:`sensorflow.temporal_tracker`), with Kalman-predicted BEV
           position carrying identity across short dropouts and emitting
           propagated ("masklet") boxes during occlusion gaps.

What is simulated vs. real
--------------------------
There are NO learned features and NO neural networks here (the platform is
numpy-only by design). Camera/LiDAR "detections" are sampled from ground truth
with realistic, modality-specific failure modes (camera: occlusion/night
misses and depth ambiguity but good class discrimination; LiDAR: accurate
geometry, degraded at long range and in rain, weak class discrimination).
The fusion itself is honest geometric + probabilistic math, and the measured
improvement arises from real sensor complementarity in that math: LiDAR
existence evidence recovers camera misses, camera class evidence flows along
the viewing ray to LiDAR-anchored cells, and inverse-variance weighting gives
fused boxes near-LiDAR position accuracy. Nothing in the comparison report is
hardcoded; every number is computed at runtime from the fused outputs.
"""
