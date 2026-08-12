"""3D perception metrics: mAP, mAR, IoU, orientation/position error."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def _rotate_corners(cx: float, cy: float, l: float, w: float, yaw: float) -> np.ndarray:
    """Return 4 BEV corners of rotated rectangle."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    corners = np.array([
        [-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2],
    ])
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return corners @ rot.T + np.array([cx, cy])


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman polygon clipping (convex)."""
    def inside(p, edge_start, edge_end):
        return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - \
               (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0]) >= 0

    def intersection(s, e, cp1, cp2):
        dc = cp1 - cp2
        dp = s - e
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(denom) < 1e-10:
            return s
        return np.array([(n1 * dp[0] - n2 * dc[0]) / denom, (n1 * dp[1] - n2 * dc[1]) / denom])

    output = subject.copy()
    for i in range(len(clip)):
        input_list = output
        output = []
        cp1, cp2 = clip[i], clip[(i + 1) % len(clip)]
        if len(input_list) == 0:
            break
        s = input_list[-1]
        for e in input_list:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    output.append(intersection(s, e, cp1, cp2))
                output.append(e)
            elif inside(s, cp1, cp2):
                output.append(intersection(s, e, cp1, cp2))
            s = e
        output = np.array(output) if output else np.array([]).reshape(0, 2)
    return output


def bev_iou(box_a: List[float], box_b: List[float]) -> float:
    """BEV rotated rectangle IoU for bbox_3d [x,y,z,l,w,h,yaw]."""
    xa, ya, la, wa, ya = box_a[0], box_a[1], box_a[3], box_a[4], box_a[6]
    xb, yb, lb, wb, yb = box_b[0], box_b[1], box_b[3], box_b[4], box_b[6]
    poly_a = _rotate_corners(xa, ya, la, wa, ya)
    poly_b = _rotate_corners(xb, yb, lb, wb, yb)
    inter = _clip_polygon(poly_a, poly_b)
    if len(inter) < 3:
        return 0.0
    inter_area = _polygon_area(inter)
    area_a = la * wa
    area_b = lb * wb
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _match_boxes(
    preds: List[List[float]],
    gts: List[List[float]],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy matching by IoU."""
    matched_pairs = []
    matched_gt = set()
    matched_pred = set()
    ious = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            iou = bev_iou(p, g)
            if iou >= iou_threshold:
                ious.append((iou, pi, gi))
    ious.sort(reverse=True)
    for iou, pi, gi in ious:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pairs.append((pi, gi, iou))
            matched_pred.add(pi)
            matched_gt.add(gi)
    unmatched_pred = [i for i in range(len(preds)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gts)) if i not in matched_gt]
    return matched_pairs, unmatched_pred, unmatched_gt


def compute_map_mar(
    all_preds: List[List[float]],
    all_gts: List[List[float]],
    iou_thresholds: List[float] = None,
) -> Dict[str, float]:
    """Compute mean AP and mean AR across IoU thresholds."""
    iou_thresholds = iou_thresholds or [0.5, 0.7]
    aps, ars = [], []
    for thresh in iou_thresholds:
        pairs, unmatched_pred, unmatched_gt = _match_boxes(all_preds, all_gts, thresh)
        tp = len(pairs)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        aps.append(precision)
        ars.append(recall)
    return {
        "map_3d": float(np.mean(aps)) if aps else 0.0,
        "mar_3d": float(np.mean(ars)) if ars else 0.0,
        "mean_iou_3d": float(np.mean([p[2] for p in _match_boxes(all_preds, all_gts, 0.5)[0]])) if all_preds and all_gts and _match_boxes(all_preds, all_gts, 0.5)[0] else 0.0,
    }


def mean_orientation_error(matched_pairs: List[Tuple[List[float], List[float]]]) -> float:
    """Mean absolute yaw error in degrees."""
    if not matched_pairs:
        return 0.0
    errors = []
    for pred, gt in matched_pairs:
        diff = abs(pred[6] - gt[6])
        diff = min(diff, 2 * math.pi - diff)
        errors.append(math.degrees(diff))
    return float(np.mean(errors))


def mean_position_error(matched_pairs: List[Tuple[List[float], List[float]]]) -> float:
    """Mean L2 BEV center distance in meters."""
    if not matched_pairs:
        return 0.0
    errors = []
    for pred, gt in matched_pairs:
        dist = math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
        errors.append(dist)
    return float(np.mean(errors))
