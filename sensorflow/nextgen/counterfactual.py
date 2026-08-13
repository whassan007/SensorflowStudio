"""Counterfactual scenario engine: deterministic transformations over
bevfusion scenes.

REUSE DECISION: scenarios are NOT a new format. A counterfactual is a
transformation of a :class:`sensorflow.bevfusion.scenes.SceneSequence` — the
platform's existing scenario substrate — so everything that already consumes
those sequences (sensor simulators, both perception engines, the evaluate
module) runs on counterfactuals unchanged. We reconstruct world-frame actor
kinematics from the source frames, transform them, and re-render frames with
the same conventions (ego +x at 10 m/s, culling bounds, occlusion flags) as
scenes.generate_sequences.

Transformation catalogue
------------------------
environment.*  clear->fog / clear->rain / day->night / glare / wet_road
actors.*       sudden_brake, swerve, stationary_to_crossing,
               occluded_emergence, teleport (stress transform: physically
               implausible BY DESIGN, used to prove the validity gate rejects)
scene.*        add_construction_zone, pedestrian_density, add_occlusion,
               distance_shift, lane_shift

Every generated scenario carries provenance (source scene id, recipe, seed,
generator + version) and the data label COUNTERFACTUAL. Provenance is
persisted with the scenario and carried into every downstream report.

Extended environment conditions (fog / glare / wet_road) are not modeled by
the bevfusion sensor simulators (which know day/night, clear/rain). They are
recorded as environment tags here and applied as documented, parameterized
detection-degradation factors by the closed-loop perception model
(closedloop.PERCEPTION_CONDITION_EFFECTS) and by the validity gate's sensor
consistency expectations.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional
from uuid import uuid4

import numpy as np

from sensorflow.bevfusion.scenes import (
    CLASS_DIMS, DT, EGO_SPEED, GTBox, SceneFrame, SceneSequence,
    generate_sequences,
)
from sensorflow.nextgen import store
from sensorflow.nextgen.models import (
    CounterfactualScenario, DataLabel, Provenance, TransformationStep,
)
from sensorflow.nextgen.worldmodel import (
    ActorTrack, SceneTransformer, TransformedScene,
)

# Culling bounds identical to scenes.generate_sequences.
_X_MIN, _X_MAX, _Y_MAX = 3.0, 78.0, 22.0


# ------------------------------------------------------- reconstruction


def reconstruct_actors(seq: SceneSequence) -> List[ActorTrack]:
    """Recover world-frame constant-velocity kinematics from a sequence.

    Source scenes are constant-velocity by construction (scenes.py), so a
    least-squares linear fit over the observed frames recovers the exact
    generating kinematics; frames where the actor was culled are filled by
    extrapolation.
    """
    n_frames = len(seq.frames)
    obs: Dict[str, Dict] = {}
    for frame in seq.frames:
        ego_x = EGO_SPEED * DT * frame.index
        for gt in frame.gt:
            rec = obs.setdefault(gt.instance_id, {
                "cls": gt.class_name, "dims": gt.bbox_3d[3:6],
                "yaw": gt.bbox_3d[6], "t": [], "x": [], "y": [],
                "occ": {},
            })
            rec["t"].append(frame.index * DT)
            rec["x"].append(gt.bbox_3d[0] + ego_x)   # world x
            rec["y"].append(gt.bbox_3d[1])           # world y (ego drives +x)
            rec["occ"][frame.index] = gt.occluded

    actors: List[ActorTrack] = []
    for iid, rec in obs.items():
        t = np.asarray(rec["t"])
        if t.size >= 2:
            vx, x0 = np.polyfit(t, np.asarray(rec["x"]), 1)
            vy, y0 = np.polyfit(t, np.asarray(rec["y"]), 1)
        else:
            x0, y0, vx, vy = rec["x"][0], rec["y"][0], 0.0, 0.0
        states = []
        for fi in range(n_frames):
            ti = fi * DT
            states.append({
                "x": float(x0 + vx * ti), "y": float(y0 + vy * ti),
                "vx": float(vx), "vy": float(vy), "yaw": float(rec["yaw"]),
                "occluded": bool(rec["occ"].get(fi, False)),
            })
        actors.append(ActorTrack(instance_id=iid, class_name=rec["cls"],
                                 dims=[float(d) for d in rec["dims"]],
                                 states=states))
    return actors


def render_sequence(actors: List[ActorTrack], environment: Dict[str, str],
                    n_frames: int, sequence_id: str) -> SceneSequence:
    """Re-render frames from world tracks with the scenes.py conventions."""
    seq = SceneSequence(
        sequence_id=sequence_id,
        time_of_day=environment.get("time_of_day", "day"),
        weather=environment.get("weather", "clear"))
    for fi in range(n_frames):
        ego_x = EGO_SPEED * DT * fi
        frame = SceneFrame(frame_id=f"{sequence_id}-f{fi:04d}", index=fi)
        for actor in actors:
            st = actor.states[fi]
            if st.get("absent"):
                continue
            rel_x, rel_y = st["x"] - ego_x, st["y"]
            if rel_x < _X_MIN or rel_x > _X_MAX or abs(rel_y) > _Y_MAX:
                continue
            l, w, h = actor.dims
            frame.gt.append(GTBox(
                instance_id=actor.instance_id, class_name=actor.class_name,
                bbox_3d=[round(rel_x, 3), round(rel_y, 3), round(h / 2, 3),
                         l, w, h, round(st["yaw"], 4)],
                occluded=bool(st["occluded"]),
                distance=round(math.hypot(rel_x, rel_y), 3)))
        seq.frames.append(frame)
    return seq


# ------------------------------------------------------- transform helpers


def _brake_from(states: List[Dict], k: int, decel: float) -> None:
    """Re-integrate a track from frame k braking at `decel` along its fixed
    heading; the speed profile is a clamped scalar (stops, never reverses)."""
    st = states[k]
    speed0 = math.hypot(st["vx"], st["vy"])
    if speed0 < 1e-6:
        return
    ux, uy = st["vx"] / speed0, st["vy"] / speed0
    for fi in range(k + 1, len(states)):
        prev = states[fi - 1]
        speed = max(0.0, math.hypot(prev["vx"], prev["vy"]) - decel * DT)
        vx, vy = ux * speed, uy * speed
        states[fi].update({
            "x": prev["x"] + vx * DT, "y": prev["y"] + vy * DT,
            "vx": vx, "vy": vy, "yaw": st["yaw"],
        })


def _ramped_lateral(y0: float, vy_target: float, t_start: float, ti: float,
                    ramp_a: float = 6.0) -> tuple:
    """Lateral position/velocity with a bounded-acceleration ramp to
    vy_target starting at t_start (keeps transforms physically plausible:
    ramp_a stays below the validity gate's pedestrian limit)."""
    if ti <= t_start:
        return y0, 0.0
    t = ti - t_start
    t_ramp = abs(vy_target) / ramp_a
    if t <= t_ramp:
        vy = math.copysign(ramp_a * t, vy_target)
        y = y0 + math.copysign(0.5 * ramp_a * t * t, vy_target)
    else:
        vy = vy_target
        y = y0 + math.copysign(0.5 * ramp_a * t_ramp * t_ramp, vy_target) \
            + vy_target * (t - t_ramp)
    return y, vy


def _clear_x(actors: List[ActorTrack], n_frames: int, y: float,
             rng: np.random.Generator, x_lo: float = 18.0, x_hi: float = 60.0,
             min_dist: float = 4.0) -> float:
    """Longitudinal spot whose distance to every existing trajectory at the
    given y stays >= min_dist (best-of-K deterministic search)."""
    best_x, best_d = (x_lo + x_hi) / 2, -1.0
    for _ in range(24):
        x = float(rng.uniform(x_lo, x_hi))
        d = min((math.hypot(st["x"] - x, st["y"] - y)
                 for a in actors for st in a.states[::4]), default=1e9)
        if d > best_d:
            best_x, best_d = x, d
        if d >= min_dist:
            return x
    return best_x


def _lead_actor(actors: List[ActorTrack]) -> Optional[ActorTrack]:
    """Nearest vehicle-like actor ahead of ego near the ego lane at t=0;
    falls back to the fastest lane-following motorized actor (braking a
    slow crossing pedestrian is not a meaningful sudden-brake scenario)."""
    best, best_x = None, float("inf")
    for a in actors:
        st = a.states[0]
        if a.class_name in ("vehicle", "truck") and st["x"] > 5.0 and abs(st["y"]) < 4.0:
            if st["x"] < best_x:
                best, best_x = a, st["x"]
    if best is not None:
        return best
    motorized = [a for a in actors
                 if a.class_name in ("vehicle", "truck", "motorcycle")
                 and abs(a.states[0]["vy"]) < 0.5]
    if motorized:
        return max(motorized, key=lambda a: math.hypot(a.states[0]["vx"],
                                                       a.states[0]["vy"]))
    return actors[0] if actors else None


def _pick(actors: List[ActorTrack], params: Dict, rng: np.random.Generator,
          prefer_class: Optional[str] = None) -> Optional[ActorTrack]:
    iid = params.get("instance_id")
    if iid:
        for a in actors:
            if a.instance_id == iid:
                return a
    if prefer_class:
        cands = [a for a in actors if a.class_name == prefer_class]
        if cands:
            return cands[int(rng.integers(0, len(cands)))]
    return actors[int(rng.integers(0, len(actors)))] if actors else None


# ------------------------------------------------------- transformations

# Each op: (actors, env, rng, params, n_frames) -> note string.

def _op_env(tag: str, value: str) -> Callable:
    def op(actors, env, rng, params, n_frames):
        env[tag] = value
        return f"environment: {tag} -> {value}"
    return op


def _op_sudden_brake(actors, env, rng, params, n_frames):
    target = _pick(actors, params, rng, prefer_class="vehicle") if params.get("instance_id") \
        else _lead_actor(actors)
    if target is None:
        return "sudden_brake: no actor available"
    t_start = float(params.get("t_start_s", 1.0))
    decel = float(params.get("decel_mps2", 7.5))
    k = max(0, min(n_frames - 1, int(round(t_start / DT))))
    st = target.states[k]
    if math.hypot(st["vx"], st["vy"]) < 1e-6:
        return f"sudden_brake: target {target.instance_id} already stationary"
    _brake_from(target.states, k, decel)
    # Chain reaction: same-lane followers brake 0.8 s later (traffic is not
    # frozen — otherwise the counterfactual plants unphysical rear-endings
    # that the validity gate would rightly reject).
    k2 = min(n_frames - 1, k + int(round(0.8 / DT)))
    speed_t = math.hypot(st["vx"], st["vy"])
    ux, uy = st["vx"] / speed_t, st["vy"] / speed_t
    n_followers = 0
    for a in actors:
        if a is target:
            continue
        s0 = a.states[k]
        along = (st["x"] - s0["x"]) * ux + (st["y"] - s0["y"]) * uy
        lateral = abs(-(st["x"] - s0["x"]) * uy + (st["y"] - s0["y"]) * ux)
        same_dir = (s0["vx"] * ux + s0["vy"] * uy) > 0.5
        if along > 0 and lateral < 2.5 and same_dir:
            _brake_from(a.states, k2, min(decel, 8.0))
            n_followers += 1
    return (f"sudden_brake: {target.instance_id} decelerates at "
            f"{decel} m/s^2 from t={t_start}s; {n_followers} follower(s) "
            f"chain-brake at t={t_start + 0.8}s")


def _op_swerve(actors, env, rng, params, n_frames):
    target = _pick(actors, params, rng, prefer_class="vehicle")
    if target is None:
        return "swerve: no actor available"
    t_start = float(params.get("t_start_s", 1.0))
    lat_acc = float(params.get("lateral_accel_mps2", 2.5))
    duration = float(params.get("duration_s", 1.2))
    k = max(0, min(n_frames - 1, int(round(t_start / DT))))
    k_end = min(n_frames - 1, int(round((t_start + duration) / DT)))
    sign = -1.0 if target.states[k]["y"] > 0 else 1.0  # swerve toward ego lane
    for fi in range(k + 1, n_frames):
        prev = target.states[fi - 1]
        ay = sign * lat_acc if fi <= k_end else 0.0
        vy = prev["vy"] + ay * DT
        target.states[fi].update({
            "x": prev["x"] + prev["vx"] * DT, "y": prev["y"] + vy * DT,
            "vx": prev["vx"], "vy": vy,
            "yaw": math.atan2(vy, prev["vx"]) if prev["vx"] or vy else prev["yaw"],
        })
    return (f"swerve: {target.instance_id} lateral accel {sign * lat_acc:+.1f} "
            f"m/s^2 for {duration}s from t={t_start}s")


def _op_stationary_to_crossing(actors, env, rng, params, n_frames):
    target = _pick(actors, params, rng, prefer_class="pedestrian")
    if target is None:
        return "stationary_to_crossing: no actor available"
    t_start = float(params.get("t_start_s", 0.8))
    st0 = target.states[0]
    # Time the crossing so the actor reaches the ego lane center (y=0) when
    # a nominal-speed ego reaches its x -> a genuine conflict geometry.
    t_conflict = max(t_start + 0.8, st0["x"] / EGO_SPEED)
    vy = -st0["y"] / max(t_conflict - t_start, 0.5)
    vy = float(np.clip(vy, -2.5, 2.5))  # pedestrian-plausible speed
    for fi in range(n_frames):
        ti = fi * DT
        s = target.states[fi]
        y, vy_i = _ramped_lateral(st0["y"], vy, t_start, ti)
        s.update({"x": st0["x"], "y": y, "vx": 0.0, "vy": vy_i,
                  "yaw": (math.pi / 2 if vy > 0 else -math.pi / 2)
                  if vy_i else s["yaw"]})
    return (f"stationary_to_crossing: {target.instance_id} crosses at "
            f"{vy:+.2f} m/s from t={t_start}s (conflict ~t={t_conflict:.1f}s)")


def _op_occluded_emergence(actors, env, rng, params, n_frames):
    """Plant an occluder truck and a pedestrian that emerges from behind it."""
    t_emerge = float(params.get("t_emerge_s", 1.6))
    y_side = float(params.get("y_m", 4.0))
    tl, tw, th = CLASS_DIMS["truck"]
    x_emerge = float(params.get("x_m", _clear_x(actors, n_frames, y_side, rng,
                                                x_lo=EGO_SPEED * t_emerge + 10.0,
                                                x_hi=EGO_SPEED * t_emerge + 34.0)))
    truck = ActorTrack(
        instance_id=f"cf-occluder-{len(actors)}", class_name="truck",
        dims=[tl, tw, th],
        states=[{"x": x_emerge, "y": y_side + 1.8, "vx": 0.0, "vy": 0.0,
                 "yaw": 0.0, "occluded": False} for _ in range(n_frames)])
    pl, pw, ph = CLASS_DIMS["pedestrian"]
    vy = -1.9
    states = []
    for fi in range(n_frames):
        ti = fi * DT
        y, vy_i = _ramped_lateral(y_side, vy, t_emerge, ti)
        states.append({"x": x_emerge + tl / 2 + 0.8, "y": y,
                       "vx": 0.0, "vy": vy_i, "yaw": -math.pi / 2,
                       "occluded": ti <= t_emerge})
    ped = ActorTrack(instance_id=f"cf-emergent-ped-{len(actors)}",
                     class_name="pedestrian", dims=[pl, pw, ph], states=states)
    actors.extend([truck, ped])
    return (f"occluded_emergence: pedestrian emerges from behind truck at "
            f"x={x_emerge:.1f} m, t={t_emerge}s")


def _op_teleport(actors, env, rng, params, n_frames):
    """STRESS TRANSFORM: instantaneous position jump — physically implausible
    by construction. Exists so tests/demos can prove the validity gate
    rejects implausible scenarios; never part of production recipes."""
    target = _pick(actors, params, rng)
    if target is None:
        return "teleport: no actor available"
    k = max(1, min(n_frames - 1, int(round(float(params.get("t_s", 1.0)) / DT))))
    jump = float(params.get("jump_m", 25.0))
    for fi in range(k, n_frames):
        target.states[fi]["x"] += jump
    return f"teleport: {target.instance_id} jumps {jump} m at frame {k} (IMPLAUSIBLE)"


def _op_construction_zone(actors, env, rng, params, n_frames):
    length = float(params.get("length_m", 20.0))
    # Deterministic search for a barrier line clear of existing trajectories.
    y_line = params.get("y_m")
    x0 = params.get("x_start_m")
    if x0 is None or y_line is None:
        best, best_d = (30.0, 3.4), -1.0
        for _ in range(36):
            cx = float(rng.uniform(20.0, 55.0 - length / 2))
            cy = float(rng.choice([3.4, 4.8, 6.2, -3.4, -4.8]))
            d = min((math.hypot(min(max(st["x"], cx), cx + length) - st["x"],
                                cy - st["y"])
                     for a in actors for st in a.states[::2]), default=1e9)
            if d > best_d:
                best, best_d = (cx, cy), d
            if d >= 3.2:
                break
        x0 = x0 if x0 is not None else best[0]
        y_line = y_line if y_line is not None else best[1]
    x0, y_line = float(x0), float(y_line)
    n_barriers = max(2, int(length / 8))
    for bi in range(n_barriers):
        actors.append(ActorTrack(
            instance_id=f"cf-construction-{bi}", class_name="truck",
            dims=[6.0, 2.2, 2.4],
            states=[{"x": x0 + bi * (length / max(n_barriers - 1, 1)),
                     "y": y_line, "vx": 0.0, "vy": 0.0, "yaw": 0.0,
                     "occluded": False} for _ in range(n_frames)]))

    # Lane-following traffic near the closed shoulder shifts away, with a
    # smooth 10 m taper. Membership is decided ONCE per actor (initial lane +
    # longitudinal travel), so no temporal discontinuity is introduced.
    def taper(x: float) -> float:
        ramp = 10.0
        if x < x0 - ramp or x > x0 + length + ramp:
            return 0.0
        if x < x0:
            u = (x - (x0 - ramp)) / ramp
        elif x > x0 + length:
            u = ((x0 + length + ramp) - x) / ramp
        else:
            u = 1.0
        return u * u * (3 - 2 * u)  # smoothstep

    away = -1.0 if y_line > 0 else 1.0  # shift direction: away from barriers
    for a in actors:
        if a.instance_id.startswith("cf-construction"):
            continue
        st0 = a.states[0]
        lane_following = abs(st0["vy"]) < 0.5
        in_band = abs(st0["y"] - y_line) <= 4.0
        if lane_following and in_band:
            for st in a.states:
                st["y"] += away * 1.6 * taper(st["x"])
    env["construction_zone"] = f"x={x0:.0f}..{x0 + length:.0f}m"
    return f"add_construction_zone: {n_barriers} barriers at x={x0:.0f}..{x0+length:.0f} m"


def _op_pedestrian_density(actors, env, rng, params, n_frames):
    n = int(params.get("n", 4))
    pl, pw, ph = CLASS_DIMS["pedestrian"]
    for pi in range(n):
        y = float(rng.uniform(4.0, 10.0)) * (1 if pi % 2 == 0 else -1)
        x = _clear_x(actors, n_frames, y, rng, x_lo=15.0, x_hi=65.0)
        vy = float(rng.uniform(0.8, 1.6)) * (-1 if y > 0 else 1)
        states = []
        for fi in range(n_frames):
            yy, vy_i = _ramped_lateral(y, vy, 0.2, fi * DT)
            states.append({"x": x, "y": yy, "vx": 0.0, "vy": vy_i,
                           "yaw": math.copysign(math.pi / 2, vy),
                           "occluded": False})
        actors.append(ActorTrack(
            instance_id=f"cf-ped-{pi}", class_name="pedestrian",
            dims=[pl, pw, ph], states=states))
    return f"pedestrian_density: +{n} crossing pedestrians"


def _op_add_occlusion(actors, env, rng, params, n_frames):
    target = _pick(actors, params, rng)
    if target is None:
        return "add_occlusion: no actor available"
    start = int(params.get("start_frame", max(2, n_frames // 3)))
    length = int(params.get("length_frames", 4))
    for fi in range(start, min(start + length, n_frames)):
        target.states[fi]["occluded"] = True
    return f"add_occlusion: {target.instance_id} occluded frames {start}..{start+length-1}"


def _op_distance_shift(actors, env, rng, params, n_frames):
    dx = float(params.get("delta_m", 10.0))
    for a in actors:
        for st in a.states:
            st["x"] += dx
    return f"distance_shift: all actors shifted {dx:+.1f} m longitudinally"


def _op_lane_shift(actors, env, rng, params, n_frames):
    dy = float(params.get("delta_m", 1.5))
    for a in actors:
        for st in a.states:
            st["y"] += dy
    return f"lane_shift: road geometry shifted {dy:+.1f} m laterally"


TRANSFORMATIONS: Dict[str, Callable] = {
    "environment.clear_to_fog": _op_env("weather_extended", "fog"),
    "environment.clear_to_rain": _op_env("weather", "rain"),
    "environment.day_to_night": _op_env("time_of_day", "night"),
    "environment.add_glare": _op_env("glare", "sunset_glare"),
    "environment.wet_road": _op_env("road_surface", "wet"),
    "actors.sudden_brake": _op_sudden_brake,
    "actors.swerve": _op_swerve,
    "actors.stationary_to_crossing": _op_stationary_to_crossing,
    "actors.occluded_emergence": _op_occluded_emergence,
    "actors.teleport": _op_teleport,
    "scene.add_construction_zone": _op_construction_zone,
    "scene.pedestrian_density": _op_pedestrian_density,
    "scene.add_occlusion": _op_add_occlusion,
    "scene.distance_shift": _op_distance_shift,
    "scene.lane_shift": _op_lane_shift,
}


def transformation_catalogue() -> List[Dict]:
    """Recipe-builder metadata for the UI."""
    docs = {
        "environment.clear_to_fog": {"family": "environment", "params": {}},
        "environment.clear_to_rain": {"family": "environment", "params": {}},
        "environment.day_to_night": {"family": "environment", "params": {}},
        "environment.add_glare": {"family": "environment", "params": {}},
        "environment.wet_road": {"family": "environment", "params": {}},
        "actors.sudden_brake": {"family": "actors",
                                "params": {"t_start_s": 1.0, "decel_mps2": 7.5}},
        "actors.swerve": {"family": "actors",
                          "params": {"t_start_s": 1.0, "lateral_accel_mps2": 2.5,
                                     "duration_s": 1.2}},
        "actors.stationary_to_crossing": {"family": "actors",
                                          "params": {"t_start_s": 0.8}},
        "actors.occluded_emergence": {"family": "actors",
                                      "params": {"t_emerge_s": 1.6}},
        "actors.teleport": {"family": "actors (stress/test only)",
                            "params": {"t_s": 1.0, "jump_m": 25.0}},
        "scene.add_construction_zone": {"family": "scene",
                                        "params": {"x_start_m": 30.0, "length_m": 20.0}},
        "scene.pedestrian_density": {"family": "scene", "params": {"n": 4}},
        "scene.add_occlusion": {"family": "scene",
                                "params": {"start_frame": 8, "length_frames": 4}},
        "scene.distance_shift": {"family": "scene", "params": {"delta_m": 10.0}},
        "scene.lane_shift": {"family": "scene", "params": {"delta_m": 1.5}},
    }
    return [{"kind": k, **v} for k, v in docs.items()]


# ------------------------------------------------------- the live transformer


class DeterministicSceneTransformer(SceneTransformer):
    """The internal world model: deterministic rule-based scene transformer.

    Pure function of (source, recipe, seed); all randomness (actor selection,
    added-actor placement) is drawn from a generator seeded with the given
    seed, so a scenario's provenance record fully reproduces it.
    """

    name = "nextgen.worldmodel.DeterministicSceneTransformer"
    version = "1.0"

    def transform(self, source: SceneSequence, recipe: List[TransformationStep],
                  seed: int) -> TransformedScene:
        rng = np.random.default_rng([seed, 0xC0FFEE])
        n_frames = len(source.frames)
        actors = reconstruct_actors(source)
        env: Dict[str, str] = {"time_of_day": source.time_of_day,
                               "weather": source.weather}
        notes: List[str] = []
        for step in recipe:
            op = TRANSFORMATIONS.get(step.kind)
            if op is None:
                raise ValueError(f"unknown transformation {step.kind!r}; "
                                 f"known: {sorted(TRANSFORMATIONS)}")
            notes.append(op(actors, env, rng, step.params, n_frames))
        seq = render_sequence(actors, env, n_frames,
                              sequence_id=f"cf-{source.sequence_id}-{seed}")
        return TransformedScene(sequence=seq, actors=actors, environment=env,
                                applied=list(recipe), notes=notes)


# ------------------------------------------------------- generation + storage


def _actor_dump(a: ActorTrack) -> Dict:
    return {"instance_id": a.instance_id, "class_name": a.class_name,
            "dims": a.dims, "states": a.states}


def _actor_load(d: Dict) -> ActorTrack:
    return ActorTrack(instance_id=d["instance_id"], class_name=d["class_name"],
                      dims=d["dims"], states=d["states"])


def generate_counterfactuals(recipe: List[TransformationStep], seed: int = 7,
                             n_scenarios: int = 3,
                             frames_per_sequence: int = 40) -> List[CounterfactualScenario]:
    """Generate counterfactual scenarios from bevfusion source scenes and
    persist them (with full provenance) under runs/nextgen/counterfactuals/."""
    sources = generate_sequences(n_sequences=n_scenarios,
                                 frames_per_sequence=frames_per_sequence,
                                 seed=seed)
    from sensorflow.nextgen.validity import overlap_pairs, scene_features
    transformer = DeterministicSceneTransformer()
    out: List[CounterfactualScenario] = []
    for si, source in enumerate(sources):
        scenario_seed = seed * 1009 + si
        source_overlaps = overlap_pairs(reconstruct_actors(source))
        source_feats = {k: [round(float(x), 3) for x in v]
                        for k, v in scene_features(source).items()}
        result = transformer.transform(source, recipe, scenario_seed)
        scenario_id = f"cfs-{uuid4().hex[:10]}"
        scenario = CounterfactualScenario(
            scenario_id=scenario_id,
            provenance=Provenance(
                source_scene_id=source.sequence_id,
                recipe=list(recipe), seed=scenario_seed,
                generator=transformer.name,
                generator_version=transformer.version,
                data_label=DataLabel.COUNTERFACTUAL),
            n_frames=len(result.sequence.frames),
            n_actors=len(result.actors),
            environment=result.environment)
        store.write_json({
            "scenario": scenario.model_dump(mode="json"),
            "sequence": result.sequence.model_dump(mode="json"),
            "actors": [_actor_dump(a) for a in result.actors],
            "notes": result.notes,
            "source_overlap_pairs": [sorted(p) for p in source_overlaps],
            "source_features": source_feats,
        }, "counterfactuals", f"{scenario_id}.json")
        out.append(scenario)
    return out


def load_bundle(scenario_id: str) -> Optional[Dict]:
    """Load a persisted scenario bundle: scenario + sequence + world tracks."""
    raw = store.read_json("counterfactuals", f"{scenario_id}.json")
    if raw is None:
        return None
    return {
        "scenario": CounterfactualScenario(**raw["scenario"]),
        "sequence": SceneSequence(**raw["sequence"]),
        "actors": [_actor_load(d) for d in raw["actors"]],
        "notes": raw.get("notes", []),
        "source_overlap_pairs": {frozenset(p) for p in
                                 raw.get("source_overlap_pairs", [])},
        "source_features": raw.get("source_features"),
    }


def save_scenario(scenario: CounterfactualScenario) -> None:
    """Update the persisted scenario record (e.g. after validity gating)."""
    raw = store.read_json("counterfactuals", f"{scenario.scenario_id}.json")
    if raw is None:
        raise KeyError(f"unknown scenario {scenario.scenario_id}")
    raw["scenario"] = scenario.model_dump(mode="json")
    store.write_json(raw, "counterfactuals", f"{scenario.scenario_id}.json")


def list_scenarios() -> List[CounterfactualScenario]:
    out = []
    for sid in store.list_json("counterfactuals"):
        raw = store.read_json("counterfactuals", f"{sid}.json")
        if raw:
            out.append(CounterfactualScenario(**raw["scenario"]))
    return out
