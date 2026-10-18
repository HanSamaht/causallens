"""PASS_RUSH_DRILL phase segmentation and feature extraction.

Official drill geometry, from the competition host in discussion 748656:
  - blocking dummies stand 4 yards from the center on each side
  - a second set stands 4 yards behind the first set
  - the finish cone sits 7 yards behind the center
  - timing starts on the snap and stops when the player passes the cone
  - LEFT execution uses a rip technique, RIGHT uses a swim technique

The arc phase is therefore anchored at 4 yards of width, the burst
phase at 7 yards of depth. Phases are geometric, not guessed: they
come from the host description, which is what makes the derived
metrics defensible in the writeup.

Input: combine_tracking.csv. Output: one row per (player, attempt)
with phase metrics, plus a per player aggregate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WIDTH_ARC = 4.0   # dummies 4 yards from center
DEPTH_FINISH = 7.0  # finish cone 7 yards behind center


def _orient(track: pd.DataFrame) -> pd.DataFrame:
    """Rotate each attempt so +x is net travel direction, +y is right."""
    t = track.sort_values("time").reset_index(drop=True)
    dx, dy = t.x.iloc[-1] - t.x.iloc[0], t.y.iloc[-1] - t.y.iloc[0]
    length = float(np.hypot(dx, dy)) or 1.0
    ux, uy = dx / length, dy / length
    rx, ry = t.x - t.x.iloc[0], t.y - t.y.iloc[0]
    t["fx"] = ux * rx + uy * ry          # forward
    t["fy"] = -uy * rx + ux * ry         # lateral
    t["dt"] = t.time.diff().dt.total_seconds().fillna(0)
    t["t"] = t.dt.cumsum()
    return t


def extract_attempt(track: pd.DataFrame) -> dict | None:
    """Phase metrics for one PASS_RUSH_DRILL attempt."""
    t = _orient(track)
    if t.shape[0] < 15 or t.t.iloc[-1] < 1.0:
        return None

    total_time = float(t.t.iloc[-1])
    max_speed = float(t.s.max())

    # first step: displacement in the first 0.5 s of movement
    moving = t[t.s > 0.3]
    if moving.empty:
        return None
    t0 = float(moving.t.iloc[0])
    first_window = t[(t.t >= t0) & (t.t <= t0 + 0.5)]
    first_step = float(first_window.fx.iloc[-1] - first_window.fx.iloc[0]) if len(first_window) > 1 else 0.0
    early_accel = float(t[(t.t >= t0) & (t.t <= t0 + 1.0)].a.mean())

    # arc phase: frames where lateral excursion toward the dummy side peaks
    side = np.sign(t.fy.abs().max()) if t.fy.abs().max() > 0.1 else 1
    lateral = side * t.fy
    arc_mask = lateral >= WIDTH_ARC * 0.75
    arc_ratio = float(arc_mask.mean())
    # path length from frame positions; the supplied dis column has
    # inconsistent units across years, positions are ground truth
    step = np.hypot(t.fx.diff(), t.fy.diff()).fillna(0).sum()
    net = float(np.hypot(t.fx.iloc[-1], t.fy.iloc[-1]))
    path_efficiency = net / step if step > 0 else np.nan

    # bend: how tightly the arc is held, direction change rate while wide
    if arc_mask.sum() >= 3:
        arc = t[arc_mask]
        ddir = arc.dir.diff().abs()
        ddir = ddir.where(ddir <= 180).fillna(0)
        bend_rate = float((ddir / arc.dt.replace(0, np.nan)).mean())
        arc_speed = float(arc.s.mean())
    else:
        bend_rate, arc_speed = np.nan, np.nan

    # burst: speed at 4 yd and 5 yd depth, time to finish
    at_depth = lambda d: (t.fx - d).abs().idxmin()
    speed_4yd = float(t.loc[at_depth(4.0)].s) if t.fx.max() >= 4.0 else np.nan
    speed_5yd = float(t.loc[at_depth(5.0)].s) if t.fx.max() >= 5.0 else np.nan

    # acceleration profile
    peak_accel = float(t.a.max())
    accel_0_1 = early_accel
    decel_min = float(t.a.min())

    return {
        "total_time": round(total_time, 3),
        "max_speed": round(max_speed, 3),
        "first_step_yds_0p5s": round(first_step, 3),
        "accel_first_1s": round(accel_0_1, 3),
        "peak_accel": round(peak_accel, 3),
        "min_accel": round(decel_min, 3),
        "arc_width_ratio": round(arc_ratio, 3),
        "path_efficiency": round(path_efficiency, 3),
        "bend_rate_deg_per_s": round(bend_rate, 2) if bend_rate == bend_rate else np.nan,
        "arc_speed": round(arc_speed, 3) if arc_speed == arc_speed else np.nan,
        "speed_at_4yd": round(speed_4yd, 3) if speed_4yd == speed_4yd else np.nan,
        "speed_at_5yd": round(speed_5yd, 3) if speed_5yd == speed_5yd else np.nan,
        "n_frames": int(len(t)),
    }


def build_pass_rush_features(combine_tracking_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per attempt, per player aggregate) frames for PASS_RUSH_DRILL."""
    ct = pd.read_csv(
        combine_tracking_path,
        usecols=["draft_year", "event_id", "nfl_id", "entity_type", "time",
                 "drill_name", "attempt", "x", "y", "s", "a", "dis", "dir"],
        dtype={"event_id": str, "nfl_id": "Float64"},
        on_bad_lines="skip",
    )
    ct["time"] = pd.to_datetime(ct["time"], format="ISO8601", errors="coerce")
    ct = ct.dropna(subset=["time", "x", "y"])
    pr = ct[
        (ct.drill_name == "PASS_RUSH_DRILL") & (ct.entity_type == "PLAYER")
    ].dropna(subset=["nfl_id"])

    rows = []
    for (event_id, attempt, nfl_id), g in pr.groupby(["event_id", "attempt", "nfl_id"]):
        m = extract_attempt(g)
        if m is not None:
            rows.append({
                "event_id": event_id,
                "attempt": attempt,
                "nfl_id": int(nfl_id),
                "draft_year": int(g.draft_year.iloc[0]),
                **m,
            })
    per_attempt = pd.DataFrame(rows)
    if per_attempt.empty:
        return per_attempt, per_attempt

    agg = per_attempt.groupby("nfl_id").agg(
        n_attempts=("event_id", "count"),
        draft_year=("draft_year", "first"),
        **{c: (c, "mean") for c in per_attempt.columns
           if c not in ("event_id", "attempt", "nfl_id", "draft_year")},
    ).reset_index()
    return per_attempt, agg


if __name__ == "__main__":
    per_attempt, agg = build_pass_rush_features("combine_tracking.csv")
    print("attempts:", len(per_attempt), "players:", len(agg))
    print(agg.head(10).to_string())
    per_attempt.to_csv("pass_rush_attempts.csv", index=False)
    agg.to_csv("pass_rush_player_features.csv", index=False)
    print("saved pass_rush_attempts.csv and pass_rush_player_features.csv")
