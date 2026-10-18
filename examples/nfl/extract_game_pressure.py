"""In-game pass rush productivity with the host's NA semantics.

Missing label rules, from competition host answers in discussions 746678
and 748921:
  - quick_pressure NA with blitzing TRUE means no pressure occurred:
    an observed non event, counted in the denominator
  - quick_pressure FALSE means pressure occurred but slower than 2.5 s
  - player_get_off NA means the player never crossed the line of
    scrimmage: excluded from get-off averages, kept in denominators
  - missing pressure/hurry/sack values are no credited event, never
    treated as a negative outcome

Rush opportunity definition (documented approximation): a defender is
credited an opportunity on every non-nullified pass play where he was
on the field. For linebackers, restrict to blitzing == TRUE, since a
coverage LB did not rush. The supplied data has no explicit rush flag
for linemen, and the host confirmed in 750096 that blitzing NA for DL
does not remove rush eligibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_game_pressure(player_play_path: str, cohort_ids: np.ndarray,
                        front7_positions: dict[int, str]) -> pd.DataFrame:
    """Per player pressure productivity over rookie regular seasons."""
    usecols = None
    pp = pd.read_csv(player_play_path, low_memory=False)
    pp.columns = [c.lower() for c in pp.columns]
    id_col = "nfl_id" if "nfl_id" in pp.columns else "nflid"
    game_col = "game_id" if "game_id" in pp.columns else "gameid"
    play_col = "play_id" if "play_id" in pp.columns else "playid"

    need = [id_col, game_col, play_col, "pass_result", "sack", "time_to_pressure",
            "time_to_qb_hurry", "blitzing", "player_get_off"]
    if "play_nullified_by_penalty" in pp.columns:
        need.append("play_nullified_by_penalty")
    have = [c for c in need if c in pp.columns]
    pp = pp[have]

    pp = pp[pp[id_col].isin(cohort_ids)].copy()

    # pass plays only, penalties removed when the column exists
    is_pass = pp["pass_result"].notna()
    if "play_nullified_by_penalty" in pp.columns:
        is_pass &= pp["play_nullified_by_penalty"].fillna("N").astype(str).str.upper() != "Y"
    pp = pp[is_pass].copy()

    pp["is_dl"] = pp[id_col].map(lambda i: front7_positions.get(int(i), "interior") == "interior"
                                 if not np.isnan(i) else False)
    # opportunity: DL on a pass play; LB only when blitzing TRUE
    opp = pp["is_dl"] | (pp["blitzing"].astype(str).str.upper() == "TRUE")
    pp = pp[opp].copy()

    pp["pressure_event"] = (
        pp["time_to_pressure"].notna()
        | pp["time_to_qb_hurry"].notna()
        | (pp["sack"].fillna(0) == 1)
    ).astype(int)

    g = pp.groupby(id_col).agg(
        n_opportunities=(play_col, "count"),
        n_pressure=("pressure_event", "sum"),
        n_sacks=("sack", lambda s: int((s == 1).sum())),
        mean_get_off=("player_get_off", lambda s: s.dropna().mean() if s.notna().any() else np.nan),
        n_get_off_obs=("player_get_off", lambda s: int(s.notna().sum())),
    ).reset_index()
    g["pressure_rate"] = g.n_pressure / g.n_opportunities.clip(lower=1)
    g = g.rename(columns={id_col: "nfl_id"})
    g["nfl_id"] = g["nfl_id"].astype(int)
    return g


if __name__ == "__main__":
    cohort = pd.read_csv("pass_rush_dataset.csv")
    front7 = dict(zip(cohort.nfl_id,
                      cohort.nfl_position.map(lambda p: "edge" if p in ("DE", "OLB") else "interior")))
    out = build_game_pressure("player_play.csv", cohort.nfl_id.values, front7)
    out.to_csv("game_pressure.csv", index=False)
    print("players with game data:", len(out))
    print(out.describe().round(3).to_string())
