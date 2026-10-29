import sys

import numpy as np
import pandas as pd

from car_recommender.components.cost_model import number, read_assumptions
from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

ADULT_SHARE = 0.5
CRITERIA = {
    "total_cost": ("lower", "5-year cost"),
    "running_cost_per_km": ("lower", "running cost per km"),
    "safety_score": ("higher", "safety"),
}
PRESETS = {
    "balanced": {"total_cost": 0.35, "running_cost_per_km": 0.25, "safety_score": 0.40},
    "family_safety": {"total_cost": 0.20, "running_cost_per_km": 0.15, "safety_score": 0.65},
    "budget_saver": {"total_cost": 0.50, "running_cost_per_km": 0.35, "safety_score": 0.15},
}


def read_safety(path, ratio: float) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    base = raw["adult_pct"] * ADULT_SHARE + raw["child_pct"] * (1 - ADULT_SHARE)
    is_old = raw["protocol"].astype("string").str.strip().str.lower().eq("old").fillna(False).astype(bool)
    score = base.where(~is_old, base / ratio)
    safety = raw[["brand", "model", "safety_source"]].assign(safety_score=score.round(1))
    return safety.drop_duplicates(["brand", "model"])


def apply_filters(costs: pd.DataFrame, budget: float, min_seats: int, fuels: list | None, transmission: str | None, min_price: float = 0.0) -> pd.DataFrame:
    on_sale = pd.to_numeric(costs["on_sale"], errors="coerce").eq(1)
    keep = on_sale & costs["ex_showroom_price"].between(min_price, budget) & (costs["seats"] >= min_seats)
    if fuels:
        keep &= costs["fuel"].isin(fuels)
    if transmission:
        keep &= costs["transmission"].eq(transmission)
    return costs[keep].copy()


def topsis(values: pd.DataFrame, weights: pd.Series) -> pd.Series:
    length = np.sqrt((values ** 2).sum()).replace(0, 1)
    weighted = (values / length) * weights
    best = pd.Series({c: weighted[c].max() if CRITERIA[c][0] == "higher" else weighted[c].min() for c in weighted})
    worst = pd.Series({c: weighted[c].min() if CRITERIA[c][0] == "higher" else weighted[c].max() for c in weighted})
    to_best = np.sqrt(((weighted - best) ** 2).sum(axis=1))
    to_worst = np.sqrt(((weighted - worst) ** 2).sum(axis=1))
    total = (to_best + to_worst).replace(0, np.nan)
    return (to_worst / total).fillna(0.5)


def add_reasons(scored: pd.DataFrame, criteria: list) -> pd.Series:
    count = len(scored)
    ranks = pd.DataFrame(
        {c: scored[c].rank(ascending=CRITERIA[c][0] == "lower", method="min") for c in criteria}
    )
    reasons = []
    for index in scored.index:
        best_two = ranks.loc[index].sort_values().index[:2]
        reasons.append("; ".join(f"{CRITERIA[c][1]}: #{int(ranks.at[index, c])} of {count}" for c in best_two))
    return pd.Series(reasons, index=scored.index)


def rank_cars(shortlist: pd.DataFrame, weights: dict) -> pd.DataFrame:
    used = [c for c, w in weights.items() if w > 0]
    if "safety_score" in used:
        groups = [
            ("rated", shortlist[shortlist["safety_score"].notna()], used),
            ("not rated", shortlist[shortlist["safety_score"].isna()], [c for c in used if c != "safety_score"]),
        ]
    else:
        groups = [("all", shortlist, used)]

    results = []
    for name, rows, criteria in groups:
        if not criteria:
            continue
        rows = rows.dropna(subset=criteria)
        if rows.empty:
            continue
        group_weights = pd.Series({c: weights[c] for c in criteria})
        group_weights = group_weights / group_weights.sum()
        scored = rows.assign(
            score=topsis(rows[criteria], group_weights).round(3),
            group=name,
            ranked_on=", ".join(CRITERIA[c][1] for c in criteria),
        )
        scored["reasons"] = add_reasons(scored, criteria)
        results.append(scored)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def recommend(costs, safety, budget, min_seats, fuels, transmission, preset, min_price=0.0) -> dict:
    with_safety = costs.merge(safety, on=["brand", "model"], how="left")
    shortlist = apply_filters(with_safety, budget, min_seats, fuels, transmission, min_price)
    ranked = rank_cars(shortlist, PRESETS[preset])
    picks = {}
    for name in ["rated", "not rated", "all"]:
        part = ranked[ranked["group"] == name] if not ranked.empty else ranked
        if not part.empty:
            picks[name] = part.sort_values("score", ascending=False).drop_duplicates(["brand", "model"]).head(3)
    by_fuel = pd.DataFrame()
    if not ranked.empty:
        best = ranked[ranked["group"].isin(["rated", "all"])]
        by_fuel = best.sort_values("score", ascending=False).drop_duplicates("fuel").reset_index(drop=True)
    return {"shortlist_size": len(shortlist), "picks": picks, "by_fuel": by_fuel}


def main():
    try:
        cfg = load_config()
        manual = resolve_path(cfg["paths"]["manual_dir"])
        processed = resolve_path(cfg["paths"]["processed_dir"])

        ratio = number(read_assumptions(manual / "cost_assumptions.csv"), "new_vs_old_protocol_ratio")
        costs = pd.read_csv(processed / "running_costs_default.csv", encoding="utf-8-sig")
        safety = read_safety(processed / "safety_ratings.csv", ratio)

        columns = ["brand", "model", "variant", "fuel", "ex_showroom_price", "total_cost",
                   "running_cost_per_km", "safety_score", "score", "reasons"]
        for preset in PRESETS:
            result = recommend(costs, safety, 1500000, 5, None, None, preset)
            logger.info("Preset %s: shortlist of %d variants", preset, result["shortlist_size"])
            for group, picks in result["picks"].items():
                logger.info("Top picks (%s):\n%s", group, picks[columns].to_string(index=False))
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()