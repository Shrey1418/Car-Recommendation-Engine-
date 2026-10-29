import sys

import pandas as pd

from car_recommender.components.cost_model import (
    add_running_cost,
    number,
    pick_prices,
    read_assumptions,
    read_fuel_prices,
)
from car_recommender.components.scoring import read_safety, recommend
from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)


def broken_rules(pick, rules: dict) -> str:
    problems = []
    if not rules["price_low"] <= pick.ex_showroom_price <= rules["price_high"]:
        problems.append("price outside range")
    if pick.seats < rules["min_seats"]:
        problems.append("too few seats")
    if rules["fuels"] and pick.fuel not in rules["fuels"]:
        problems.append("fuel not allowed")
    if rules["transmission"] and pick.transmission != rules["transmission"]:
        problems.append("wrong transmission")
    if pd.to_numeric(pick.on_sale, errors="coerce") != 1:
        problems.append("not on sale")
    return "; ".join(problems) if problems else "ok"


def run_persona(persona: dict, variants, safety, price_table, assumptions) -> list:
    years = int(number(assumptions, "ownership_years"))
    growth = number(assumptions, "fuel_price_growth_pct_per_year")
    annual_km = float(persona["annual_km"])
    rules = {
        "price_low": float(persona["price_low_lakh"]) * 100000,
        "price_high": float(persona["price_high_lakh"]) * 100000,
        "min_seats": int(persona["min_seats"]),
        "fuels": [f for f in persona["fuels"].split("|") if f],
        "transmission": persona["transmission"] or None,
    }
    base = {"id": persona["id"], "name": persona["name"], "preset": persona["preset"], "expect": persona["expect"]}

    prices = pick_prices(price_table, persona["city"], assumptions)
    costs = add_running_cost(variants, prices, annual_km, years, growth)
    result = recommend(
        costs, safety, rules["price_high"], rules["min_seats"], rules["fuels"] or None,
        rules["transmission"], persona["preset"], rules["price_low"],
    )

    rows = []
    for group, picks in result["picks"].items():
        for rank, pick in enumerate(picks.itertuples(index=False), start=1):
            rows.append({
                **base,
                "shortlist": result["shortlist_size"],
                "group": group,
                "rank": rank,
                "car": f"{pick.brand} {pick.model} {pick.variant}",
                "fuel": pick.fuel,
                "transmission": pick.transmission,
                "price_lakh": round(pick.ex_showroom_price / 100000, 2),
                "five_year_cost_lakh": round(pick.total_cost / 100000, 2),
                "safety_score": pick.safety_score,
                "score": pick.score,
                "auto_check": broken_rules(pick, rules),
                "your_verdict": "",
            })
    if not rows:
        rows.append({**base, "shortlist": result["shortlist_size"], "group": "none", "car": "NO RESULT", "auto_check": "ok", "your_verdict": ""})
    return rows


def main():
    try:
        cfg = load_config()
        manual = resolve_path(cfg["paths"]["manual_dir"])
        processed = resolve_path(cfg["paths"]["processed_dir"])

        assumptions = read_assumptions(manual / "cost_assumptions.csv")
        ratio = number(assumptions, "new_vs_old_protocol_ratio")
        variants = pd.read_csv(processed / "variants_export.csv", encoding="utf-8-sig")
        safety = read_safety(processed / "safety_ratings.csv", ratio)
        price_table = read_fuel_prices(manual / "fuel_prices_india.csv")
        personas = pd.read_csv(manual / "personas.csv", dtype="string", encoding="utf-8-sig").fillna("")

        rows = []
        for persona in personas.to_dict("records"):
            try:
                rows.extend(run_persona(persona, variants, safety, price_table, assumptions))
            except ValueError as error:
                logger.warning("Persona %s skipped: %s", persona["id"], error)
                rows.append({"id": persona["id"], "name": persona["name"], "car": "SKIPPED", "auto_check": str(error)})

        results = pd.DataFrame(rows)
        results.to_csv(processed / "persona_results.csv", index=False)

        failed = results[~results["auto_check"].isin(["ok"])]
        logger.info("Personas run: %d, result rows: %d", len(personas), len(results))
        logger.info("Rows that broke a filter rule: %d", len(failed))
        logger.info("Personas with no result: %s", results[results["car"] == "NO RESULT"]["id"].tolist())
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()