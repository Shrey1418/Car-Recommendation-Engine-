import re
import sys

import numpy as np
import pandas as pd

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CITY = "Delhi"
NOISE_WORDS = r"\b(cng|zng|mt|amt|hy|duo)\b"
NOTE = "ex-showroom price + fuel only; insurance, service and registration not included"


def read_assumptions(path) -> dict:
    table = pd.read_csv(path, dtype="string", encoding="utf-8-sig").fillna("")
    names = table.iloc[:, 0].str.strip()
    values = table.iloc[:, 1].str.strip()
    return dict(zip(names, values))


def number(assumptions: dict, name: str) -> float:
    text = assumptions.get(name, "")
    if text == "":
        raise ValueError(f"cost_assumptions.csv has no value for '{name}'")
    return float(text)


def read_fuel_prices(path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    find = lambda word: next(column for column in raw.columns if word in column.lower())
    city_column = next(column for column in raw.columns if column.strip().lower() == "city")
    return pd.DataFrame(
        {
            "city": raw[city_column].astype("string").str.strip(),
            "petrol": pd.to_numeric(raw[find("petrol")], errors="coerce"),
            "diesel": pd.to_numeric(raw[find("diesel")], errors="coerce"),
            "cng": pd.to_numeric(raw[find("cng")], errors="coerce"),
        }
    )


def pick_prices(table: pd.DataFrame, city: str, assumptions: dict, overrides: dict | None = None) -> dict:
    match = table[table["city"].str.lower() == city.lower()]
    if match.empty:
        raise ValueError(f"City '{city}' not in fuel_prices_india.csv. Available: {sorted(table['city'])}")
    prices = match.iloc[0][["petrol", "diesel", "cng"]].to_dict()
    home = number(assumptions, "home_ev_charging_price")
    public = number(assumptions, "public_fast_charging_price")
    share = number(assumptions, "ev_home_charging_share")
    prices["electric"] = share * home + (1 - share) * public
    for fuel, value in (overrides or {}).items():
        if value is not None:
            prices[fuel] = float(value)
    return prices


def growth_factor(years: int, growth_pct: float) -> float:
    return sum((1 + growth_pct / 100) ** year for year in range(years))


def add_running_cost(variants: pd.DataFrame, prices: dict, annual_km: float, years: int, growth_pct: float) -> pd.DataFrame:
    costs = variants.copy()
    fuel = costs["fuel"]
    price = pd.Series(
        np.select(
            [fuel.isin(["Petrol", "Hybrid"]), fuel.eq("Diesel"), fuel.eq("CNG"), fuel.eq("Electric")],
            [prices["petrol"], prices["diesel"], prices["cng"], prices["electric"]],
            default=np.nan,
        ),
        index=costs.index,
    )
    mileage = costs["claimed_mileage"].where(costs["claimed_mileage"] > 0)
    kwh_per_km = costs["battery_kwh"] / costs["ev_range_km"].where(costs["ev_range_km"] > 0)

    per_km = pd.Series(np.nan, index=costs.index)
    by_distance = fuel.isin(["Petrol", "Hybrid", "Diesel", "CNG"])
    per_km[by_distance] = price[by_distance] / mileage[by_distance]
    electric = fuel.eq("Electric")
    per_km[electric] = kwh_per_km[electric] * price[electric]

    costs["fuel_price_used"] = price
    costs["running_cost_per_km"] = per_km
    costs["fuel_cost_total"] = per_km * annual_km * growth_factor(years, growth_pct)
    costs["total_cost"] = costs["ex_showroom_price"] + costs["fuel_cost_total"]
    costs["cost_note"] = NOTE
    return costs


def simple_name(text: str) -> str:
    text = re.sub(r"[()\-]", " ", str(text).lower())
    text = re.sub(NOISE_WORDS, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def same_variant_pairs(costs: pd.DataFrame) -> pd.DataFrame:
    keys = ["brand", "model", "transmission", "name_key"]
    petrol = costs[costs["fuel"] == "Petrol"].assign(name_key=lambda d: d["variant"].map(simple_name))
    alt = costs[costs["fuel"].isin(["Diesel", "CNG", "Hybrid"])].assign(name_key=lambda d: d["variant"].map(simple_name))
    pairs = alt.add_suffix("_alt").merge(
        petrol.drop_duplicates(keys).add_suffix("_petrol"),
        left_on=[key + "_alt" for key in keys],
        right_on=[key + "_petrol" for key in keys],
    )
    pairs["comparison_basis"] = "same variant, same transmission"
    return pairs


def electric_pairs(costs: pd.DataFrame, families: pd.DataFrame) -> pd.DataFrame:
    electric = costs[costs["fuel"] == "Electric"].add_suffix("_alt")
    petrol = costs[costs["fuel"] == "Petrol"].add_suffix("_petrol")
    linked = electric.merge(
        families[["brand", "ev_model", "ice_model"]],
        left_on=["brand_alt", "model_alt"],
        right_on=["brand", "ev_model"],
    )
    candidates = linked.merge(petrol, left_on=["brand", "ice_model"], right_on=["brand_petrol", "model_petrol"])
    candidates["gap"] = (candidates["ex_showroom_price_alt"] - candidates["ex_showroom_price_petrol"]).abs()
    best = candidates.sort_values("gap").groupby("variant_id_alt", as_index=False).head(1)
    best = best.assign(comparison_basis="nearest-price petrol variant of the sibling model")
    return best.drop(columns=["gap", "brand", "ev_model", "ice_model"])


def verdict(row, years: int) -> str:
    extra, saving, net = row["extra_price"], row["saving_per_km"], row["five_year_net_saving"]
    if pd.isna(extra) or pd.isna(saving):
        return "not enough data to compare (missing mileage, range or price)"
    if extra <= 0 and saving >= 0:
        return "cheaper to buy and cheaper to run"
    if extra > 0 and saving <= 0:
        return "costs more to buy and more to run"
    if extra <= 0 and saving < 0:
        return f"cheaper to buy by Rs {-extra:,.0f} but costs more to run; net {net:,.0f} over {years} years"
    months = row["breakeven_months"]
    if months <= years * 12:
        return f"pays back in {months:.0f} months; saves Rs {net:,.0f} over {years} years"
    return f"does not pay back within {years} years (needs {months / 12:.1f} years)"


def compare_fuels(costs: pd.DataFrame, families: pd.DataFrame, annual_km: float, years: int) -> pd.DataFrame:
    pairs = pd.concat([same_variant_pairs(costs), electric_pairs(costs, families)], ignore_index=True)
    pairs["extra_price"] = pairs["ex_showroom_price_alt"] - pairs["ex_showroom_price_petrol"]
    pairs["saving_per_km"] = pairs["running_cost_per_km_petrol"] - pairs["running_cost_per_km_alt"]
    pairs["five_year_net_saving"] = (pairs["fuel_cost_total_petrol"] - pairs["fuel_cost_total_alt"]) - pairs["extra_price"]
    usable_saving = pairs["saving_per_km"].where(pairs["saving_per_km"] > 0)
    pairs["breakeven_km"] = np.where(pairs["extra_price"] <= 0, 0.0, pairs["extra_price"] / usable_saving)
    pairs["breakeven_months"] = np.ceil(pairs["breakeven_km"] / annual_km * 12)
    pairs["verdict"] = pairs.apply(verdict, axis=1, years=years)

    for column in ["extra_price", "breakeven_km", "breakeven_months", "five_year_net_saving"]:
        pairs[column] = pairs[column].round(0).astype("Int64")
    pairs["saving_per_km"] = pairs["saving_per_km"].round(2)

    columns = [
        "brand_alt", "model_alt", "variant_alt", "fuel_alt", "transmission_alt",
        "model_petrol", "variant_petrol", "extra_price", "saving_per_km",
        "breakeven_km", "breakeven_months", "five_year_net_saving", "comparison_basis", "verdict",
    ]
    return pairs[columns].rename(columns={"brand_alt": "brand"})


def main():
    try:
        cfg = load_config()
        manual = resolve_path(cfg["paths"]["manual_dir"])
        processed = resolve_path(cfg["paths"]["processed_dir"])

        assumptions = read_assumptions(manual / "cost_assumptions.csv")
        annual_km = number(assumptions, "annual_km_default")
        years = int(number(assumptions, "ownership_years"))
        growth = number(assumptions, "fuel_price_growth_pct_per_year")

        prices = pick_prices(read_fuel_prices(manual / "fuel_prices_india.csv"), DEFAULT_CITY, assumptions)
        variants = pd.read_csv(processed / "variants_export.csv", encoding="utf-8-sig")
        families = pd.read_csv(manual / "ev_families.csv", encoding="utf-8-sig")

        costs = add_running_cost(variants, prices, annual_km, years, growth)
        comparisons = compare_fuels(costs, families, annual_km, years)

        costs.to_csv(processed / "running_costs_default.csv", index=False)
        comparisons.to_csv(processed / "fuel_comparisons_default.csv", index=False)

        logger.info("City %s, prices %s, %s km a year, %d years, growth %s%%", DEFAULT_CITY, prices, annual_km, years, growth)
        logger.info("Variants with a running cost: %d of %d", int(costs["running_cost_per_km"].notna().sum()), len(costs))
        unknown = costs[costs["running_cost_per_km"].isna()]["fuel"].value_counts().to_dict()
        logger.info("Variants without a running cost, by fuel: %s", unknown)
        logger.info("Fuel comparisons built: %d", len(comparisons))
        logger.info("By alternative fuel: %s", comparisons["fuel_alt"].value_counts().to_dict())
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()