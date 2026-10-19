import re
import sys

import numpy as np
import pandas as pd

from car_recommender.components.model_matching import model_key
from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

SPEC_COLUMNS = ["power_ps", "torque_nm", "cylinders", "fuel_tank_l", "length_mm", "width_mm", "height_mm"]
STRIP_WORDS = {
    "mt", "at", "amt", "dct", "cvt", "tc", "imt", "ags", "cng", "petrol",
    "diesel", "hybrid", "ev", "electric", "dualtone", "dt",
}
ENGINE_TOLERANCE = 0.02
SPEC_SPREAD_LIMIT = 0.05
MILEAGE_SPREAD_LIMIT = 1.15


def variant_key(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return "".join(word for word in words if word not in STRIP_WORDS)


def engines_agree(left: pd.Series, right: pd.Series, fuel: pd.Series) -> pd.Series:
    close = (left - right).abs() <= ENGINE_TOLERANCE * np.maximum(left, right)
    return close | (fuel == "Electric")


def consistent(group: pd.Series):
    values = group.dropna()
    if values.empty:
        return np.nan
    if (values.max() - values.min()) > SPEC_SPREAD_LIMIT * values.max():
        return np.nan
    return values.median()


def attach_specs(catalog: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    catalog = catalog.copy()
    specs = specs.copy()
    for table in (catalog, specs):
        table["mkey"] = table["model"].map(model_key)
        table["vkey"] = table["variant"].map(variant_key)
    catalog["row_id"] = range(len(catalog))

    pairs = catalog.merge(
        specs, on=["brand", "mkey", "fuel", "transmission"], suffixes=("", "_s")
    )
    pairs = pairs[engines_agree(pairs["engine_cc"], pairs["engine_cc_s"], pairs["fuel"])]

    results = []
    for source, subset in (
        ("variant_match", pairs[pairs["vkey"] == pairs["vkey_s"]]),
        ("engine_match", pairs),
    ):
        grouped = subset.groupby("row_id")[SPEC_COLUMNS].agg(consistent)
        grouped = grouped.dropna(how="all")
        grouped["spec_source"] = source
        results.append(grouped)

    variant_rows, engine_rows = results
    engine_rows = engine_rows[~engine_rows.index.isin(variant_rows.index)]
    matched = pd.concat([variant_rows, engine_rows])

    merged = catalog.merge(matched, left_on="row_id", right_index=True, how="left")
    merged["spec_source"] = merged["spec_source"].fillna("none")
    logger.info("Spec match results: %s", merged["spec_source"].value_counts().to_dict())
    return merged.drop(columns=["mkey", "vkey", "row_id"])


def attach_model_attributes(merged: pd.DataFrame, attributes: pd.DataFrame) -> pd.DataFrame:
    attributes = attributes[["brand", "model", "body_type", "seats"]].copy()
    attributes["body_type"] = attributes["body_type"].astype("string").str.strip()
    attributes["seats"] = pd.to_numeric(attributes["seats"], errors="coerce")
    attributes = attributes.drop_duplicates(["brand", "model"])

    merged = merged.merge(attributes, on=["brand", "model"], how="left")
    missing = int((merged["body_type"].isna() | merged["seats"].isna()).sum())
    logger.warning("Variants still missing body type or seats: %d", missing)
    return merged


def fill_mileage(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    is_ev = merged["fuel"] == "Electric"
    key = ["brand", "model", "fuel", "transmission", "engine_cc"]

    grouped = merged.groupby(key, dropna=False)["claimed_mileage"]
    sibling = grouped.transform("median")
    spread = grouped.transform("max") / grouped.transform("min")

    can_fill = merged["claimed_mileage"].isna() & sibling.notna() & (spread <= MILEAGE_SPREAD_LIMIT) & ~is_ev

    merged["mileage_source"] = np.where(merged["claimed_mileage"].notna(), "listed", "missing")
    merged.loc[can_fill, "claimed_mileage"] = sibling[can_fill].round(2)
    merged.loc[can_fill, "mileage_source"] = "sibling_median"
    merged.loc[is_ev, "mileage_source"] = "not_applicable"

    logger.info("Mileage sources: %s", merged["mileage_source"].value_counts().to_dict())
    return merged


def parse_span(text) -> tuple:
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(text))]
    if not numbers:
        return np.nan, np.nan
    return min(numbers), max(numbers)


def attach_ev_ranges(merged: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["mkey"] = ev["Model"].map(model_key)
    ranges = ev["ARAI Claimed Range (km)"].map(parse_span)
    batteries = ev["Battery Capacity (kWh)"].map(parse_span)
    ev["ev_range_km"] = ranges.map(lambda p: p[0])
    ev["ev_range_high"] = ranges.map(lambda p: p[1])
    ev["battery_kwh"] = batteries.map(lambda p: p[0])
    ev["ev_range_basis"] = np.where(ev["ev_range_km"] == ev["ev_range_high"], "single_value", "low_of_span")
    ev = ev.rename(columns={"Brand": "brand"})[
        ["brand", "mkey", "ev_range_km", "battery_kwh", "ev_range_basis"]
    ].drop_duplicates(["brand", "mkey"])

    merged = merged.copy()
    merged["mkey"] = merged["model"].map(model_key)
    merged = merged.merge(ev, on=["brand", "mkey"], how="left").drop(columns="mkey")

    is_ev = merged["fuel"] == "Electric"
    no_range = merged[is_ev & merged["ev_range_km"].isna()]
    if len(no_range):
        models = sorted(set(no_range["brand"] + " " + no_range["model"]))
        logger.warning("EV variants without a range: %d. Models: %s", len(no_range), models)
    merged.loc[~is_ev, ["ev_range_km", "battery_kwh", "ev_range_basis"]] = np.nan
    return merged


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])

        catalog = pd.read_csv(interim / "catalog_clean.csv")
        specs = pd.read_csv(interim / "specs_clean.csv")
        attributes = pd.read_csv(resolve_path(cfg["sources"]["model_attributes"]), encoding="utf-8-sig")
        ev = pd.read_csv(resolve_path(cfg["sources"]["ev_ranges"]), encoding="utf-8-sig")

        merged = attach_specs(catalog, specs)
        merged = attach_model_attributes(merged, attributes)
        merged = fill_mileage(merged)
        merged = attach_ev_ranges(merged, ev)

        columns = [
            "brand", "model", "variant", "fuel", "transmission", "transmission_subtype",
            "ex_showroom_price", "engine_cc", "claimed_mileage", "mileage_source",
            "body_type", "seats", "power_ps", "torque_nm", "cylinders", "fuel_tank_l",
            "length_mm", "width_mm", "height_mm", "spec_source",
            "ev_range_km", "battery_kwh", "ev_range_basis",
        ]
        out_path = interim / "variants_merged.csv"
        merged[columns].to_csv(out_path, index=False)
        logger.info("Saved %d merged variants to %s", len(merged), out_path)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()