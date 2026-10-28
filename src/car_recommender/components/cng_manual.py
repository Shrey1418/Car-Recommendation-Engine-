import sys

import pandas as pd

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

KEYS = ["brand", "model", "variant", "fuel", "transmission"]
LABELS = {"official": "manual_official", "aggregator": "manual_aggregator"}
REQUIRED = [
    "brand", "model", "variant", "transmission", "ex_showroom_price", "claimed_mileage",
    "engine_cc", "cng_boot_space", "source_type", "source", "as_of_date",
]
PRICE = "ex_showroom_price"
OPTIONAL_NUMBERS = ["claimed_mileage", "engine_cc"]


def read_cng(path) -> pd.DataFrame:
    cng = pd.read_csv(path, dtype="string", encoding="utf-8-sig").fillna("")
    absent = [column for column in REQUIRED if column not in cng.columns]
    if absent:
        raise ValueError(f"cng_variants.csv is missing columns: {absent}")
    for column in REQUIRED:
        cng[column] = cng[column].str.strip()
    cng = cng[cng["variant"] != ""].copy()
    for column in [PRICE] + OPTIONAL_NUMBERS:
        text = cng[column]
        cng[column] = pd.to_numeric(text, errors="coerce")
        cng[column + "_typed_but_bad"] = (text != "") & cng[column].isna()
    return cng


def keep_usable(cng: pd.DataFrame) -> pd.DataFrame:
    problems = pd.Series("", index=cng.index)
    problems = problems.where(cng[PRICE].notna(), problems + "ex_showroom_price is missing or not a number; ")
    for column in OPTIONAL_NUMBERS:
        problems = problems.where(~cng[column + "_typed_but_bad"], problems + f"{column} is not a number; ")
    problems = problems.where(cng["transmission"].isin(["Manual", "Automatic"]), problems + "transmission must be Manual or Automatic; ")
    problems = problems.where(cng["source_type"].str.lower().isin(LABELS), problems + "source_type must be official or aggregator; ")
    for column in ["source", "as_of_date"]:
        problems = problems.where(cng[column] != "", problems + f"{column} is empty; ")
    for index, text in problems[problems != ""].items():
        logger.warning("CNG row skipped (%s | %s | %s): %s", cng.at[index, "brand"], cng.at[index, "model"], cng.at[index, "variant"], text)
    return cng[problems == ""].copy()


def attach_attributes(cng: pd.DataFrame, attributes: pd.DataFrame) -> pd.DataFrame:
    attributes = attributes[["brand", "model", "body_type", "seats", "on_sale"]].copy()
    for column in ["brand", "model"]:
        attributes[column] = attributes[column].astype("string").str.strip()
    is_off_sale = attributes["on_sale"].astype("string").str.strip().str.upper().eq("N").fillna(False)
    attributes["on_sale"] = ~is_off_sale.astype(bool)
    joined = cng.merge(attributes, on=["brand", "model"], how="left", indicator=True)
    for row in joined[joined["_merge"] == "left_only"].itertuples():
        logger.warning("CNG row skipped, add this model to model_attributes.csv first: %s | %s", row.brand, row.model)
    return joined[joined["_merge"] == "both"].drop(columns="_merge")


def petrol_engine_by_model(merged: pd.DataFrame) -> dict:
    petrol = merged[merged["fuel"].astype("string").str.lower() == "petrol"]
    values = petrol.groupby(["brand", "model"])["engine_cc"].agg(lambda s: set(s.dropna()))
    return {key: next(iter(found)) for key, found in values.items() if len(found) == 1}


def fill_engine(cng: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    engine_map = petrol_engine_by_model(merged)
    sibling = pd.Series(
        [engine_map.get((brand, model)) for brand, model in zip(cng["brand"], cng["model"])],
        index=cng.index,
        dtype="float",
    )
    copied = cng["engine_cc"].isna() & sibling.notna()
    cng = cng.copy()
    cng["engine_cc"] = cng["engine_cc"].where(cng["engine_cc"].notna(), sibling)
    cng["engine_note"] = ""
    cng.loc[copied, "engine_note"] = "; engine_cc copied from petrol sibling"
    for brand, model in cng.loc[cng["engine_cc"].isna(), ["brand", "model"]].drop_duplicates().itertuples(index=False):
        logger.warning("engine_cc left blank (petrol engines differ or unknown): %s | %s", brand, model)
    return cng


def build_rows(cng: pd.DataFrame) -> pd.DataFrame:
    label = cng["source_type"].str.lower().map(LABELS)
    rows = pd.DataFrame(
        {
            "brand": cng["brand"],
            "model": cng["model"],
            "variant": cng["variant"],
            "fuel": "CNG",
            "transmission": cng["transmission"],
            "ex_showroom_price": cng[PRICE],
            "engine_cc": cng["engine_cc"],
            "claimed_mileage": cng["claimed_mileage"],
            "mileage_source": label.where(cng["claimed_mileage"].notna(), "missing"),
            "body_type": cng["body_type"],
            "seats": cng["seats"].astype(float),
            "on_sale": cng["on_sale"],
            "spec_source": "manual CNG entry: " + cng["source"] + " (" + cng["as_of_date"] + ")" + cng["engine_note"],
        }
    )
    return rows.drop_duplicates(KEYS)


def add_to_merged(merged: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    seen = merged[KEYS].drop_duplicates().assign(_exists=True)
    checked = rows.merge(seen, on=KEYS, how="left")
    fresh = checked[checked["_exists"].isna()].drop(columns="_exists")
    skipped = len(rows) - len(fresh)
    if skipped:
        logger.info("CNG rows already in the merged file, skipped: %d", skipped)
    if "variant_id" in merged.columns:
        start = int(merged["variant_id"].max()) + 1
        fresh = fresh.assign(variant_id=range(start, start + len(fresh)))
    return pd.concat([merged, fresh], ignore_index=True), len(fresh)


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])
        manual = resolve_path(cfg["paths"]["manual_dir"])

        path = manual / "cng_variants.csv"
        if not path.exists():
            logger.warning("No CNG file at %s, nothing added", path)
            return

        merged = pd.read_csv(interim / "variants_merged.csv")
        attributes = pd.read_csv(manual / "model_attributes.csv", encoding="utf-8-sig")

        usable = keep_usable(read_cng(path))
        with_attributes = attach_attributes(usable, attributes)
        rows = build_rows(fill_engine(with_attributes, merged))

        combined, added = add_to_merged(merged, rows)
        combined.to_csv(interim / "variants_merged.csv", index=False)

        logger.info("CNG variants added: %d, total variants now: %d", added, len(combined))
        logger.info("CNG variants in file now: %d", int(combined["fuel"].eq("CNG").sum()))
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()