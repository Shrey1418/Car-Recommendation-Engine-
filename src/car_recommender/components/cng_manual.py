import sys

import pandas as pd

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

SOURCES = [("cng_variants.csv", "CNG"), ("petrol_variants.csv", "Petrol"), ("diesel_variants.csv", "Diesel")]
KEYS = ["brand", "model", "variant", "fuel", "transmission"]
LABELS = {"official": "manual_official", "aggregator": "manual_aggregator"}
REQUIRED = [
    "brand", "model", "variant", "transmission", "ex_showroom_price", "claimed_mileage",
    "engine_cc", "source_type", "source", "as_of_date",
]
PRICE = "ex_showroom_price"
OPTIONAL_NUMBERS = ["claimed_mileage", "engine_cc"]


def read_manual(path, fuel: str) -> pd.DataFrame:
    manual = pd.read_csv(path, dtype="string", encoding="utf-8-sig").fillna("")
    absent = [column for column in REQUIRED if column not in manual.columns]
    if absent:
        raise ValueError(f"{path.name} is missing columns: {absent}")
    for column in REQUIRED:
        manual[column] = manual[column].str.strip()
    manual = manual[manual["variant"] != ""].copy()
    for column in [PRICE] + OPTIONAL_NUMBERS:
        text = manual[column]
        manual[column] = pd.to_numeric(text, errors="coerce")
        manual[column + "_typed_but_bad"] = (text != "") & manual[column].isna()
    manual["fuel"] = fuel
    return manual


def keep_usable(manual: pd.DataFrame) -> pd.DataFrame:
    problems = pd.Series("", index=manual.index)
    problems = problems.where(manual[PRICE].notna(), problems + "ex_showroom_price is missing or not a number; ")
    for column in OPTIONAL_NUMBERS:
        problems = problems.where(~manual[column + "_typed_but_bad"], problems + f"{column} is not a number; ")
    problems = problems.where(manual["transmission"].isin(["Manual", "Automatic"]), problems + "transmission must be Manual or Automatic; ")
    problems = problems.where(manual["source_type"].str.lower().isin(LABELS), problems + "source_type must be official or aggregator; ")
    for column in ["source", "as_of_date"]:
        problems = problems.where(manual[column] != "", problems + f"{column} is empty; ")
    for index, text in problems[problems != ""].items():
        logger.warning("Manual row skipped (%s | %s | %s): %s", manual.at[index, "brand"], manual.at[index, "model"], manual.at[index, "variant"], text)
    return manual[problems == ""].copy()


def attach_attributes(manual: pd.DataFrame, attributes: pd.DataFrame) -> pd.DataFrame:
    attributes = attributes[["brand", "model", "body_type", "seats", "on_sale"]].copy()
    for column in ["brand", "model"]:
        attributes[column] = attributes[column].astype("string").str.strip()
    is_off_sale = attributes["on_sale"].astype("string").str.strip().str.upper().eq("N").fillna(False)
    attributes["on_sale"] = ~is_off_sale.astype(bool)
    joined = manual.merge(attributes, on=["brand", "model"], how="left", indicator=True)
    for row in joined[joined["_merge"] == "left_only"].itertuples():
        logger.warning("Manual row skipped, add this model to model_attributes.csv first: %s | %s", row.brand, row.model)
    return joined[joined["_merge"] == "both"].drop(columns="_merge")


def petrol_engine_by_model(merged: pd.DataFrame) -> dict:
    petrol = merged[merged["fuel"].astype("string").str.lower() == "petrol"]
    values = petrol.groupby(["brand", "model"])["engine_cc"].agg(lambda s: set(s.dropna()))
    return {key: next(iter(found)) for key, found in values.items() if len(found) == 1}


def fill_engine(manual: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    engine_map = petrol_engine_by_model(merged)
    sibling = pd.Series(
        [engine_map.get((brand, model)) for brand, model in zip(manual["brand"], manual["model"])],
        index=manual.index,
        dtype="float",
    )
    copied = manual["engine_cc"].isna() & sibling.notna()
    manual = manual.copy()
    manual["engine_cc"] = manual["engine_cc"].where(manual["engine_cc"].notna(), sibling)
    manual["engine_note"] = ""
    manual.loc[copied, "engine_note"] = "; engine_cc copied from petrol sibling"
    for brand, model in manual.loc[manual["engine_cc"].isna(), ["brand", "model"]].drop_duplicates().itertuples(index=False):
        logger.warning("engine_cc left blank (petrol engines differ or unknown): %s | %s", brand, model)
    return manual


def build_rows(manual: pd.DataFrame) -> pd.DataFrame:
    label = manual["source_type"].str.lower().map(LABELS)
    rows = pd.DataFrame(
        {
            "brand": manual["brand"],
            "model": manual["model"],
            "variant": manual["variant"],
            "fuel": manual["fuel"],
            "transmission": manual["transmission"],
            "ex_showroom_price": manual[PRICE],
            "engine_cc": manual["engine_cc"],
            "claimed_mileage": manual["claimed_mileage"],
            "mileage_source": label.where(manual["claimed_mileage"].notna(), "missing"),
            "body_type": manual["body_type"],
            "seats": manual["seats"].astype(float),
            "on_sale": manual["on_sale"],
            "spec_source": "manual entry: " + manual["source"] + " (" + manual["as_of_date"] + ")" + manual["engine_note"],
        }
    )
    return rows.drop_duplicates(KEYS)


def update_existing(merged: pd.DataFrame, rows: pd.DataFrame) -> int:
    updated = 0
    for row in rows.itertuples(index=False):
        same_key = (
            merged["brand"].eq(row.brand)
            & merged["model"].eq(row.model)
            & merged["variant"].eq(row.variant)
            & merged["fuel"].eq(row.fuel)
            & merged["transmission"].eq(row.transmission)
        )
        old_source = merged["spec_source"].fillna("").astype(str)
        todo = same_key & ~old_source.str.contains(row.spec_source, regex=False)
        if not todo.any():
            continue
        merged.loc[todo, "ex_showroom_price"] = row.ex_showroom_price
        if pd.notna(row.engine_cc):
            merged.loc[todo, "engine_cc"] = row.engine_cc
        if pd.notna(row.claimed_mileage):
            merged.loc[todo, "claimed_mileage"] = row.claimed_mileage
            merged.loc[todo, "mileage_source"] = row.mileage_source
        merged.loc[todo, "spec_source"] = old_source[todo] + "; updated by " + row.spec_source
        updated += int(todo.sum())
    return updated


def add_to_merged(merged: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    seen = merged[KEYS].drop_duplicates().assign(_exists=True)
    checked = rows.merge(seen, on=KEYS, how="left")
    fresh = checked[checked["_exists"].isna()].drop(columns="_exists")
    if "variant_id" in merged.columns:
        start = int(merged["variant_id"].max()) + 1
        fresh = fresh.assign(variant_id=range(start, start + len(fresh)))
    return pd.concat([merged, fresh], ignore_index=True), len(fresh)


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])
        manual_dir = resolve_path(cfg["paths"]["manual_dir"])

        frames = []
        for file_name, fuel in SOURCES:
            path = manual_dir / file_name
            if not path.exists():
                logger.info("No %s, skipped", file_name)
                continue
            frames.append(keep_usable(read_manual(path, fuel)))
        if not frames:
            logger.warning("No manual variant files found, nothing added")
            return

        merged = pd.read_csv(interim / "variants_merged.csv")
        attributes = pd.read_csv(manual_dir / "model_attributes.csv", encoding="utf-8-sig")

        usable = pd.concat(frames, ignore_index=True)
        with_attributes = attach_attributes(usable, attributes)
        rows = build_rows(fill_engine(with_attributes, merged))

        updated = update_existing(merged, rows)
        combined, added = add_to_merged(merged, rows)
        combined.to_csv(interim / "variants_merged.csv", index=False)

        logger.info("Existing variants updated: %d", updated)
        logger.info("Manual variants added: %d, total variants now: %d", added, len(combined))
        logger.info("Variants by fuel now: %s", combined["fuel"].value_counts().to_dict())
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()