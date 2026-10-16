import re
import sys

import pandas as pd

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

BRAND_FIXES = {
    "Bmw": "BMW",
    "Mg": "MG",
    "Byd": "BYD",
    "Mercedes Benz": "Mercedes-Benz",
    "Mclaren": "McLaren",
    "Rolls Royce": "Rolls-Royce",
}

FUEL_MAP = {
    "Petrol": "Petrol",
    "Diesel": "Diesel",
    "Electric": "Electric",
    "CNG": "CNG",
    "LPG": "LPG",
    "Hybrid (Electric + Petrol)": "Hybrid",
    "Mild Hybrid(Electric + Petrol)": "Mild Hybrid",
    "Mild Hybrid (Electric + Diesel)": "Mild Hybrid",
    "Plug-in Hybrid (Electric + Petrol)": "Plug-in Hybrid",
}

UNCLEAR_SUBTYPES = {"IMT"}


def clean_model_name(name: str) -> str:
    text = str(name)
    text = re.sub(r"\b(19|20)\d{2}\b", "", text)
    text = re.sub(r"\b\d+(ST|ND|RD|TH) Gen\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bOld Generation\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def is_old_generation(name: str) -> bool:
    text = str(name)
    has_old_label = re.search(r"\bOld Generation\b", text, flags=re.IGNORECASE) is not None
    has_year_range = re.search(r"(19|20)\d{2}\D*(19|20)\d{2}", text) is not None
    return has_old_label or has_year_range


def split_transmission(text):
    if not isinstance(text, str):
        return None, None
    head = text.split(",")[0].split(" - ")[0].strip()
    match = re.match(r"^(Manual|Automatic)(?: \(([\w-]+)\))?$", head)
    if not match:
        return None, None
    kind, subtype = match.group(1), match.group(2)
    if subtype and subtype.upper() in UNCLEAR_SUBTYPES:
        return None, None
    return kind, subtype


def clean_atharvanilawar(raw: pd.DataFrame, max_price: float):
    df = raw.drop_duplicates().copy()
    logger.info("Rows after dropping exact duplicates: %d", len(df))

    df = df.rename(
        columns={
            "Brand": "brand",
            "Car": "model_raw",
            "Variant": "variant",
            "Price": "ex_showroom_price",
            "Price_Type": "price_type",
            "Discontinued": "discontinued",
            "Fuel Type": "fuel_raw",
            "Mileage": "claimed_mileage",
            "Transmission": "transmission_raw",
            "Displacement (cc)": "engine_cc",
        }
    )
    df = df[
        [
            "brand", "model_raw", "variant", "ex_showroom_price", "price_type",
            "discontinued", "fuel_raw", "claimed_mileage", "transmission_raw", "engine_cc",
        ]
    ].copy()

    df["brand"] = df["brand"].str.strip().replace(BRAND_FIXES)
    df["variant"] = df["variant"].astype(str).str.strip()
    df["model"] = df["model_raw"].map(clean_model_name)
    df["old_generation"] = df["model_raw"].map(is_old_generation)
    df["on_sale"] = df["discontinued"].astype(str).str.lower().eq("no") & ~df["old_generation"]
    df["claimed_mileage"] = pd.to_numeric(df["claimed_mileage"], errors="coerce")
    df["engine_cc"] = pd.to_numeric(df["engine_cc"], errors="coerce")

    df = df[df["on_sale"]].copy()
    logger.info("Rows on sale (not discontinued, not old generation): %d", len(df))

    df = df[(df["price_type"] == "ex-showroom") & df["ex_showroom_price"].notna()].copy()
    logger.info("Rows with an ex-showroom price: %d", len(df))

    df = df[df["ex_showroom_price"] <= max_price].copy()
    logger.info("Rows priced at or under %s: %d", f"{max_price:,.0f}", len(df))

    df["fuel"] = df["fuel_raw"].map(FUEL_MAP)
    parts = df["transmission_raw"].map(split_transmission)
    df["transmission"] = parts.map(lambda p: p[0])
    df["transmission_subtype"] = parts.map(lambda p: p[1])

    key = ["brand", "model", "variant", "fuel", "transmission"]
    bad_fuel = df["fuel"].isna()
    bad_trans = df["transmission"].isna()
    bad_dupe = df.duplicated(key, keep=False) & ~(bad_fuel | bad_trans)
    bad = bad_fuel | bad_trans | bad_dupe

    reasons = pd.Series("", index=df.index)
    reasons[bad_fuel] += "unmapped fuel; "
    reasons[bad_trans] += "unmapped transmission; "
    reasons[bad_dupe] += "same variant under two model names, prices conflict; "

    rejected = df[bad].assign(reject_reason=reasons[bad])
    clean = df[~bad].copy()
    logger.info("Clean rows: %d, rejected rows: %d", len(clean), len(rejected))

    columns = [
        "brand", "model", "model_raw", "variant", "fuel", "transmission",
        "transmission_subtype", "ex_showroom_price", "claimed_mileage", "engine_cc",
    ]
    return clean[columns].reset_index(drop=True), rejected.reset_index(drop=True)


def main():
    try:
        cfg = load_config()
        raw = pd.read_csv(resolve_path(cfg["sources"]["atharvanilawar"]), low_memory=False)
        clean, rejected = clean_atharvanilawar(raw, cfg["scope"]["max_price"])

        interim = resolve_path(cfg["paths"]["interim_dir"])
        interim.mkdir(parents=True, exist_ok=True)
        clean.to_csv(interim / "atharvanilawar_clean.csv", index=False)
        rejected.to_csv(interim / "atharvanilawar_rejected.csv", index=False)
        logger.info("Saved cleaned file and rejects to %s", interim)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()