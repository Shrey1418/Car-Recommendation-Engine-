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

BODY_MAP = {
    "Hatchback": "Hatchback",
    "Sedan": "Sedan",
    "SUV": "SUV",
    "Crossover": "SUV",
    "MPV": "MPV",
    "MUV": "MPV",
    "Coupe": "Coupe",
    "Convertible": "Convertible",
    "Pick-up": "Pickup",
}

TYPE_MAP = {
    "Manual": ("Manual", None),
    "Automatic": ("Automatic", None),
    "AMT": ("Automatic", "AMT"),
    "DCT": ("Automatic", "DCT"),
    "CVT": ("Automatic", "CVT"),
}

SHREYR_FUEL_MAP = {
    "Petrol": "Petrol",
    "Diesel": "Diesel",
    "CNG": "CNG",
    "CNG + Petrol": "CNG",
    "Hybrid": "Hybrid",
    "Electric": "Electric",
}

KGM_TO_NM = 9.80665


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

def fix_make_and_model(make, model):
    make, model = str(make).strip(), str(model).strip()
    if make == "Maruti Suzuki R" and model == "Wagon":
        return "Maruti Suzuki", "Wagon R"
    if make == "Land Rover Rover":
        make = "Land Rover"
        if model.startswith("Range") and not model.startswith("Range Rover"):
            model = model.replace("Range", "Range Rover", 1)
    return BRAND_FIXES.get(make, make), model


def map_body_type(text):
    for token in str(text).split(","):
        token = token.strip()
        if token in BODY_MAP:
            return BODY_MAP[token]
    return None


def clean_shreyrmishra(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates().copy()
    logger.info("shreyrmishra rows after dropping exact duplicates: %d", len(df))

    df = df.rename(
        columns={
            "Make": "brand",
            "Model": "model",
            "Variant": "variant",
            "Displacement": "engine_cc",
            "Cylinders": "cylinders",
            "Fuel_Tank_Capacity": "fuel_tank_l",
            "Fuel_Type": "fuel_raw",
            "Height": "height_mm",
            "Length": "length_mm",
            "Width": "width_mm",
            "Body_Type": "body_raw",
            "Seating_Capacity": "seats",
            "Type": "type_raw",
            "Power.1": "power_ps",
            "Torque.1": "torque_nm",
        }
    )

    pairs = [fix_make_and_model(b, m) for b, m in zip(df["brand"], df["model"])]
    df["brand"] = [pair[0] for pair in pairs]
    df["model"] = [pair[1] for pair in pairs]
    df["variant"] = df["variant"].astype(str).str.strip()

    df["fuel"] = df["fuel_raw"].map(SHREYR_FUEL_MAP)
    df["body_type"] = df["body_raw"].map(map_body_type)
    parts = df["type_raw"].map(TYPE_MAP)
    df["transmission"] = parts.map(lambda p: p[0] if isinstance(p, tuple) else None)
    df["transmission_subtype"] = parts.map(lambda p: p[1] if isinstance(p, tuple) else None)

    is_kgm = (df["power_ps"] > 0) & (df["torque_nm"] > 0) & (df["torque_nm"] / df["power_ps"] < 0.6)
    df["torque_was_kgm"] = is_kgm
    df.loc[is_kgm, "torque_nm"] = (df.loc[is_kgm, "torque_nm"] * KGM_TO_NM).round(1)
    logger.info("Torque rows converted from kgm to Nm: %d", int(is_kgm.sum()))

    too_high = (df["power_ps"] > 0) & (df["torque_nm"] / df["power_ps"] > 6)
    df.loc[too_high, "torque_nm"] = None
    logger.warning("Torque set to unknown because it is impossibly high for the power: %d", int(too_high.sum()))

    zero_means_unknown = [
        "engine_cc", "cylinders", "fuel_tank_l", "height_mm", "length_mm",
        "width_mm", "seats", "power_ps", "torque_nm",
    ]
    df[zero_means_unknown] = df[zero_means_unknown].where(df[zero_means_unknown] != 0)

    for column in ["fuel", "body_type", "transmission"]:
        missing = int(df[column].isna().sum())
        if missing:
            logger.warning("shreyrmishra rows with unknown %s: %d", column, missing)

    key = ["brand", "model", "variant", "fuel", "transmission"]
    dupes = int(df.duplicated(key, keep=False).sum())
    if dupes:
        logger.warning("shreyrmishra rows sharing the same variant key: %d. Review them.", dupes)

    columns = [
        "brand", "model", "variant", "fuel", "transmission", "transmission_subtype",
        "body_type", "seats", "engine_cc", "cylinders", "power_ps", "torque_nm",
        "torque_was_kgm", "fuel_tank_l", "length_mm", "width_mm", "height_mm",
    ]
    return df[columns].reset_index(drop=True)


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])
        interim.mkdir(parents=True, exist_ok=True)

        raw_a = pd.read_csv(resolve_path(cfg["sources"]["atharvanilawar"]), low_memory=False)
        clean_a, rejected_a = clean_atharvanilawar(raw_a, cfg["scope"]["max_price"])
        clean_a.to_csv(interim / "atharvanilawar_clean.csv", index=False)
        rejected_a.to_csv(interim / "atharvanilawar_rejected.csv", index=False)

        raw_s = pd.read_csv(resolve_path(cfg["sources"]["shreyrmishra"]), low_memory=False)
        clean_s = clean_shreyrmishra(raw_s)
        clean_s.to_csv(interim / "shreyrmishra_clean.csv", index=False)

        logger.info("Saved cleaned files and rejects to %s", interim)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()