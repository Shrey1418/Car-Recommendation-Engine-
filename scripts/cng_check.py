import re
from pathlib import Path

import pandas as pd

CNG_FILE = Path("data/manual/cng_variants.csv")
CATALOG_FILE = Path("data/processed/variants_export.csv")

REQUIRED = [
    "brand", "model", "variant", "transmission", "ex_showroom_price", "claimed_mileage",
    "engine_cc", "cng_boot_space", "source_type", "source", "as_of_date",
]
KEY = ["brand", "model", "variant", "transmission"]
PREMIUM_RANGE = (50000, 150000)
MILEAGE_RANGE = (15, 40)
NOISE_WORDS = r"\b(cng|zng|mt|amt|hy|duo)\b"


def to_number(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def simple_name(text: str) -> str:
    text = re.sub(r"[()\-]", " ", text.lower())
    text = re.sub(NOISE_WORDS, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def check_label_fields(row) -> list[str]:
    errors = []
    if row["transmission"] not in ["Manual", "Automatic"]:
        errors.append("transmission must be Manual or Automatic")
    if row["cng_boot_space"] not in ["usable", "reduced"]:
        errors.append("cng_boot_space must be usable or reduced")
    if row["source_type"] not in ["official", "aggregator"]:
        errors.append("source_type must be official or aggregator")
    for column in ["variant", "source", "as_of_date"]:
        if row[column] == "":
            errors.append(f"{column} is empty")
    return errors


def check_numbers(row) -> tuple[list[str], list[str], float | None]:
    errors, blanks = [], []
    price = to_number(row["ex_showroom_price"])
    if price is None or price <= 0:
        errors.append("ex_showroom_price must be a plain number")
        price = None

    mileage = to_number(row["claimed_mileage"])
    if row["claimed_mileage"] == "":
        blanks.append("mileage is blank (CNG running cost can't be computed)")
    elif mileage is None:
        errors.append("claimed_mileage is not a plain number")
    elif not MILEAGE_RANGE[0] <= mileage <= MILEAGE_RANGE[1]:
        errors.append(f"mileage {mileage} is outside {MILEAGE_RANGE[0]}-{MILEAGE_RANGE[1]} km/kg, recheck")
    return errors, blanks, price


def main():
    cng = pd.read_csv(CNG_FILE, dtype="string", encoding="utf-8-sig").fillna("")
    absent = [column for column in REQUIRED if column not in cng.columns]
    if absent:
        raise SystemExit(f"cng_variants.csv is missing columns: {absent}")
    for column in REQUIRED:
        cng[column] = cng[column].str.strip()
    cng = cng[cng["model"] != ""].reset_index(drop=True)

    catalog = pd.read_csv(CATALOG_FILE, dtype="string", encoding="utf-8-sig").fillna("")
    catalog["price_number"] = pd.to_numeric(catalog["ex_showroom_price"], errors="coerce")
    known_models = set(zip(catalog["brand"], catalog["model"]))
    petrol = catalog[catalog["fuel"].str.lower() == "petrol"].copy()
    petrol["simple"] = petrol["variant"].map(simple_name)
    petrol_price = {
        (r.brand, r.model, r.transmission, r.simple): r.price_number
        for r in petrol.drop_duplicates(["brand", "model", "transmission", "simple"]).itertuples()
    }

    duplicated = cng.duplicated(KEY, keep=False)

    premiums = []
    clean = with_blanks = with_errors = 0
    for number, row in cng.iterrows():
        label = f"line {number + 2}: {row['brand']} | {row['model']} | {row['variant']} | {row['transmission']}"
        errors = check_label_fields(row)
        if duplicated[number]:
            errors.append("duplicate row (same brand, model, variant, transmission)")
        number_errors, blanks, price = check_numbers(row)
        errors += number_errors

        if (row["brand"], row["model"]) not in known_models:
            errors.append("brand+model not in the catalog (check spelling)")
        elif price is not None:
            twin_key = (row["brand"], row["model"], row["transmission"], simple_name(row["variant"]))
            twin = petrol_price.get(twin_key)
            if twin is None or pd.isna(twin):
                print(f"NOTE   {label}: no petrol twin found, premium not checked")
            else:
                premium = price - twin
                premiums.append((row["brand"], row["model"], premium))
                if not PREMIUM_RANGE[0] <= premium <= PREMIUM_RANGE[1]:
                    errors.append(f"premium over petrol twin is {premium:,.0f} (CNG {price:,.0f} vs petrol {twin:,.0f}), recheck price basis")

        for text in errors:
            print(f"ERROR  {label}: {text}")
        for text in blanks:
            print(f"BLANK  {label}: {text}")
        if errors:
            with_errors += 1
        elif blanks:
            with_blanks += 1
        else:
            clean += 1

    if premiums:
        table = pd.DataFrame(premiums, columns=["brand", "model", "premium"])
        print("\nPremium over petrol twin, per model (rows with a twin):")
        print(table.groupby(["brand", "model"])["premium"].agg(["count", "min", "median", "max"]).round(0).to_string())

    print(f"\nRows: {len(cng)} | clean: {clean} | only blanks: {with_blanks} | with errors: {with_errors}")


if __name__ == "__main__":
    main()