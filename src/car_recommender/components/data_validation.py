import sys

import pandas as pd
import pandera.pandas as pa

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

FUELS = ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid", "Mild Hybrid", "Plug-in Hybrid"]
BODY_TYPES = ["Hatchback", "Sedan", "SUV", "MPV", "Coupe", "Convertible", "Pickup"]


def build_schema(max_price: float) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "brand": pa.Column(str, nullable=False),
            "model": pa.Column(str, nullable=False),
            "variant": pa.Column(str, nullable=False),
            "fuel": pa.Column(str, pa.Check.isin(FUELS)),
            "transmission": pa.Column(str, pa.Check.isin(["Manual", "Automatic"])),
            "ex_showroom_price": pa.Column(float, pa.Check.in_range(200_000, max_price)),
            "body_type": pa.Column(str, pa.Check.isin(BODY_TYPES), nullable=False),
            "seats": pa.Column(float, pa.Check.in_range(2, 10), nullable=False),
            "engine_cc": pa.Column(float, pa.Check.in_range(500, 7000), nullable=True),
            "claimed_mileage": pa.Column(float, pa.Check.in_range(5, 45), nullable=True),
            "power_ps": pa.Column(float, pa.Check.in_range(20, 900), nullable=True),
            "torque_nm": pa.Column(float, pa.Check.in_range(30, 1500), nullable=True),
            "ev_range_km": pa.Column(float, pa.Check.in_range(80, 900), nullable=True),
            "battery_kwh": pa.Column(float, pa.Check.in_range(5, 200), nullable=True),
        },
        checks=[
            pa.Check(
                lambda df: (df["fuel"] != "Electric") | df["ev_range_km"].notna(),
                error="electric variant has no range",
            ),
        ],
        strict=False,
    )


def validate(df: pd.DataFrame, max_price: float):
    schema = build_schema(max_price)
    try:
        schema.validate(df, lazy=True)
        failures = pd.DataFrame(columns=["index", "column", "check"])
    except pa.errors.SchemaErrors as err:
        failures = err.failure_cases

    failures = failures.dropna(subset=["index"]).copy()
    failures["index"] = failures["index"].astype(int)
    failures["reason"] = failures["column"].fillna("row").astype(str) + ": " + failures["check"].astype(str)
    reasons = failures.groupby("index")["reason"].agg("; ".join)

    bad = df.index.isin(reasons.index)
    rejected = df[bad].assign(reject_reason=reasons.reindex(df.index[bad]).values)
    return df[~bad].copy(), rejected


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])

        merged = pd.read_csv(interim / "variants_merged.csv")
        logger.info("Variants read: %d", len(merged))

        clean, rejected = validate(merged, cfg["scope"]["max_price"])
        logger.info("Passed validation: %d, rejected: %d", len(clean), len(rejected))

        if len(rejected):
            top = rejected["reject_reason"].value_counts().head(5)
            for reason, count in top.items():
                logger.warning("Reject reason (%d rows): %s", count, reason)

        clean.reset_index(drop=True).to_csv(interim / "variants_validated.csv", index=False)
        rejected.reset_index(drop=True).to_csv(interim / "variants_rejected.csv", index=False)
        logger.info("Saved variants_validated.csv and variants_rejected.csv to %s", interim)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()