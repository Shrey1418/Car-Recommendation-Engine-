import sys
from pathlib import Path

import pandas as pd

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

APP_DATA = Path(__file__).resolve().parents[3] / "app_data"
VARIANT_COLUMNS = [
    "variant_id", "brand", "model", "variant", "fuel", "transmission", "ex_showroom_price",
    "engine_cc", "claimed_mileage", "seats", "on_sale", "ev_range_km", "battery_kwh",
]
SAFETY_COLUMNS = ["brand", "model", "adult_pct", "child_pct", "protocol", "safety_source"]


def main():
    try:
        cfg = load_config()
        processed = resolve_path(cfg["paths"]["processed_dir"])
        APP_DATA.mkdir(exist_ok=True)

        variants = pd.read_csv(processed / "variants_export.csv", encoding="utf-8-sig")[VARIANT_COLUMNS]
        safety = pd.read_csv(processed / "safety_ratings.csv", encoding="utf-8-sig")[SAFETY_COLUMNS]

        variants.to_csv(APP_DATA / "variants_app.csv", index=False)
        safety.to_csv(APP_DATA / "safety_app.csv", index=False)
        logger.info("Wrote %d variants and %d safety rows to %s", len(variants), len(safety), APP_DATA)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()