import sqlite3
import sys

import pandas as pd

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])
        processed = resolve_path(cfg["paths"]["processed_dir"])
        processed.mkdir(parents=True, exist_ok=True)

        variants = pd.read_csv(interim / "variants_validated.csv")
        variants["seats"] = variants["seats"].astype(int)
        logger.info("Validated variants read: %d", len(variants))

        schema = resolve_path("sql/schema.sql").read_text(encoding="utf-8")
        db_path = processed / "car_data.db"

        with sqlite3.connect(db_path) as conn:
            conn.executescript(schema)
            variants.to_sql("variants", conn, if_exists="append", index=False)
            stored = conn.execute("SELECT COUNT(*) FROM variants").fetchone()[0]

            by_fuel = conn.execute(
                "SELECT fuel, COUNT(*) FROM variants GROUP BY fuel ORDER BY 2 DESC"
            ).fetchall()
            export = pd.read_sql_query("SELECT * FROM variants ORDER BY variant_id", conn)

        if stored != len(variants):
            raise ValueError(f"Row count mismatch: file has {len(variants)}, database has {stored}")

        logger.info("Rows in database: %d", stored)
        logger.info("Variants by fuel: %s", dict(by_fuel))

        export.to_csv(processed / "variants_export.csv", index=False)
        logger.info("Saved %s and variants_export.csv", db_path)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()