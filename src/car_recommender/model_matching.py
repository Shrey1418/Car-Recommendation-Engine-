import re
import sys

import pandas as pd
from rapidfuzz import fuzz, process

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

FUZZY_CUTOFF = 90


def model_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def most_common(series: pd.Series):
    values = series.dropna()
    return values.mode().iloc[0] if len(values) else None


def build_model_attributes(catalog: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    models = catalog[["brand", "model"]].drop_duplicates().reset_index(drop=True)
    models["key"] = models["model"].map(model_key)

    spec_models = specs.groupby(["brand", "model"], as_index=False).agg(
        body_type=("body_type", most_common),
        seats=("seats", most_common),
    )
    spec_models["key"] = spec_models["model"].map(model_key)

    rows = []
    for _, row in models.iterrows():
        pool = spec_models[spec_models["brand"] == row["brand"]].reset_index(drop=True)
        hit, match_type, score = None, "none", None

        exact = pool[pool["key"] == row["key"]]
        if len(exact):
            hit, match_type, score = exact.iloc[0], "exact", 100.0
        elif len(pool):
            best = process.extractOne(
                row["key"], pool["key"].tolist(), scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF
            )
            if best:
                hit, match_type, score = pool.iloc[best[2]], "fuzzy", round(best[1], 1)

        is_exact = match_type == "exact"
        is_fuzzy = match_type == "fuzzy"
        rows.append(
            {
                "brand": row["brand"],
                "model": row["model"],
                "body_type": hit["body_type"] if is_exact else None,
                "seats": hit["seats"] if is_exact else None,
                "match_type": match_type,
                "matched_specs_model": hit["model"] if hit is not None else None,
                "match_score": score,
                "suggested_body_type": hit["body_type"] if is_fuzzy else None,
                "suggested_seats": hit["seats"] if is_fuzzy else None,
            }
        )

    result = pd.DataFrame(rows)
    counts = result["match_type"].value_counts().to_dict()
    logger.info("Models: %d. Match results: %s", len(result), counts)
    missing = int((result["body_type"].isna() | result["seats"].isna()).sum())
    logger.warning("Models still missing body type or seats: %d (fill these by hand)", missing)
    return result


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])

        catalog = pd.read_csv(interim / "catalog_clean.csv")
        specs = pd.read_csv(interim / "specs_clean.csv")
        result = build_model_attributes(catalog, specs)

        out_path = interim / "model_attributes_prefill.csv"
        result.sort_values(["brand", "model"]).to_csv(out_path, index=False)
        logger.info("Saved %s", out_path)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()