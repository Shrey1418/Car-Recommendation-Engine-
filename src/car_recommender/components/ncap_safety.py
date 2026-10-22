import sys

import pandas as pd

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

NEW_PROTOCOL_MIN_ADULT_MAX = 30
KEYS = ["our_brand", "our_model"]


def load_map(path) -> pd.DataFrame:
    mapping = pd.read_csv(path, dtype="string").fillna("")
    for column in ["our_brand", "our_model", "note"]:
        mapping[column] = mapping[column].str.strip()
    mapping = mapping[(mapping["our_brand"] != "") & (mapping["our_model"] != "")].copy()
    mapping["sibling"] = mapping["note"].str.lower().eq("sibling")
    mapping = mapping[["pdf_name", "our_brand", "our_model", "sibling"]]
    before = len(mapping)
    mapping = mapping.drop_duplicates(["pdf_name", "our_brand", "our_model"])
    if len(mapping) < before:
        logger.warning("Dropped %d duplicate map rows", before - len(mapping))
    return mapping.reset_index(drop=True)


def add_percentages(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    scores["adult_pct"] = (scores["adult_score"] / scores["adult_max"] * 100).round(1)
    scores["child_pct"] = (scores["child_score"] / scores["child_max"] * 100).round(1)
    scores["protocol"] = scores["adult_max"].ge(NEW_PROTOCOL_MIN_ADULT_MAX).map({True: "new", False: "old"})
    return scores


def pick_latest(joined: pd.DataFrame) -> pd.DataFrame:
    joined = joined.copy()
    joined["is_new"] = joined["protocol"].eq("new")
    ordered = joined.sort_values(
        KEYS + ["year", "is_new", "adult_pct"],
        ascending=[True, True, False, False, True],
    )
    chosen = ordered.groupby(KEYS).head(1).copy()

    same = joined.merge(chosen[KEYS + ["year", "protocol"]], on=KEYS + ["year", "protocol"])
    stats = (
        same.groupby(KEYS)
        .agg(sheets_in_latest=("pdf_name", "count"), adult_pct_best=("adult_pct", "max"))
        .reset_index()
    )
    chosen = chosen.merge(stats, on=KEYS)
    chosen["multiple_versions"] = chosen["sheets_in_latest"] > 1
    chosen["safety_source"] = (
        "Global NCAP "
        + chosen["year"].astype(str)
        + " ("
        + chosen["protocol"].astype(str)
        + " protocol)"
        + chosen["sibling"].map({True: ", sibling version", False: ""})
    )
    return chosen


def check_against_catalog(chosen: pd.DataFrame, variants: pd.DataFrame) -> None:
    catalog = variants.groupby(["brand", "model"])["on_sale"].any().reset_index()

    found = chosen[KEYS].merge(
        catalog, left_on=KEYS, right_on=["brand", "model"], how="left", indicator=True
    )
    bad = found[found["_merge"] == "left_only"]
    for row in bad.itertuples():
        logger.warning("Map names a car not in your catalog (typo?): %s | %s", row.our_brand, row.our_model)

    rated = catalog.merge(
        chosen[KEYS], left_on=["brand", "model"], right_on=KEYS, how="left", indicator=True
    )
    rated["has_rating"] = rated["_merge"].eq("both")
    on_sale = rated[rated["on_sale"]]
    logger.info(
        "Coverage: all models %d of %d, on-sale models %d of %d",
        rated["has_rating"].sum(), len(rated), on_sale["has_rating"].sum(), len(on_sale),
    )


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])
        manual = resolve_path(cfg["paths"]["manual_dir"])
        processed = resolve_path(cfg["paths"]["processed_dir"])

        scores = pd.read_csv(interim / "ncap_scores.csv")
        scores = scores[scores["parse_status"].eq("ok")]
        if scores["pdf_name"].duplicated().any():
            raise ValueError("ncap_scores.csv has duplicate pdf_name values")
        scores = add_percentages(
            scores[["pdf_name", "year", "adult_score", "adult_max", "child_score", "child_max"]]
        )

        mapping = load_map(manual / "ncap_model_map.csv")
        joined = mapping.merge(scores, on="pdf_name", how="left", validate="many_to_one")
        missing = joined[joined["adult_pct"].isna()]
        if len(missing):
            raise ValueError(f"Map rows with no score (pdf_name typo?): {missing['pdf_name'].tolist()}")

        chosen = pick_latest(joined)
        variants = pd.read_csv(interim / "variants_validated.csv")
        check_against_catalog(chosen, variants)

        final = chosen.rename(columns={"our_brand": "brand", "our_model": "model", "year": "test_year"})
        final = final[
            ["brand", "model", "adult_pct", "child_pct", "protocol", "test_year",
             "multiple_versions", "adult_pct_best", "sibling", "safety_source", "pdf_name"]
        ].sort_values(["brand", "model"])
        out = processed / "safety_ratings.csv"
        final.to_csv(out, index=False)

        logger.info("Models rated: %d, protocol: %s", len(final), final["protocol"].value_counts().to_dict())
        logger.info("Multiple versions tested: %d, sibling ratings: %d",
                    final["multiple_versions"].sum(), final["sibling"].sum())
        logger.info("Saved %s", out)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()