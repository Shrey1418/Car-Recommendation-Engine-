import sys

import pandas as pd

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

NEW_PROTOCOL_MIN_ADULT_MAX = 30
KEYS = ["our_brand", "our_model"]
SOURCE_RANK = {"Global NCAP": 0, "Bharat NCAP": 1}
PROTOCOL_LABEL = {"new": "new protocol", "old": "old protocol", "ais197": "AIS 197"}
TEST_COLUMNS = [
    "sheet", "source", "protocol", "rating_label", "year",
    "adult_score", "adult_max", "child_score", "child_max",
]


def load_map(path, sheet_column, source) -> pd.DataFrame:
    mapping = pd.read_csv(path, dtype="string").fillna("")
    if "note" not in mapping.columns:
        mapping["note"] = ""
    for column in ["our_brand", "our_model", "note"]:
        mapping[column] = mapping[column].str.strip()
    mapping = mapping[(mapping["our_brand"] != "") & (mapping["our_model"] != "")].copy()
    mapping["sibling"] = mapping["note"].str.lower().eq("sibling")
    mapping["source"] = source
    mapping = mapping.rename(columns={sheet_column: "sheet"})
    mapping = mapping[["source", "sheet", "our_brand", "our_model", "sibling"]]
    before = len(mapping)
    mapping = mapping.drop_duplicates(["source", "sheet", "our_brand", "our_model"])
    if len(mapping) < before:
        logger.warning("%s map: dropped %d duplicate rows", source, before - len(mapping))
    return mapping.reset_index(drop=True)


def load_global_tests(interim) -> pd.DataFrame:
    scores = pd.read_csv(interim / "ncap_scores.csv")
    scores = scores[scores["parse_status"].eq("ok")].copy()
    if scores["pdf_name"].duplicated().any():
        raise ValueError("ncap_scores.csv has duplicate pdf_name values")
    scores = scores.rename(columns={"pdf_name": "sheet"})
    scores["source"] = "Global NCAP"
    scores["rating_label"] = ""
    scores["protocol"] = scores["adult_max"].ge(NEW_PROTOCOL_MIN_ADULT_MAX).map({True: "new", False: "old"})
    return scores[TEST_COLUMNS]


def load_bharat_tests(interim) -> pd.DataFrame:
    scores = pd.read_csv(interim / "bncap_scores.csv")
    scores = scores[scores["parse_status"].eq("ok")].copy()
    if scores["detail_url"].duplicated().any():
        raise ValueError("bncap_scores.csv has duplicate detail_url values")
    scores = scores.rename(columns={"detail_url": "sheet", "test_year": "year"})
    scores["source"] = "Bharat NCAP"
    scores["rating_label"] = scores["rating_label"].fillna("")
    scores["protocol"] = "ais197"
    return scores[TEST_COLUMNS]


def add_percentages(tests: pd.DataFrame) -> pd.DataFrame:
    tests = tests.copy()
    tests["year"] = tests["year"].astype(int)
    tests["adult_pct"] = (tests["adult_score"] / tests["adult_max"] * 100).round(1)
    tests["child_pct"] = (tests["child_score"] / tests["child_max"] * 100).round(1)
    return tests


def pick_latest(joined: pd.DataFrame) -> pd.DataFrame:
    joined = joined.copy()
    joined["source_rank"] = joined["source"].map(SOURCE_RANK)
    joined["is_new"] = joined["protocol"].ne("old")
    ordered = joined.sort_values(
        KEYS + ["year", "source_rank", "is_new", "adult_pct"],
        ascending=[True, True, False, True, False, True],
    )
    chosen = ordered.groupby(KEYS).head(1).copy()

    same_keys = KEYS + ["year", "source", "protocol"]
    same = joined.merge(chosen[same_keys], on=same_keys)
    stats = (
        same.groupby(KEYS)
        .agg(sheets_in_latest=("sheet", "count"), adult_pct_best=("adult_pct", "max"))
        .reset_index()
    )
    chosen = chosen.merge(stats, on=KEYS)
    chosen["multiple_versions"] = chosen["sheets_in_latest"] > 1
    chosen["select_variants"] = chosen["rating_label"].ne("") & chosen["rating_label"].ne("Base Rating")
    chosen["safety_source"] = (
        chosen["source"]
        + " "
        + chosen["year"].astype(str)
        + " ("
        + chosen["protocol"].map(PROTOCOL_LABEL)
        + ")"
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

        tests = add_percentages(pd.concat([load_global_tests(interim), load_bharat_tests(interim)], ignore_index=True))
        mapping = pd.concat(
            [
                load_map(manual / "ncap_model_map.csv", "pdf_name", "Global NCAP"),
                load_map(manual / "bncap_model_map.csv", "detail_url", "Bharat NCAP"),
            ],
            ignore_index=True,
        )

        joined = mapping.merge(tests, on=["source", "sheet"], how="left", validate="many_to_one")
        missing = joined[joined["adult_pct"].isna()]
        if len(missing):
            raise ValueError(f"Map rows with no score (typo in sheet name?): {missing['sheet'].tolist()}")

        both = joined.groupby(KEYS)["source"].nunique().eq(2).sum()
        chosen = pick_latest(joined)
        variants = pd.read_csv(interim / "variants_validated.csv")
        check_against_catalog(chosen, variants)

        final = chosen.rename(columns={"our_brand": "brand", "our_model": "model", "year": "test_year"})
        final = final[
            ["brand", "model", "adult_pct", "child_pct", "source", "protocol", "test_year",
             "rating_label", "select_variants", "multiple_versions", "adult_pct_best",
             "sibling", "safety_source", "sheet"]
        ].sort_values(["brand", "model"])
        out = processed / "safety_ratings.csv"
        final.to_csv(out, index=False)

        logger.info("Models rated: %d, by source: %s", len(final), final["source"].value_counts().to_dict())
        logger.info("Models tested by both sources: %d", both)
        logger.info("Multiple versions tested: %d, sibling ratings: %d, select-variants ratings: %d",
                    final["multiple_versions"].sum(), final["sibling"].sum(), final["select_variants"].sum())
        logger.info("Saved %s", out)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()