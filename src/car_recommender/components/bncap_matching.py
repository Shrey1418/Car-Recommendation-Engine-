import re
import sys

import pandas as pd

from car_recommender.components.ncap_matching import ascii_text, match_one
from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

DROP = {"gasoline", "petrol", "gen", "3rd", "4th", "5th", "new", "facelift"}


def clean_model(text) -> str:
    text = re.sub(r"\(.*?\)", " ", ascii_text(text))
    text = re.sub(r"[-.]", " ", text)
    words = [w for w in text.split() if w.lower() not in DROP and not re.fullmatch(r"20\d\d", w)]
    return " ".join(words)


def build_map(scores: pd.DataFrame, our_models: pd.DataFrame) -> pd.DataFrame:
    catalog_brands = set(our_models["brand"])
    rows = []
    for card in scores.itertuples():
        for part in str(card.model).split("/"):
            text = clean_model(part)
            if not text:
                continue
            model, match_type, suggestion = (None, "none", None)
            if card.brand in catalog_brands:
                model, match_type, suggestion = match_one(card.brand, text, our_models)
            rows.append(
                {
                    "detail_url": card.detail_url,
                    "title": card.title,
                    "rating_label": card.rating_label,
                    "test_year": card.test_year,
                    "bncap_brand": card.brand,
                    "bncap_model_text": text,
                    "our_brand": card.brand if model else None,
                    "our_model": model,
                    "match_type": match_type,
                    "suggested_model": suggestion,
                    "note": None,
                }
            )
    return pd.DataFrame(rows)


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])
        manual = resolve_path(cfg["paths"]["manual_dir"])

        scores = pd.read_csv(interim / "bncap_scores.csv")
        our_models = pd.read_csv(interim / "variants_validated.csv")[["brand", "model"]].drop_duplicates()

        mapping = build_map(scores, our_models)
        logger.info("BNCAP rows: %d, match types: %s", len(mapping), mapping["match_type"].value_counts().to_dict())

        mapping.to_csv(interim / "bncap_model_map_prefill.csv", index=False)
        target = manual / "bncap_model_map.csv"
        if target.exists():
            logger.info("Your edited file already exists, left untouched: %s", target)
        else:
            mapping.to_csv(target, index=False)
            logger.info("Created %s. Fill in the blank our_brand and our_model cells by hand", target)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()