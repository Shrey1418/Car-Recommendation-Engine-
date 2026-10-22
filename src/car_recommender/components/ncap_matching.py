import re
import sys
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

from car_recommender.components.model_matching import model_key
from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

FUZZY_CUTOFF = 85
DROP_WORDS = {"new", "facelift"}


def ascii_text(text) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()


def split_brand(text: str, brands: list):
    plain = ascii_text(text).strip()
    for brand in sorted(brands, key=len, reverse=True):
        if plain.lower().startswith(ascii_text(brand).lower()):
            return brand, plain[len(brand):].strip()
    return None, plain


def model_parts(title: str, tested_model: str) -> list:
    source = title if "/" in str(title) else tested_model
    source = re.sub(r"\(.*?\)", "", ascii_text(source))
    source = source.split(",")[0]
    return [part.strip() for part in source.split("/") if part.strip()]


def strip_words(text: str) -> str:
    words = [word for word in re.findall(r"[A-Za-z0-9]+", text) if word.lower() not in DROP_WORDS]
    return model_key(" ".join(words))


def match_one(brand, text, our_models):
    pool = our_models[our_models["brand"] == brand].reset_index(drop=True)
    pool_keys = pool["model"].map(model_key)

    hit = pool[pool_keys == model_key(text)]
    if len(hit):
        return hit.iloc[0]["model"], "exact", None

    hit = pool[pool_keys == strip_words(text)]
    if len(hit):
        return hit.iloc[0]["model"], "exact_after_dropping_new_or_facelift", None

    if len(pool):
        best = process.extractOne(
            model_key(text), pool_keys.tolist(), scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF
        )
        if best:
            return None, "fuzzy_suggestion", pool.iloc[best[2]]["model"]
    return None, "none", None


def build_map(scores: pd.DataFrame, our_models: pd.DataFrame) -> pd.DataFrame:
    brands = sorted(our_models["brand"].unique())
    rows = []
    for entry in scores.itertuples():
        brand_guess, remainder = split_brand(entry.tested_model, brands)
        for index, part in enumerate(model_parts(entry.title, entry.tested_model)):
            text = split_brand(part, brands)[1] if index == 0 else part
            model, match_type, suggestion = (None, "none", None)
            if brand_guess:
                model, match_type, suggestion = match_one(brand_guess, text, our_models)
            rows.append(
                {
                    "pdf_name": entry.pdf_name,
                    "ncap_title": entry.title,
                    "year": entry.year,
                    "adult_max": entry.adult_max,
                    "ncap_brand": brand_guess,
                    "ncap_model_text": text,
                    "our_brand": brand_guess if model else None,
                    "our_model": model,
                    "match_type": match_type,
                    "suggested_model": suggestion,
                }
            )
    return pd.DataFrame(rows)


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])
        manual = resolve_path(cfg["paths"]["manual_dir"])

        scores = pd.read_csv(interim / "ncap_scores.csv")
        our_models = pd.read_csv(interim / "variants_validated.csv")[["brand", "model"]].drop_duplicates()

        mapping = build_map(scores, our_models)
        logger.info("NCAP rows: %d, match types: %s", len(mapping), mapping["match_type"].value_counts().to_dict())

        prefill = interim / "ncap_model_map_prefill.csv"
        mapping.to_csv(prefill, index=False)
        target = manual / "ncap_model_map.csv"
        if target.exists():
            logger.info("Your edited file already exists, left untouched: %s", target)
        else:
            mapping.to_csv(target, index=False)
            logger.info("Created %s. Fill in the blank our_brand and our_model cells by hand", target)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()