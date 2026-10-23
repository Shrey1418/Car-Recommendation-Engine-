import re
import sys

import pandas as pd
from bs4 import BeautifulSoup

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

SCORE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(Adult|Child)", re.IGNORECASE)
YEAR = re.compile(r"(20\d{2})")
EXPECTED_MAX = {"adult": 32.0, "child": 49.0}
BRAND_ALIASES = {"maruti": "Maruti Suzuki"}


def clean(text):
    return " ".join(text.split())


def text_of(card, selector):
    node = card.select_one(selector)
    return clean(node.get_text(" ", strip=True)) if node else ""


def split_title(title, brands):
    known = {name.lower(): name for name in brands}
    known.update(BRAND_ALIASES)
    for key in sorted(known, key=len, reverse=True):
        if title.lower().startswith(key):
            rest = title[len(key):]
            if rest == "" or not rest[0].isalnum():
                return known[key], rest.lstrip(" –-").strip()
    parts = re.split(r"[\s–-]+", title, maxsplit=1)
    return (parts[0].title(), parts[1].strip()) if len(parts) == 2 else ("", title)


def parse_card(card, brands):
    title = text_of(card, "a.listingtitle")
    brand, model = split_title(title, brands)

    link = card.select_one("a.listingtitle")
    rating_text = " | ".join(clean(s.get_text(" ", strip=True)) for s in card.select("span.ratingtext"))
    rating_text = re.sub(r"(\d)\s+\.(\d)", r"\1.\2", rating_text)
    scores = {kind.lower(): (float(got), float(top)) for got, top, kind in SCORE.findall(rating_text)}

    validity = text_of(card, "div.vehicle_make")
    year = YEAR.search(validity)
    notes = " ".join(filter(None, [text_of(card, "div.vehicle_description"), text_of(card, "div.vehicle_declaimer")]))

    row = {
        "brand": brand,
        "model": model,
        "title": title,
        "rating_label": text_of(card, "div.Variants"),
        "validity_text": validity,
        "test_year": int(year.group(1)) if year else None,
        "adult_score": scores.get("adult", (None, None))[0],
        "adult_max": scores.get("adult", (None, None))[1],
        "child_score": scores.get("child", (None, None))[0],
        "child_max": scores.get("child", (None, None))[1],
        "notes": notes,
        "detail_url": link["href"] if link else "",
    }
    row["problem"] = check_row(row)
    row["parse_status"] = "ok" if not row["problem"] else "check_needed"
    return row


def check_row(row):
    problems = []
    for kind in ("adult", "child"):
        score, top = row[f"{kind}_score"], row[f"{kind}_max"]
        if score is None:
            problems.append(f"{kind} score missing")
        elif top != EXPECTED_MAX[kind]:
            problems.append(f"{kind} max is {top}")
        elif score > top:
            problems.append(f"{kind} score above max")
    if row["test_year"] is None:
        problems.append("no year in validity text")
    if not row["brand"] or not row["model"]:
        problems.append("could not split title into brand and model")
    return "; ".join(problems)


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])
        html = (resolve_path("data/raw/bncap") / "ratings.html").read_text(encoding="utf-8")
        brands = pd.read_csv(interim / "variants_validated.csv")["brand"].dropna().unique().tolist()

        cards = BeautifulSoup(html, "html.parser").select("li.vehicle")
        logger.info("Cards found: %d", len(cards))

        table = pd.DataFrame([parse_card(card, brands) for card in cards])
        interim.mkdir(parents=True, exist_ok=True)
        table.to_csv(interim / "bncap_scores.csv", index=False)

        logger.info("Parse status: %s", table["parse_status"].value_counts().to_dict())
        logger.info("Rating labels: %s", table["rating_label"].value_counts().to_dict())
        logger.info("Test years: %s", table["test_year"].value_counts().sort_index().to_dict())
        for row in table[table["parse_status"] != "ok"].itertuples():
            logger.warning("Check %s: %s", row.title, row.problem)
        logger.info("Saved %s", interim / "bncap_scores.csv")
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()