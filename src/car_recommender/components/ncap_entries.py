import re
import sys
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup

from car_recommender.components.ncap_fetch import find_pdf_links
from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

ENTRY_PATTERN = re.compile(
    r"^(?P<title>.+?)\s+Tested model:\s*(?P<tested_model>.+?)\s+"
    r"Body type:\s*(?P<body_type>.+?)\s+Year of publication:\s*(?P<year>\d{4})"
)


def clean_text(node) -> str:
    return " ".join(node.get_text(" ", strip=True).split())


def entry_text_for(anchor):
    node = anchor.parent
    while node is not None:
        text = clean_text(node)
        if "Tested model:" in text:
            return text if text.count("Tested model:") == 1 else None
        node = node.parent
    return None


def parse_entries(html: str, base_url: str):
    links = find_pdf_links(html, base_url)
    rows, unparsed = [], []
    for href, anchor in links.items():
        text = entry_text_for(anchor)
        match = ENTRY_PATTERN.match(text) if text else None
        if not match:
            unparsed.append({"pdf_url": href, "raw_text": (text or "")[:300]})
            continue
        rows.append(
            {
                "title": match["title"],
                "tested_model": match["tested_model"],
                "body_type": match["body_type"],
                "year": int(match["year"]),
                "pdf_url": href,
                "pdf_name": urlparse(href).path.rsplit("/", 1)[-1],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(unparsed)


def main():
    try:
        cfg = load_config()
        url = cfg["scraping"]["ncap_results_url"]
        interim = resolve_path(cfg["paths"]["interim_dir"])
        interim.mkdir(parents=True, exist_ok=True)

        html = (resolve_path("data/raw/ncap") / "india_results.html").read_text(encoding="utf-8")
        entries, unparsed = parse_entries(html, url)

        logger.info("Entries parsed: %d, not parsed: %d", len(entries), len(unparsed))
        if len(entries):
            logger.info("Entries per year: %s", entries["year"].value_counts().sort_index().to_dict())
            logger.info("Duplicate PDF names: %d", int(entries["pdf_name"].duplicated().sum()))
        if len(unparsed):
            logger.warning("Entries that could not be read are saved for review")

        entries.to_csv(interim / "ncap_entries.csv", index=False)
        unparsed.to_csv(interim / "ncap_entries_unparsed.csv", index=False)
        logger.info("Saved ncap_entries.csv and ncap_entries_unparsed.csv to %s", interim)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()