import re
import sys
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from car_recommender.components.ncap_fetch import fetch_cached
from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

ADULT_SCORE = re.compile(r"\d+\.\d+\s*/\s*32")


def find_vehicle_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    found = {}
    for anchor in soup.find_all("a", href=True):
        href = urljoin(base_url, anchor["href"])
        if urlparse(href).path.startswith("/vehicle/") and href not in found:
            found[href] = anchor
    return found


def card_text(anchor, levels=3):
    node = anchor
    for _ in range(levels):
        if node.parent is not None:
            node = node.parent
    return " ".join(node.get_text(" ", strip=True).split())[:500]


def main():
    try:
        cfg = load_config()
        scraping = cfg["scraping"]
        url = scraping["bncap_ratings_url"]
        cache_file = resolve_path("data/raw/bncap") / "ratings.html"

        html = fetch_cached(
            url, cache_file, scraping["cache_hours"], scraping["user_agent"],
            scraping["timeout_seconds"], scraping["request_delay_seconds"],
        )
        links = find_vehicle_links(html, url)
        logger.info("Unique vehicle links: %d (the sitemap lists 37)", len(links))
        logger.info("Adult scores found in the raw HTML: %d", len(ADULT_SCORE.findall(html)))

        for href, anchor in list(links.items())[:3]:
            logger.info("Link: %s", href)
            logger.info("Card text: %s", card_text(anchor))
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()