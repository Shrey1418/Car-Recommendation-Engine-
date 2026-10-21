import sys
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)


def allowed_by_robots(url, user_agent, timeout):
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    response = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=timeout)
    if response.status_code in (401, 403):
        return False
    if response.status_code >= 400:
        return True
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch(user_agent, url)


def fetch_cached(url, cache_file, cache_hours, user_agent, timeout, delay):
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < cache_hours:
            logger.info("Using cached page (%.1f hours old): %s", age_hours, cache_file)
            return cache_file.read_text(encoding="utf-8")

    if not allowed_by_robots(url, user_agent, timeout):
        raise PermissionError(f"robots.txt does not allow fetching {url}")

    time.sleep(delay)
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    response.raise_for_status()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(response.text, encoding="utf-8")
    logger.info("Fetched %s (%d bytes) and saved to %s", url, len(response.text), cache_file)
    return response.text


def find_pdf_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    seen = {}
    for anchor in soup.find_all("a", href=True):
        href = urljoin(base_url, anchor["href"])
        if urlparse(href).path.lower().endswith(".pdf") and href not in seen:
            seen[href] = anchor
    return seen


def describe(anchor, levels=3):
    lines = []
    node = anchor
    for _ in range(levels):
        node = node.parent
        if node is None:
            break
        text = " ".join(node.get_text(" ", strip=True).split())[:150]
        lines.append(f"<{node.name}> {text}")
    return lines


def main():
    try:
        cfg = load_config()
        scraping = cfg["scraping"]
        url = scraping["ncap_results_url"]
        cache_file = resolve_path("data/raw/ncap") / "india_results.html"

        html = fetch_cached(
            url, cache_file, scraping["cache_hours"], scraping["user_agent"],
            scraping["timeout_seconds"], scraping["request_delay_seconds"],
        )
        links = find_pdf_links(html, url)
        logger.info("Unique PDF links found: %d", len(links))

        for href, anchor in list(links.items())[:3]:
            logger.info("Link: %s", href)
            logger.info("Link text: %r", anchor.get_text(" ", strip=True))
            for line in describe(anchor):
                logger.info("  around it: %s", line)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()