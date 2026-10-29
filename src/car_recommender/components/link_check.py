import re
import sys
import time
from urllib.parse import urlparse

import pandas as pd
import requests

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

CONTACT_EMAIL = "put-your-real-email-here"
HEADERS = {"User-Agent": f"CarRecommenderLinkCheck/1.0 (personal student project; contact: {CONTACT_EMAIL})"}
DELAY_SECONDS = 1.5
TIMEOUT_SECONDS = 20


def read_links(path) -> pd.DataFrame:
    links = pd.read_csv(path, dtype="string", encoding="utf-8-sig").fillna("")
    links["official_url"] = links["official_url"].str.strip()
    return links


def page_title(html: str) -> str:
    found = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", found.group(1)).strip()[:80] if found else ""


def check_one(session: requests.Session, url: str) -> dict:
    try:
        response = session.get(url, timeout=TIMEOUT_SECONDS, allow_redirects=True)
    except requests.RequestException as error:
        return {"official_url": url, "status": "", "final_url": "", "page_title": "", "result": "error: " + type(error).__name__}

    original_path = urlparse(url).path.strip("/")
    final_path = urlparse(response.url).path.strip("/")
    if response.status_code in (401, 403, 429):
        result = "blocked, check by hand"
    elif response.status_code >= 400:
        result = "dead"
    elif original_path and not final_path:
        result = "redirected to homepage"
    else:
        result = "ok"
    return {
        "official_url": url,
        "status": response.status_code,
        "final_url": response.url,
        "page_title": page_title(response.text),
        "result": result,
    }


def main():
    try:
        cfg = load_config()
        manual = resolve_path(cfg["paths"]["manual_dir"])
        processed = resolve_path(cfg["paths"]["processed_dir"])

        links = read_links(manual / "official_links.csv")
        urls = [u for u in links["official_url"].unique() if u]
        logger.info("Checking %d distinct links for %d models", len(urls), len(links))

        session = requests.Session()
        session.headers.update(HEADERS)
        results = []
        for number, url in enumerate(urls, start=1):
            results.append(check_one(session, url))
            if number % 20 == 0:
                logger.info("Checked %d of %d", number, len(urls))
            time.sleep(DELAY_SECONDS)

        checked = links.merge(pd.DataFrame(results), on="official_url", how="left")
        checked["homepage_link"] = checked["official_url"].map(lambda u: bool(u) and urlparse(u).path.strip("/") == "")
        checked["shared_by_models"] = checked.groupby("official_url")["model"].transform("count")
        checked.to_csv(processed / "link_check.csv", index=False)

        logger.info("Results: %s", checked["result"].fillna("no link").value_counts().to_dict())
        logger.info("Homepage-only links: %d", int(checked["homepage_link"].sum()))
        for row in checked[checked["result"].fillna("no link") != "ok"].itertuples():
            logger.warning("%s | %s -> %s (%s)", row.brand, row.model, row.result, row.official_url)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()