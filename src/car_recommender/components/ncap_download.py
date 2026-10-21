import sys
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)


def load_robots(url, user_agent, timeout):
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    response = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=timeout)
    parser = RobotFileParser()
    if response.status_code in (401, 403):
        parser.disallow_all = True
    elif response.status_code >= 400:
        parser.allow_all = True
    else:
        parser.parse(response.text.splitlines())
    return parser


def download_one(session, url, target, timeout):
    response = session.get(url, timeout=timeout)
    if response.status_code in (429, 503):
        raise RuntimeError(f"Server asked us to slow down (HTTP {response.status_code})")
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("file does not look like a PDF")
    target.write_bytes(response.content)
    return len(response.content)


def download_all(entries, folder, session, robots, user_agent, timeout, delay):
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for row in entries.itertuples():
        target = folder / row.pdf_name
        if target.exists() and target.stat().st_size > 0:
            results.append({"pdf_name": row.pdf_name, "status": "cached", "bytes": target.stat().st_size, "error": ""})
            continue
        if not robots.can_fetch(user_agent, row.pdf_url):
            results.append({"pdf_name": row.pdf_name, "status": "blocked_by_robots", "bytes": 0, "error": ""})
            continue
        time.sleep(delay)
        try:
            size = download_one(session, row.pdf_url, target, timeout)
            results.append({"pdf_name": row.pdf_name, "status": "downloaded", "bytes": size, "error": ""})
        except RuntimeError as exc:
            logger.warning("Stopping early: %s", exc)
            results.append({"pdf_name": row.pdf_name, "status": "stopped", "bytes": 0, "error": str(exc)})
            break
        except Exception as exc:
            results.append({"pdf_name": row.pdf_name, "status": "failed", "bytes": 0, "error": str(exc)})
    return pd.DataFrame(results)


def main():
    try:
        cfg = load_config()
        scraping = cfg["scraping"]
        user_agent, timeout = scraping["user_agent"], scraping["timeout_seconds"]
        interim = resolve_path(cfg["paths"]["interim_dir"])
        folder = resolve_path("data/raw/ncap/pdfs")

        entries = pd.read_csv(interim / "ncap_entries.csv")
        robots = load_robots(entries["pdf_url"].iloc[0], user_agent, timeout)

        session = requests.Session()
        session.headers["User-Agent"] = user_agent
        results = download_all(
            entries, folder, session, robots, user_agent, timeout, scraping["request_delay_seconds"]
        )

        logger.info("Download results: %s", results["status"].value_counts().to_dict())
        failed = results[results["status"].isin(["failed", "blocked_by_robots", "stopped"])]
        for row in failed.itertuples():
            logger.warning("Not downloaded: %s (%s) %s", row.pdf_name, row.status, row.error)

        results.to_csv(interim / "ncap_downloads.csv", index=False)
        logger.info("PDFs are in %s", folder)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()