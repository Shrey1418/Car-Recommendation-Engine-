import sys
import urllib.error
import urllib.request
from urllib.robotparser import RobotFileParser

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

SITES = {
    "Tata": "https://www.tatamotors.com/",
    "Hyundai": "https://www.hyundai.com/in",
    "Kia": "https://www.kia.com/in",
    "Mahindra": "https://auto.mahindra.com",
    "MG": "https://www.mgmotor.co.in",
    "Toyota": "https://www.toyotabharat.com",
    "Renault": "https://www.renault.co.in",
    "Nissan": "https://www.nissan.in",
}


def fetch_robots(base_url, user_agent, timeout):
    root = "/".join(base_url.split("/")[:3])
    request = urllib.request.Request(root + "/robots.txt", headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        return err.code, ""
    except Exception as exc:
        return None, str(exc)


def evaluate(status, text, user_agent, base_url):
    if status is None:
        return "UNREACHABLE", None
    if status in (401, 403):
        return f"BLOCKED (HTTP {status}), do not scrape", None
    if status >= 400:
        return f"no robots.txt (HTTP {status}), still read the Terms of Use", None
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    delay = parser.crawl_delay(user_agent)
    if not parser.can_fetch(user_agent, base_url + "/"):
        return "DISALLOWED for us, do not scrape", delay
    return "allowed by robots.txt, still read the Terms of Use", delay


def main():
    try:
        cfg = load_config()
        user_agent = cfg["scraping"]["user_agent"]
        timeout = cfg["scraping"]["timeout_seconds"]
        save_dir = resolve_path("data/raw/robots")
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, base_url in SITES.items():
            status, text = fetch_robots(base_url, user_agent, timeout)
            verdict, delay = evaluate(status, text, user_agent, base_url)
            if status == 200:
                (save_dir / f"{name.lower()}_robots.txt").write_text(text, encoding="utf-8")
            logger.info("%-9s HTTP %s | %s | crawl-delay: %s", name, status, verdict, delay)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()