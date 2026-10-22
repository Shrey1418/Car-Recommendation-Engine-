from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

BASE = "https://www.bncap.in/vehicle-safety-ratings/"
FILE = Path("data/raw/bncap/ratings.html")


def vehicle_hrefs(node):
    hrefs = set()
    for anchor in node.find_all("a", href=True):
        href = urljoin(BASE, anchor["href"])
        if urlparse(href).path.startswith("/vehicle/"):
            hrefs.add(href)
    return hrefs


def find_card(anchor):
    card = anchor
    node = anchor.parent
    while node is not None and len(vehicle_hrefs(node)) == 1:
        card = node
        node = node.parent
    return card


soup = BeautifulSoup(FILE.read_text(encoding="utf-8"), "html.parser")
seen = []
for anchor in soup.find_all("a", href=True):
    href = urljoin(BASE, anchor["href"])
    if urlparse(href).path.startswith("/vehicle/") and href not in seen:
        seen.append(href)
        if len(seen) <= 3:
            card = find_card(anchor)
            print("=" * 70)
            print(href)
            print("TEXT:", " ".join(card.get_text(" ", strip=True).split()))
            print("HTML:", str(card)[:2500])