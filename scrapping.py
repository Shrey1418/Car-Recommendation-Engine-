"""
Scrape Global NCAP (GNCAP) and Bharat NCAP (BNCAP) results from Autocar India
and save both lists into ONE csv: data/ncap_ratings.csv

Run from the project root:
    python scrapping.py
Requires: pip install requests beautifulsoup4 pandas
"""
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

SOURCES = {
    "GNCAP": "https://www.autocarindia.com/auto-features/every-indian-car-tested-by-global-ncap-under-new-protocols-431384",
    "BNCAP": "https://www.autocarindia.com/auto-features/every-car-tested-by-bharat-ncap-so-far-431936",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (personal research project)"}
OUT_FILE = Path("data/ncap_ratings.csv")

NUM = r"(\d+(?:\.\d+)?)"
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

# Child stars for GNCAP, as stated in the article (headings only give adult stars)
GNCAP_CHILD_STARS = {
    "Volkswagen Taigun": 5, "Skoda Kushaq": 5, "Volkswagen Virtus": 5, "Skoda Slavia": 5,
    "Maruti Suzuki Ignis": 0, "Maruti Suzuki S Presso": 0, "Maruti Suzuki Wagon R": 0,
    "Maruti Suzuki Alto K10": 0, "Hyundai Verna": 5, "Tata Harrier": 5, "Tata Safari": 5,
    "Mahindra Scorpio N": 3, "Tata Nexon": 5, "Citroen eC3": 1, "Mahindra Bolero Neo": 1,
    "Honda Amaze": 0, "Kia Carens": 5, "Maruti Suzuki Dzire": 5, "Nissan Magnite": 3,
    "Maruti Suzuki Victoris": 5, "Maruti Suzuki Ciaz": 3, "Maruti Suzuki Celerio": 2,
}

# BNCAP child-star thresholds (out of 49), inferred from published results.
BNCAP_COP_THRESHOLDS = [(41, 5), (35, 4), (27, 3), (18, 2), (9, 1)]

# Longest names are matched first, so "Maruti Suzuki" wins over any shorter name
BRANDS = [
    "Maruti Suzuki", "Volkswagen", "Mahindra", "Hyundai", "Citroen", "Renault",
    "Toyota", "Nissan", "Vinfast", "Honda", "Skoda", "Tata", "Kia",
]
# Short forms used in the articles -> the full brand name we store
BRAND_ALIASES = {"Maruti": "Maruti Suzuki"}


def get_soup(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def sections(soup):
    """Yield (heading_text, body_text) for every <h2>; body runs until the next <h2>."""
    for h2 in soup.find_all("h2"):
        heading = h2.get_text(" ", strip=True)
        parts = []
        for el in h2.find_all_next(["h2", "h3", "p"]):
            if el.name == "h2":
                break
            parts.append(el.get_text(" ", strip=True))
        yield heading, " ".join(parts)


def first(pattern, text, group=1, flags=re.I):
    m = re.search(pattern, text, flags)
    return m.group(group) if m else None


def find_cop(text):
    v = first(rf"{NUM}\s*(?:points\s*)?(?:out of|/)\s*(?:a\s*)?(?:maximum\s*|possible\s*)?(?:of\s*)?49\b", text)
    if v is None:
        v = first(rf"child[^.]{{0,80}}?scor\w*\s*{NUM}\s*points", text)
    return float(v) if v else None


def find_max(text, default):
    v = first(rf"{NUM}\s*(?:points\s*)?(?:out of|/)\s*(?:a\s*)?(?:maximum\s*|possible\s*)?(?:of\s*)?(32|34)\b", text, 2)
    return int(v) if v else default


# ----------------------------------------------------------------- GNCAP
def parse_gncap(soup):
    head_re = re.compile(rf"^\s*(\d+)\.\s*(.+?)\s*-\s*(\d)\s*stars?\s*\({NUM}", re.I)
    rows = []
    for heading, text in sections(soup):
        m = head_re.match(heading)
        if not m:
            continue
        rank, names, stars, aop = int(m[1]), m[2], int(m[3]), float(m[4])
        cop = find_cop(text)
        adult_max = find_max(text, 34)
        date = first(rf"announced in ({MONTHS} \d{{4}})", text)

        parts = [p.strip() for p in names.split("/")]
        for i, p in enumerate(parts):
            if i > 0 and len(p.split()) == 1:  # "Tata Harrier/Safari" -> "Tata Safari"
                p = parts[0].split()[0] + " " + p
            rows.append({
                "source": "GNCAP", "rank": rank, "car_model": p,
                "adult_score": aop, "adult_max": adult_max, "adult_stars": stars,
                "child_score": cop, "child_max": 49,
                "child_stars": GNCAP_CHILD_STARS.get(p),
                "tested": date, "child_stars_basis": "article",
            })
    return rows


# ----------------------------------------------------------------- BNCAP
def bncap_scores(text):
    aop = first(rf"{NUM}\s*(?:points\s*)?(?:out of|/)\s*(?:a\s*)?(?:maximum\s*|possible\s*)?(?:of\s*)?32\b", text)
    cop = first(rf"{NUM}\s*(?:points\s*)?(?:out of|/)\s*(?:a\s*)?(?:maximum\s*|possible\s*)?(?:of\s*)?49\b", text)
    if aop is None or cop is None:
        pair = re.search(rf"{NUM}\s*and\s*(?:a\s*)?{NUM}\s*(?:points\s*)?,?\s*respectively", text, re.I)
        if pair:
            aop = aop or pair[1]
            cop = cop or pair[2]
    if aop is None:
        aop = first(rf"{NUM}\s*(?:points\s*)?(?:in|for)\s*(?:both\s*)?AOP", text) \
            or first(rf"AOP[^\d]{{0,40}}?{NUM}", text)
    if cop is None:
        cop = first(rf"{NUM}\s*(?:points\s*)?(?:in|for)\s*COP", text) \
            or first(rf"COP[^\d]{{0,40}}?{NUM}", text)
    return (float(aop) if aop else None, float(cop) if cop else None)


def derive_child_stars(cop):
    if cop is None:
        return None
    for threshold, stars in BNCAP_COP_THRESHOLDS:
        if cop >= threshold:
            return stars
    return 0


def parse_bncap(soup):
    head_re = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
    rows = []
    for heading, text in sections(soup):
        m = head_re.match(heading)
        stars = first(r"BNCAP rating:\s*(\d)\s*stars?", text)
        if not m or stars is None:
            continue  # skips "Explore Cars", "More Stories", ...
        aop, cop = bncap_scores(text)
        child = first(r"(\d)\s*stars?\s*(?:in|for)\s*(?:the\s*)?COP", text)
        basis = "article"
        if child is None:
            child, basis = derive_child_stars(cop), "derived from score"
        date = first(rf"crash-tested in ({MONTHS} \d{{4}})", text)
        rows.append({
            "source": "BNCAP", "rank": int(m[1]), "car_model": m[2],
            "adult_score": aop, "adult_max": 32, "adult_stars": int(stars),
            "child_score": cop, "child_max": 49,
            "child_stars": int(child) if child is not None else None,
            "tested": date, "child_stars_basis": basis,
        })

    # The Renault Duster paragraph says its scores are identical to the Nissan Tekton
    by_name = {r["car_model"]: r for r in rows}
    duster, tekton = by_name.get("Renault Duster"), by_name.get("Nissan Tekton")
    if duster and tekton and duster["adult_score"] is None:
        duster["adult_score"] = tekton["adult_score"]
        duster["child_score"] = tekton["child_score"]
        duster["child_stars"] = tekton["child_stars"]
        duster["child_stars_basis"] = "copied from Nissan Tekton (identical per article)"
    return rows


# ------------------------------------------------------------------ brands
def split_company_model(name):
    name = name.strip()
    names = {b: b for b in BRANDS}
    names.update(BRAND_ALIASES)
    for alias in sorted(names, key=len, reverse=True):  # longest first
        if name.lower().startswith(alias.lower() + " "):
            return names[alias], name[len(alias):].strip()
    return None, name  # unknown brand: reported in main()


# ------------------------------------------------------------------ saving
def save_csv(df, path):
    """Save the CSV; survive the two usual Windows PermissionError causes."""
    path = Path(path)

    # Cause 1: a FOLDER with this name already exists -> save inside it
    if path.is_dir():
        print(f"\nNote: '{path}' is a folder, not a file. Saving inside it instead.")
        path = path / "ncap_ratings.csv"

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(path, index=False)
    except PermissionError:
        # Cause 2: file is open in Excel / locked -> save under a new name
        alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        print(f"\nCould not write '{path}' (is it open in Excel?). Saving to '{alt}' instead.")
        df.to_csv(alt, index=False)
        path = alt
    return path


# ------------------------------------------------------------------ main
def main():
    all_rows = []
    for name, parser in (("GNCAP", parse_gncap), ("BNCAP", parse_bncap)):
        rows = parser(get_soup(SOURCES[name]))
        print(f"{name}: {len(rows)} rows")
        all_rows += rows
        time.sleep(2)  # be polite

    df = pd.DataFrame(all_rows)

    # Split "Maruti Suzuki Dzire" -> company_name "Maruti Suzuki", model_name "Dzire"
    df[["company_name", "model_name"]] = df["car_model"].apply(
        lambda n: pd.Series(split_company_model(n))
    )
    df = df.drop(columns="car_model")

    unknown = df[df["company_name"].isna()]
    if not unknown.empty:
        print("\nUnknown brand - add it to BRANDS:")
        print(unknown["model_name"].tolist())

    # Put the identifying columns first
    cols = ["source", "rank", "company_name", "model_name"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    df = df.sort_values(["source", "company_name", "model_name"]).reset_index(drop=True)

    saved_to = save_csv(df, OUT_FILE)

    missing = df[df[["adult_score", "child_score", "child_stars"]].isna().any(axis=1)]
    if not missing.empty:
        print("\nRows with missing values - check these by hand:")
        print(missing[["source", "company_name", "model_name", "adult_score", "child_score", "child_stars"]])

    print(f"\nSaved {len(df)} rows to {saved_to}")


if __name__ == "__main__":
    main()