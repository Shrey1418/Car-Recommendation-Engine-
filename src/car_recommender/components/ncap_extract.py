import re
import sys

import pandas as pd
from pypdf import PdfReader

from car_recommender.config import load_config, resolve_path
from car_recommender.exception import CustomException
from car_recommender.logger import get_logger

logger = get_logger(__name__)

NUMBER = r"(\d+(?:\.\d+)?)"
SCORE = re.compile(NUMBER + r"\s*max\.?\s*" + NUMBER + r"\s*(Adult|Child)", re.IGNORECASE)
STARS = re.compile(r"All this explain(?:s|ed) the (\d)\s*star result", re.IGNORECASE)
YEAR = re.compile(r"Year of publication\s*(\d{4})", re.IGNORECASE)


def read_pdf_text(path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = " ".join(" ".join(pages).split())
    return re.sub(r"(\d)\s+\.(\d)", r"\1.\2", text)


def parse_factsheet(text: str) -> dict:
    row = {
        "adult_score": None, "adult_max": None, "child_score": None, "child_max": None,
        "adult_stars": None, "child_stars": None, "pdf_year": None, "parse_status": "unreadable", "note": "",
    }
    found = {}
    for score, maximum, label in SCORE.findall(text):
        found.setdefault(label.lower(), (float(score), float(maximum)))
    stars = STARS.findall(text)
    year = YEAR.search(text)

    if year:
        row["pdf_year"] = int(year.group(1))
    if "adult" in found and "child" in found:
        row["adult_score"], row["adult_max"] = found["adult"]
        row["child_score"], row["child_max"] = found["child"]
        row["parse_status"] = "ok"
    if len(stars) == 2:
        row["adult_stars"], row["child_stars"] = int(stars[0]), int(stars[1])
    elif row["parse_status"] == "ok":
        row["note"] = f"stars not read (found {len(stars)} star statements)"
    return row


def check_row(row: dict) -> str:
    if row["parse_status"] == "unreadable":
        return row["note"]
    problems = []
    if row["adult_score"] > row["adult_max"] or row["child_score"] > row["child_max"]:
        problems.append("score is above its maximum")
    if row["adult_stars"] is not None and not (0 <= row["adult_stars"] <= 5 and 0 <= row["child_stars"] <= 5):
        problems.append("stars outside 0-5")
    return "; ".join(problems)


def main():
    try:
        cfg = load_config()
        interim = resolve_path(cfg["paths"]["interim_dir"])
        folder = resolve_path("data/raw/ncap/pdfs")
        entries = pd.read_csv(interim / "ncap_entries.csv")

        rows, unreadable = [], []
        for entry in entries.itertuples():
            text = read_pdf_text(folder / entry.pdf_name)
            row = parse_factsheet(text)
            problem = check_row(row)
            if problem:
                row["note"] = (row["note"] + "; " + problem).strip("; ")
                row["parse_status"] = "check_needed" if row["parse_status"] != "unreadable" else "unreadable"
            if row["parse_status"] != "ok":
                unreadable.append({"pdf_name": entry.pdf_name, "text_start": text[:600]})
            rows.append({"pdf_name": entry.pdf_name, **row})

        scores = entries.merge(pd.DataFrame(rows), on="pdf_name")
        logger.info("Fact sheets by status: %s", scores["parse_status"].value_counts().to_dict())
        logger.info("Status by year:\n%s", pd.crosstab(scores["year"], scores["parse_status"]).to_string())
        logger.info(
            "Adult maximum seen: %s",
            scores["adult_max"].value_counts(dropna=False).to_dict(),
        )
        year_mismatch = scores[scores["pdf_year"].notna() & (scores["pdf_year"] != scores["year"])]
        if len(year_mismatch):
            logger.warning("Year on the page differs from year inside the PDF for %d sheets", len(year_mismatch))

        scores.to_csv(interim / "ncap_scores.csv", index=False)
        pd.DataFrame(unreadable).to_csv(interim / "ncap_scores_unreadable.csv", index=False)
        logger.info("Saved ncap_scores.csv and ncap_scores_unreadable.csv to %s", interim)
    except Exception as exc:
        raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    main()