import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_recommender.components.cost_model import (  # noqa: E402
    add_running_cost,
    compare_fuels,
    number,
    pick_prices,
    read_assumptions,
    read_fuel_prices,
)
from car_recommender.components.scoring import read_safety, recommend  # noqa: E402
from car_recommender.config import load_config, resolve_path  # noqa: E402

APP_DATA = Path(__file__).resolve().parents[1] / "app_data"

PRESET_LABELS = {
    "Balanced (mix of low cost and good safety)": "balanced",
    "Family: safety first": "family_safety",
    "Lowest cost to own, within my range": "budget_saver",
}
PRESET_HELP = {
    "Balanced (mix of low cost and good safety)": "Mix of both: reasonable cost to own and good safety, leaning towards the better car in your price range.",
    "Family: safety first": "Safety counts the most; cost and running cost matter less.",
    "Lowest cost to own, within my range": "Cheapest to buy and run inside your price range; safety counts the least.",
}
FUELS = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
GROUP_TITLES = {
    "rated": "Top picks (safety-rated cars)",
    "all": "Top picks",
    "not rated": "Also consider: no safety rating on file (ranked on cost only, not comparable with the list above)",
}


@st.cache_data
def load_static():
    cfg = load_config()
    manual = resolve_path(cfg["paths"]["manual_dir"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    assumptions = read_assumptions(manual / "cost_assumptions.csv")
    ratio = number(assumptions, "new_vs_old_protocol_ratio")
    variants = pd.read_csv(APP_DATA / "variants_app.csv", encoding="utf-8-sig")
    safety = read_safety(APP_DATA / "safety_app.csv", ratio)
    price_table = read_fuel_prices(manual / "fuel_prices_india.csv")
    families = pd.read_csv(manual / "ev_families.csv", encoding="utf-8-sig")
    links = pd.read_csv(manual / "official_links.csv", dtype="string", encoding="utf-8-sig").fillna("")
    gaps = pd.read_csv(manual / "known_gaps.csv", encoding="utf-8-sig")
    return variants, safety, assumptions, price_table, families, links, gaps


@st.cache_data
def build_costs(city: str, annual_km: float, home_charging: bool = True):
    variants, _, assumptions, price_table, families, _, _ = load_static()
    years = int(number(assumptions, "ownership_years"))
    growth = number(assumptions, "fuel_price_growth_pct_per_year")
    overrides = None if home_charging else {"electric": number(assumptions, "public_fast_charging_price")}
    prices = pick_prices(price_table, city, assumptions, overrides)
    costs = add_running_cost(variants, prices, annual_km, years, growth)
    comparisons = compare_fuels(costs, families, annual_km, years)
    return costs, comparisons, prices


def lakh(value: float) -> str:
    return f"₹{value / 100000:.2f} lakh"


def link_line(brand: str, model: str, links: pd.DataFrame) -> str:
    row = links[(links["brand"] == brand) & (links["model"] == model)]
    if row.empty:
        return f"no link on file for {brand} {model}."
    row = row.iloc[0]
    url = row["official_url"]
    status = row.get("link_status", "")
    if not url:
        return f"official page not available yet, please check the {brand} website."
    if status == "brand site only":
        return f"[{brand} website]({url}) (the model page is not in our list yet)"
    if status == "unverified":
        return f"[official model page]({url}) (could not be auto-verified)"
    return f"[official model page]({url})"


def fuel_note(pick, comparisons: pd.DataFrame) -> str:
    match = comparisons[
        (comparisons["brand"] == pick.brand)
        & (comparisons["model_alt"] == pick.model)
        & (comparisons["variant_alt"] == pick.variant)
        & (comparisons["fuel_alt"] == pick.fuel)
        & (comparisons["transmission_alt"] == pick.transmission)
    ]
    if match.empty:
        return ""
    row = match.iloc[0]
    return f"{row['verdict']} (basis: {row['comparison_basis']})"


def show_pick(rank: int, pick, comparisons: pd.DataFrame, links: pd.DataFrame) -> None:
    with st.container(border=True):
        st.subheader(f"{rank}. {pick.brand} {pick.model} {pick.variant}")
        st.write(
            f"{pick.fuel}, {pick.transmission} | ex-showroom {lakh(pick.ex_showroom_price)} | "
            f"5-year cost {lakh(pick.total_cost)} | running cost ₹{pick.running_cost_per_km:.2f} per km"
        )
        if pd.notna(pick.safety_score):
            st.write(f"Safety score: {pick.safety_score:.0f} out of 100 ({pick.safety_source})")
        else:
            st.write("Safety: no rating on file")
        st.write(f"Why it ranks here: {pick.reasons}")
        note = fuel_note(pick, comparisons)
        if note:
            st.write(f"Versus its petrol twin: {note}")
        st.write("Comfort and convenience features: " + link_line(pick.brand, pick.model, links))


def main() -> None:
    st.set_page_config(page_title="Car Recommendation Engine India", layout="wide")
    st.title("Car Recommendation Engine, India")
    st.caption(
        "Ranks cars by your need using TOPSIS on 5-year cost, running cost and safety. "
        "Cost means ex-showroom price plus fuel only. A personal portfolio project, not financial advice."
    )

    _, safety, _, price_table, _, links, gaps = load_static()

    with st.sidebar:
        st.header("Your need")
        preset_label = st.radio("What matters most", list(PRESET_LABELS))
        st.caption(PRESET_HELP.get(preset_label, ""))
        price_low, price_high = st.slider("Price range, ex-showroom (lakh rupees)", 3.0, 50.0, (8.0, 15.0), 0.5)
        seats = st.selectbox("Minimum seats", [4, 5, 6, 7], index=1)
        fuels = st.multiselect("Fuel types", FUELS, default=FUELS)
        transmission = st.selectbox("Transmission", ["Any", "Manual", "Automatic"])
        cities = sorted(price_table["city"].dropna().tolist())
        city = st.selectbox("City (for fuel prices)", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
        annual_km = st.slider("Distance driven per year (km)", 3000, 30000, 12000, 1000)
        home_charging = st.checkbox("I can charge an EV at home", value=True)

    costs, comparisons, prices = build_costs(city, float(annual_km), home_charging)
    st.sidebar.caption(
        f"Fuel prices used: petrol ₹{prices['petrol']:.2f}, diesel ₹{prices['diesel']:.2f}, "
        f"CNG ₹{prices['cng']:.2f} per litre/kg, electricity ₹{prices['electric']:.2f} per kWh"
    )

    result = recommend(
        costs,
        safety,
        price_high * 100000,
        seats,
        fuels or None,
        None if transmission == "Any" else transmission,
        PRESET_LABELS[preset_label],
        price_low * 100000,
    )

    if result["shortlist_size"] == 0:
        st.warning("No car passes these filters. Raise the budget or relax seats, fuel or transmission.")
    else:
        st.write(f"{result['shortlist_size']} variants pass your filters.")
        if "Diesel" in fuels or not fuels:
            listed = ", ".join((gaps["brand"] + " " + gaps["model"]).head(8))
            st.warning(f"Diesel data is incomplete. These diesel models are missing or thin in our catalog: {listed}. Diesel results may leave out good options; see Known data gaps below.")
        by_fuel = result["by_fuel"]
        if len(by_fuel) > 1:
            st.subheader("Best safety-rated pick for each fuel type")
            table = by_fuel.assign(
                car=by_fuel["brand"] + " " + by_fuel["model"] + " " + by_fuel["variant"],
                price_lakh=(by_fuel["ex_showroom_price"] / 100000).round(2),
                five_year_cost_lakh=(by_fuel["total_cost"] / 100000).round(2),
            )
            st.dataframe(table[["fuel", "car", "price_lakh", "five_year_cost_lakh", "safety_score", "score"]], hide_index=True)
        for group in ["rated", "all", "not rated"]:
            picks = result["picks"].get(group)
            if picks is None:
                continue
            st.header(GROUP_TITLES[group])
            for rank, pick in enumerate(picks.itertuples(index=False), start=1):
                show_pick(rank, pick, comparisons, links)

    with st.expander("Known data gaps (please read)"):
        st.markdown(
            "- Diesel variants are missing or incomplete for the models listed below, so diesel results can be incomplete.\n"
            "- About 4 in 10 on-sale variants have no safety rating on file. They are shown in a separate list.\n"
            "- Comfort and convenience features are not scored. Use the official links on each pick.\n"
            "- Power, torque and size are not used because that data is mostly missing.\n"
            "- Cost is ex-showroom price plus fuel only. Insurance, service and registration are not included.\n"
            "- EV comparison uses the nearest-price petrol variant of the sibling model, which favours short payback.\n"
            "- Mileage and EV range are claimed figures. Real-world numbers are usually lower."
        )
        st.dataframe(gaps, hide_index=True)


main()