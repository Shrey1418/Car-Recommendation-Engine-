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
from car_recommender.components.scoring import CRITERIA, PRESETS, read_safety, recommend  # noqa: E402
from car_recommender.config import load_config, resolve_path  # noqa: E402

APP_DATA = Path(__file__).resolve().parents[1] / "app_data"

PRESET_LABELS = {
    "Balanced (mix of low cost and good safety)": "balanced",
    "Family: safety first": "family_safety",
    "Lowest cost to own, within my range": "budget_saver",
}
PRESET_HELP = {
    "Balanced (mix of low cost and good safety)": "A mix of both: a reasonable cost to own and good safety, leaning towards the better car in your price range.",
    "Family: safety first": "Safety counts the most. Cost and running cost matter less.",
    "Lowest cost to own, within my range": "The cheapest to buy and run inside your price range. Safety counts the least.",
}
FUELS = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
button, input, textarea, select, h1, h2, h3, h4, p, li, label {
    font-family: 'Inter', sans-serif;
}
h1 { font-size: 2.1rem; font-weight: 700; }
h2 { font-size: 1.6rem; font-weight: 600; }
h3 { font-size: 1.25rem; font-weight: 600; }
p, li, label { font-size: 1rem; line-height: 1.6; }
[data-testid="stCaptionContainer"] { font-size: 0.9rem; }
</style>
"""


@st.cache_data
def load_static():
    cfg = load_config()
    manual = resolve_path(cfg["paths"]["manual_dir"])
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


def safety_badge(score, source) -> str:
    if pd.isna(score):
        return "❔ Not rated in our data"
    if score >= 85:
        label = "🛡️ Excellent"
    elif score >= 70:
        label = "✅ Good"
    elif score >= 50:
        label = "⚠️ Average"
    else:
        label = "❗ Low"
    text = f"{label} ({score:.0f} out of 100)"
    if "sibling" in str(source).lower():
        text += ", based on a sibling version of this car"
    return text


def diesel_note(brand: str, model: str, gaps: pd.DataFrame) -> str:
    row = gaps[(gaps["brand"] == brand) & (gaps["model"] == model)]
    if row.empty:
        return ""
    if "only" in str(row.iloc[0]["note"]).lower():
        return "ℹ️ Diesel versions of this model exist, but our data has only some of them."
    return "ℹ️ Diesel versions of this model exist, but they are not in our data."


def link_line(brand: str, model: str, links: pd.DataFrame) -> str:
    row = links[(links["brand"] == brand) & (links["model"] == model)]
    if row.empty:
        return f"we do not have a link for {brand} {model} yet."
    row = row.iloc[0]
    url = row["official_url"]
    status = row.get("link_status", "")
    if not url:
        return f"the official page link is not available yet. Please check the {brand} website."
    if status == "brand site only":
        return f"[{brand} website]({url}) (the model page is not in our list yet)"
    if status == "unverified":
        return f"[official model page]({url}) (we could not confirm that this link works)"
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


def show_pick(rank: int, pick, comparisons: pd.DataFrame, links: pd.DataFrame, gaps: pd.DataFrame) -> None:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    with st.container(border=True):
        st.subheader(f"{medals.get(rank, '')} {pick.brand} {pick.model} {pick.variant}")
        st.write(f"**{pick.fuel}**, {str(pick.transmission).lower()} gearbox")
        col_price, col_cost, col_run = st.columns(3)
        col_price.metric("Ex-showroom price", lakh(pick.ex_showroom_price))
        col_cost.metric("5-year cost", lakh(pick.total_cost))
        col_run.metric("Running cost", f"₹{pick.running_cost_per_km:.2f} per km")
        st.write(f"**Safety:** {safety_badge(pick.safety_score, pick.safety_source)}")
        st.write(f"**Why it ranks here:** {pick.reasons}")
        st.caption("These ranks are among all the variants that match your filters.")
        note = fuel_note(pick, comparisons)
        if note:
            st.write(f"**Compared with its petrol twin:** {note}")
        diesel = diesel_note(pick.brand, pick.model, gaps)
        if diesel:
            st.info(diesel)
        st.write("**Comfort and convenience features:** " + link_line(pick.brand, pick.model, links))


def by_fuel_table(by_fuel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Fuel": by_fuel["fuel"],
            "Car": by_fuel["brand"] + " " + by_fuel["model"] + " " + by_fuel["variant"],
            "Price (₹ lakh)": (by_fuel["ex_showroom_price"] / 100000).round(2),
            "5-year cost (₹ lakh)": (by_fuel["total_cost"] / 100000).round(2),
            "Safety": [safety_badge(s, src) for s, src in zip(by_fuel["safety_score"], by_fuel["safety_source"])],
        }
    )


def all_cars_table(ranked: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()
    priority = {"rated": 0, "all": 0, "not rated": 1}
    best = (
        ranked.assign(_priority=ranked["group"].map(priority))
        .sort_values(["_priority", "score"], ascending=[True, False])
        .drop_duplicates(["brand", "model"])
        .reset_index(drop=True)
    )
    top_group_size = int((best["group"] == best.loc[0, "group"]).sum())
    best["Pick"] = ""
    for position in range(min(3, top_group_size)):
        best.loc[position, "Pick"] = f"🏆 #{position + 1}"
    return pd.DataFrame(
        {
            "Pick": best["Pick"],
            "Car": best["brand"] + " " + best["model"] + " " + best["variant"],
            "Fuel": best["fuel"],
            "Gearbox": best["transmission"],
            "Price (₹ lakh)": (best["ex_showroom_price"] / 100000).round(2),
            "5-year cost (₹ lakh)": (best["total_cost"] / 100000).round(2),
            "Running cost (₹/km)": best["running_cost_per_km"].round(2),
            "Safety": [safety_badge(s, src) for s, src in zip(best["safety_score"], best["safety_source"])],
            "Match score": best["score"],
            "Note": [
                "Diesel not fully in our data" if diesel_note(b, m, gaps) else ""
                for b, m in zip(best["brand"], best["model"])
            ],
        }
    )


def highlight(table: pd.DataFrame):
    try:
        import jinja2  # noqa: F401
    except ImportError:
        return table
    return table.style.apply(
        lambda row: ["background-color: rgba(255, 193, 7, 0.25)" if row["Pick"] else "" for _ in row],
        axis=1,
    )


def ranking_explanation() -> str:
    lines = []
    for label, key in PRESET_LABELS.items():
        parts = [f"{CRITERIA[c][1]} {int(w * 100)}%" for c, w in PRESETS[key].items() if w > 0]
        lines.append(f"- **{label}:** " + ", ".join(parts))
    return "\n".join(lines)


def known_gaps_box(gaps: pd.DataFrame) -> None:
    with st.expander("How the ranking works"):
        st.markdown(
            "Cars are ranked with a method called TOPSIS. It finds the cars that are closest to the best "
            "result on every criterion at once, using the weights below.\n\n"
            + ranking_explanation()
            + "\n\nSafety badges: 🛡️ 85 or more, ✅ 70 to 84, ⚠️ 50 to 69, ❗ below 50. "
            "Scores from the older crash-test protocol are divided by 1.5, so newer, stricter tests count for more."
        )
    with st.expander("Known data gaps (please read)"):
        st.markdown(
            "- **Diesel:** the models listed below are sold with diesel engines in India, but their diesel variants are missing or incomplete in our data.\n"
            "- **Safety ratings:** about 4 in 10 on-sale variants have no rating in our data. This does not mean they are unsafe or untested. Please check the Bharat NCAP and Global NCAP websites.\n"
            "- **Comfort and convenience features** are not scored. Use the official links on each pick.\n"
            "- **Power, torque and size** are not used because that data is mostly missing. The price tier stands in for \"more car for the money\", which is only a rough guide.\n"
            "- **Cost** means the ex-showroom price plus fuel only. Insurance, service and registration are not included.\n"
            "- **EV comparison** uses the nearest-priced petrol variant of the sibling model, which favours a short payback.\n"
            "- **Mileage and EV range** are the claimed figures. Real-world numbers are usually lower."
        )
        st.dataframe(
            gaps.rename(columns={"brand": "Brand", "model": "Model", "note": "What is missing"}),
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title="Car Recommendation Engine India", page_icon="🚗", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("🚗 Car Recommendation Engine, India")
    st.caption(
        "Tell us what you need and we rank cars for you using TOPSIS on 5-year cost, running cost and safety. "
        "Cost means the ex-showroom price plus fuel only. This is a personal portfolio project and not financial advice."
    )

    _, safety, _, price_table, _, links, gaps = load_static()

    with st.sidebar:
        st.header("Your need")
        preset_label = st.radio("What matters most?", list(PRESET_LABELS))
        st.caption(PRESET_HELP.get(preset_label, ""))
        price_low, price_high = st.slider("Price range, ex-showroom (₹ lakh)", 3.0, 50.0, (8.0, 15.0), 0.5)
        seats = st.selectbox("Minimum number of seats", [4, 5, 6, 7], index=1)
        fuels = st.multiselect("Fuel types", FUELS, default=FUELS)
        transmission = st.selectbox("Gearbox", ["Any", "Manual", "Automatic"])
        cities = sorted(price_table["city"].dropna().tolist())
        city = st.selectbox("City (for fuel prices)", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
        annual_km = st.slider("Distance driven per year (km)", 3000, 30000, 12000, 1000)
        home_charging = st.checkbox("I can charge an electric car at home", value=True)

    costs, comparisons, prices = build_costs(city, float(annual_km), home_charging)
    st.sidebar.caption(
        f"Fuel prices used: petrol ₹{prices['petrol']:.2f} per litre, diesel ₹{prices['diesel']:.2f} per litre, "
        f"CNG ₹{prices['cng']:.2f} per kg, electricity ₹{prices['electric']:.2f} per kWh."
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
        st.warning("No car matches these filters. Try a wider price range, fewer seats, or more fuel types.")
    else:
        table = all_cars_table(result.get("ranked", pd.DataFrame()), gaps)
        st.write(f"**{result['shortlist_size']} variants** of **{len(table)} models** match your filters.")

        if "Diesel" in fuels or not fuels:
            listed = ", ".join((gaps["brand"] + " " + gaps["model"]).head(8))
            st.warning(
                f"Our diesel data is incomplete. These models are sold with diesel engines, but their diesel variants "
                f"are missing or thin in our data: {listed}. Diesel results may leave out good options. "
                "See \"Known data gaps\" at the bottom of the page."
            )

        top_group = next((g for g in ["rated", "all", "not rated"] if g in result["picks"]), None)
        if top_group:
            st.header("🏆 Your top 3 picks")
            if top_group == "not rated":
                st.info("None of these cars has a safety rating in our data, so they are ranked on cost only.")
            for rank, pick in enumerate(result["picks"][top_group].itertuples(index=False), start=1):
                show_pick(rank, pick, comparisons, links, gaps)

        by_fuel = result.get("by_fuel", pd.DataFrame())
        if len(by_fuel) > 1:
            st.header("Best safety-rated pick for each fuel type")
            st.dataframe(by_fuel_table(by_fuel), hide_index=True)

        if not table.empty:
            st.header("All cars in your range")
            st.caption("One row per model, showing its best-matching variant. The highlighted rows are your top picks.")
            st.dataframe(highlight(table), hide_index=True)
            st.caption(
                "Match scores can be compared only between cars of the same kind: rated cars with rated cars, "
                "and not-rated cars with not-rated cars. A car marked ❔ has no safety rating in our data, "
                "which does not mean it is unsafe."
            )

    known_gaps_box(gaps)


main()