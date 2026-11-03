import math
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

APP_NAME = "CRAMAS"
TAGLINE = "A place where you get the Best for your need"
KEY = ["brand", "model", "variant", "fuel", "transmission"]

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
html, body, [data-testid="stAppViewContainer"], button, input, textarea, select, h1, h2, h3, h4, p, li, label {
    font-family: 'Inter', sans-serif;
}
h1 { font-size: 2.6rem; font-weight: 700; letter-spacing: 3px; margin-bottom: 0; }
h2 { font-size: 1.6rem; font-weight: 600; }
h3 { font-size: 1.25rem; font-weight: 600; }
p, li, label { font-size: 1rem; line-height: 1.6; }
[data-testid="stCaptionContainer"] { font-size: 0.9rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #27313C; border-radius: 12px; }
.block-container { padding-top: 2rem; }
.st-key-qa [data-testid="stWidgetLabel"] p { font-size: 1.2rem; font-weight: 700; }
.st-key-qa [data-testid="stRadio"] div[role="radiogroup"] label p { font-size: 1.1rem; font-weight: 500; }
.st-key-qa [data-testid="stCheckbox"] p { font-size: 1.2rem; font-weight: 700; }
.st-key-qa div[data-baseweb="select"] { font-size: 1.05rem; }
</style>
"""

TABLE_CONFIG = {
    "Pick": st.column_config.TextColumn(width=62),
    "Brand": st.column_config.TextColumn(width=80),
    "Car": st.column_config.TextColumn(width=115),
    "Variant": st.column_config.TextColumn(width=140),
    "Fuel": st.column_config.TextColumn(width=75),
    "Gearbox": st.column_config.TextColumn(width=95),
    "Price (₹ L)": st.column_config.NumberColumn(format="%.2f", width=85, help="Ex-showroom price in lakh rupees."),
    "5-yr cost (₹ L)": st.column_config.NumberColumn(
        format="%.2f", width=105, help="Ex-showroom price plus 5 years of fuel, in lakh rupees."
    ),
    "Running (₹/km)": st.column_config.NumberColumn(format="%.2f", width=105),
    "Safety": st.column_config.NumberColumn(
        format="%d", width=70,
        help="Average of adult and child crash-test scores, out of 100. Blank means not rated in our data.",
    ),
    "Index rating": st.column_config.ProgressColumn(
        min_value=0, max_value=100, format="%d", width=105,
        help="Our fixed formula (balanced weights) across all cars on sale up to ₹50 lakh. It does not change with your answers.",
    ),
    "Your rating": st.column_config.ProgressColumn(
        min_value=0, max_value=100, format="%d", width=105,
        help="The same method, but with your priority, among the cars that match your answers.",
    ),
    "Rated on": st.column_config.TextColumn(
        width=110, help="Cars with no safety rating are scored on cost only. Compare ratings only within the same value."
    ),
    "Official page": st.column_config.LinkColumn(display_text="Open ↗", width=100),
}


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


@st.cache_data
def build_index(city: str, annual_km: float, home_charging: bool = True) -> pd.DataFrame:
    costs, _, _ = build_costs(city, annual_km, home_charging)
    _, safety, _, _, _, _, _ = load_static()
    result = recommend(costs, safety, 50 * 100000, 0, None, None, "balanced", 0.0)
    ranked = result.get("ranked", pd.DataFrame())
    if ranked.empty:
        return pd.DataFrame(columns=KEY + ["index_rating"])
    values = ranked["score"] * 100 if ranked["score"].max() <= 1 else ranked["score"]
    return ranked[KEY].assign(index_rating=values.round(0)).drop_duplicates(KEY)


def lakh(value: float) -> str:
    return f"₹{value / 100000:.2f} lakh"


def rating_text(score) -> str:
    if score is None or pd.isna(score):
        return "n/a"
    value = score * 100 if score <= 1 else score
    return f"{value:.0f} / 100"


def index_text(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.0f} / 100"


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


def safety_source_line(source) -> str:
    if pd.isna(source) or not str(source).strip():
        return ""
    return f"Safety source: {source}"


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
        st.subheader(f"{medals.get(rank, '')} {pick.brand} {pick.model}")
        st.caption(f"{pick.variant} · {pick.fuel} · {str(pick.transmission).lower()} gearbox")
        st.metric("Your rating", rating_text(getattr(pick, "score", None)))
        st.caption(f"Index rating: {index_text(getattr(pick, 'index_rating', None))}")
        st.write(f"**Price:** {lakh(pick.ex_showroom_price)}")
        st.write(f"**5-year cost:** {lakh(pick.total_cost)}")
        st.write(f"**Running cost:** ₹{pick.running_cost_per_km:.2f} per km")
        st.write(f"**Safety:** {safety_badge(pick.safety_score, pick.safety_source)}")
        source_line = safety_source_line(pick.safety_source)
        if source_line:
            st.caption(source_line)
        with st.expander("Why this pick"):
            st.write(pick.reasons)
            note = fuel_note(pick, comparisons)
            if note:
                st.write(f"**Compared with its petrol twin:** {note}")
            st.caption("Rank and rating are among all the variants that match your answers.")
        diesel = diesel_note(pick.brand, pick.model, gaps)
        if diesel:
            st.info(diesel)
        st.write("**Features:** " + link_line(pick.brand, pick.model, links))


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


def cars_table(ranked: pd.DataFrame, index: pd.DataFrame, links: pd.DataFrame, top_picks) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()
    priority = {"rated": 0, "all": 0, "not rated": 1}
    link_urls = links[["brand", "model", "official_url"]].drop_duplicates(["brand", "model"])
    table = ranked.merge(index, on=KEY, how="left").merge(link_urls, on=["brand", "model"], how="left")
    table = (
        table.assign(_priority=table["group"].map(priority))
        .sort_values(["_priority", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    pick_labels = {}
    if top_picks is not None:
        for number_, key in enumerate(top_picks[KEY].itertuples(index=False, name=None), start=1):
            pick_labels[tuple(key)] = f"🏆 #{number_}"
    labels = [pick_labels.get(tuple(key), "") for key in table[KEY].itertuples(index=False, name=None)]

    your = table["score"] * 100 if table["score"].max() <= 1 else table["score"]
    url = table["official_url"].astype(object).where(lambda s: s.fillna("").ne(""), None)
    basis = {"rated": "Cost + safety", "not rated": "Cost only", "all": "All cars together"}

    return pd.DataFrame(
        {
            "Pick": labels,
            "Brand": table["brand"],
            "Car": table["model"],
            "Variant": table["variant"],
            "Fuel": table["fuel"],
            "Gearbox": table["transmission"],
            "Price (₹ L)": (table["ex_showroom_price"] / 100000).round(2),
            "5-yr cost (₹ L)": (table["total_cost"] / 100000).round(2),
            "Running (₹/km)": table["running_cost_per_km"].round(2),
            "Safety": table["safety_score"],
            "Index rating": table["index_rating"],
            "Your rating": your.round(0),
            "Rated on": table["group"].map(basis),
            "Official page": url,
        }
    )


def filter_table(table: pd.DataFrame) -> pd.DataFrame:
    st.caption("Filter the table below. Your top 3 picks above do not change.")
    col_brand, col_safety, col_run, col_index = st.columns(4)
    brands = col_brand.multiselect("Brand", sorted(table["Brand"].dropna().unique()))
    min_safety = col_safety.slider(
        "Minimum safety score", 0, 100, 0, 5, help="Cars with no rating are hidden when this is above 0."
    )
    top_run = table["Running (₹/km)"].max()
    run_max = float(math.ceil(top_run)) if pd.notna(top_run) else 10.0
    run_max = max(run_max, 1.0)
    max_run = col_run.slider("Maximum running cost (₹/km)", 0.0, run_max, run_max, 0.5)
    min_index = col_index.slider("Minimum index rating", 0, 100, 0, 5)

    shown = table
    if brands:
        shown = shown[shown["Brand"].isin(brands)]
    if min_safety > 0:
        shown = shown[shown["Safety"] >= min_safety]
    shown = shown[shown["Running (₹/km)"].fillna(0) <= max_run]
    if min_index > 0:
        shown = shown[shown["Index rating"] >= min_index]
    return shown


def ranking_explanation() -> str:
    lines = []
    for label, key in PRESET_LABELS.items():
        parts = [f"{CRITERIA[c][1]} {int(w * 100)}%" for c, w in PRESETS[key].items() if w > 0]
        lines.append(f"- **{label}:** " + ", ".join(parts))
    return "\n".join(lines)


def info_boxes(gaps: pd.DataFrame) -> None:
    with st.expander("How the ratings work"):
        st.markdown(
            "Cars are scored with a method called TOPSIS. It finds the cars that are closest to the best "
            "result on every criterion at once, using the weights below.\n\n"
            + ranking_explanation()
            + "\n\n**Your rating** uses the priority you chose, among the cars that match your answers. "
            "**Index rating** uses the balanced weights across all cars on sale up to ₹50 lakh, so it does not "
            "change when you change your answers.\n\n"
            "Cars with no safety rating are scored on cost only (see the \"Rated on\" column), so compare ratings "
            "only between cars rated the same way.\n\n"
            "Safety badges: 🛡️ 85 or more, ✅ 70 to 84, ⚠️ 50 to 69, ❗ below 50. "
            "Scores from the older crash-test protocol are divided by 1.5, so newer, stricter tests count for more."
        )
    with st.expander("Known data gaps (please read)"):
        st.markdown(
            "- **Diesel:** the models listed below are sold with diesel engines in India, but their diesel variants are missing or incomplete in our data.\n"
            "- **Safety ratings:** about 4 in 10 on-sale variants have no rating in our data. This does not mean they are unsafe or untested. Please check the Bharat NCAP and Global NCAP websites.\n"
            "- **Comfort and convenience features** are not scored. Use the official links.\n"
            "- **Power, torque and size** are not used because that data is mostly missing. The price tier stands in for \"more car for the money\", which is only a rough guide.\n"
            "- **Price** means the ex-showroom price. On-road price is not shown. **Cost** means ex-showroom price plus fuel only. Insurance, service and registration are not included.\n"
            "- **EV comparison** uses the nearest-priced petrol variant of the sibling model, which favours a short payback.\n"
            "- **Mileage and EV range** are the claimed figures. Real-world numbers are usually lower."
        )
        st.dataframe(
            gaps.rename(columns={"brand": "Brand", "model": "Model", "note": "What is missing"}),
            hide_index=True,
        )


def qa_panel(cities: list):
    with st.container(border=True, key="qa"):
        st.subheader("Quick Q&A")

        preset_label = st.radio("What matters most to you?", list(PRESET_LABELS), horizontal=True)
        st.caption(PRESET_HELP.get(preset_label, ""))

        col_price, col_seats = st.columns([3, 1])
        price_low, price_high = col_price.slider("Price range, ex-showroom (₹ lakh)", 3.0, 50.0, (8.0, 15.0), 0.5)
        seats = col_seats.selectbox("Minimum seats", [4, 5, 6, 7], index=1)

        col_fuel, col_city, col_gear = st.columns([3, 1, 1])
        fuels = col_fuel.multiselect("Fuel types", FUELS, default=FUELS)
        city = col_city.selectbox(
            "City (fuel prices)", cities, index=cities.index("Delhi") if "Delhi" in cities else 0
        )
        transmission = col_gear.selectbox("Gearbox", ["Any", "Manual", "Automatic"])

        col_home, col_km = st.columns([1, 1])
        home_charging = col_home.checkbox("I can charge an electric car at home", value=True)
        annual_km = col_km.slider("Distance driven per year (km)", 3000, 30000, 12000, 1000)
    return preset_label, price_low, price_high, seats, fuels, transmission, city, annual_km, home_charging


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} | Car recommender for India", page_icon="🚗", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title(APP_NAME)
    st.markdown(f"**{TAGLINE}**")
    st.caption(
        "Answer a few questions and we rank cars for you using TOPSIS on 5-year cost, running cost and safety. "
        "Cost means the ex-showroom price plus fuel only. This is a personal portfolio project and not financial advice."
    )

    _, safety, _, price_table, _, links, gaps = load_static()
    cities = sorted(price_table["city"].dropna().tolist())

    preset_label, price_low, price_high, seats, fuels, transmission, city, annual_km, home_charging = qa_panel(cities)

    costs, comparisons, prices = build_costs(city, float(annual_km), home_charging)
    index = build_index(city, float(annual_km), home_charging)
    st.caption(
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
        st.warning("No car matches these answers. Try a wider price range, fewer seats, or more fuel types.")
    else:
        ranked = result.get("ranked", pd.DataFrame())
        model_count = len(ranked.drop_duplicates(["brand", "model"])) if not ranked.empty else 0
        st.write(f"**{result['shortlist_size']} variants** of **{model_count} models** match your answers.")

        if "Diesel" in fuels or not fuels:
            listed = ", ".join((gaps["brand"] + " " + gaps["model"]).head(8))
            st.warning(
                "Our diesel data is incomplete. Some models are sold with diesel engines, but their diesel variants "
                f"are missing or thin in our data, for example: {listed}. Diesel results may leave out good options. "
                "See \"Known data gaps\" at the bottom of the page."
            )

        top_group = next((g for g in ["rated", "all", "not rated"] if g in result["picks"]), None)
        picks_df = None
        if top_group:
            st.header("🏆 Your top 3 picks")
            if top_group == "not rated":
                st.info("None of these cars has a safety rating in our data, so they are ranked on cost only.")
            picks_df = result["picks"][top_group].merge(index, on=KEY, how="left")
            picks = list(picks_df.itertuples(index=False))
            if picks:
                columns = st.columns(len(picks))
                for rank, (column, pick) in enumerate(zip(columns, picks), start=1):
                    with column:
                        show_pick(rank, pick, comparisons, links, gaps)

        by_fuel = result.get("by_fuel", pd.DataFrame())
        if len(by_fuel) > 1:
            st.header("Best safety-rated pick for each fuel type")
            st.dataframe(by_fuel_table(by_fuel), hide_index=True)

        table = cars_table(ranked, index, links, picks_df)
        if not table.empty:
            st.header("All matching cars")
            shown = filter_table(table)
            st.caption(f"Showing {len(shown)} of {len(table)} variants. Click a column header to sort.")
            st.dataframe(shown, hide_index=True, height=520, column_config=TABLE_CONFIG)
            st.caption(
                "Compare ratings only between cars with the same \"Rated on\" value. A blank safety score means "
                "no rating in our data, which does not mean the car is unsafe."
            )

    info_boxes(gaps)


main()