# Car Recommendation Engine, India

A needs-based car recommender for the Indian market. You enter a price range, seats, fuel types and what matters most to you. The app ranks cars using TOPSIS on 5-year cost, running cost and crash-test safety, and explains each pick.

**Live app:** https://7z92f9uefywnvu6bhjgchu.streamlit.app

This is a personal portfolio project, not financial advice.

## What it does

- Covers about 900 car variants in India (about 690 currently on sale), up to ₹50 lakh ex-showroom.
- Filters by price range, seats, fuel, gearbox and city.
- Ranks cars with TOPSIS using one of three priorities: balanced, family safety first, or lowest cost to own.
- Works out a 5-year cost (ex-showroom price plus fuel) and the payback of CNG and electric cars against a petrol twin.
- Shows safety with crash-test scores from Global NCAP and Bharat NCAP, with newer, stricter tests counting for more.
- Lists every model in your range and highlights the top 3.
- Links to each model's official page for comfort and convenience features.

## How the ranking works

1. **Hard filters** remove any car outside your price range, seats, fuel or gearbox, or not on sale.
2. **Criteria** are scored with TOPSIS, which finds the cars closest to the best result on every criterion at once:

| Priority | 5-year cost | Running cost per km | Safety | Price tier |
|---|---|---|---|---|
| Balanced | 30% | 20% | 30% | 20% |
| Family: safety first | 15% | 10% | 55% | 20% |
| Lowest cost to own | 50% | 35% | 15% | 0% |

   "Price tier" rewards a higher price inside your range as a rough guide to more car for the money, because size, power and feature data is mostly missing.
3. **Safety score** is the average of the adult and child crash-test scores. Scores from the older Global NCAP protocol are divided by 1.5. Cars with no rating are ranked separately on cost only, and are never given a guessed score.

## Cost assumptions

- 12,000 km a year and 5 years of ownership by default, with no fuel price growth. Both are adjustable in the app.
- Fuel prices are city-level petrol, diesel and CNG prices from energy.thecore.in, as of October 2026. This is a third-party aggregator, not an official oil-company source, and prices change often.
- Electricity is a blend of 80% home charging at ₹6 per kWh and 20% public fast charging at ₹20 per kWh. Untick home charging in the app to use the public price.
- Insurance, service, registration and charging losses are not included.
- Mileage and EV range are the manufacturers' claimed figures. Real-world numbers are usually lower.

## Validation

I ran 15 scenario personas (a first-time buyer, a family of seven, a high-km commuter, a diesel SUV seeker, impossible filters and others) through the same code as the app. Every pick respected its filters. The test found three real problems. Two are fixed. The diesel one is only disclosed in the app, because the data is still missing:

1. Diesel gaps were not visible to the user. The app now warns about them.
2. A family without home charging was shown only electric cars. The app now has a home-charging switch and a "best pick for each fuel type" table.
3. Wide price ranges returned only cheap cars. Filters now use a price range, with a price-tier criterion.

| Result | First pass | After fixes |
|---|---|---|
| Pass | 9 | [fill] |
| Partial | 5 | [fill] |
| Fail | 1 | [fill] |

This is a sanity check on 15 scenarios I designed, not a user study or an accuracy measure. Run it with `python -m car_recommender.components.persona_test`.

## Known gaps

- **Diesel:** many popular diesel models (for example Creta, Nexon, Harrier, Safari, Seltos) are missing or incomplete in the data. The app says so.
- **Safety:** about 4 in 10 on-sale variants have no rating in the data. This does not mean they are unsafe.
- **Features:** comfort and convenience features are not scored. The app links to the official pages.
- **Power, torque and size** are mostly missing, so they are not used.
- **Links:** some official links are brand homepages or could not be verified.
- **EV comparison** uses the nearest-priced petrol variant of the sibling model, which favours a short payback.

## Tech stack

| Area | What was used |
|---|---|
| Language | Python 3.13 |
| Data handling | pandas, NumPy, pandera, rapidfuzz, PyYAML |
| Data collection | requests, requests-cache, BeautifulSoup, lxml, pypdf |
| Storage | SQLite |
| Ranking | TOPSIS written from scratch with NumPy and pandas |
| App | Streamlit, hosted on Streamlit Community Cloud |

## Run it locally

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    streamlit run app\streamlit_app.py

The app reads the small files in `app_data/`. The raw, scraped and intermediate data used to build them is not in this repo. To rebuild, place the two Kaggle datasets in `data/raw/` and run the pipeline in this order:

    python -m car_recommender.components.data_merging
    python -m car_recommender.components.cng_manual
    python -m car_recommender.components.data_validation
    python -m car_recommender.components.database_load
    python -m car_recommender.components.cost_model
    python -m car_recommender.components.export_app_data

## Data sources and credits

- **Car catalog and specifications:** two Kaggle datasets, used under their MIT licence:
  - [Indian Cars Dataset](https://www.kaggle.com/datasets/atharvanilawar/indian-cars-dataset) by Atharva Nilawar
  - [Car Dataset](https://www.kaggle.com/datasets/shreyrmishra/car-dataset) by SHREY.R.MISHRA
- **Safety ratings:** public crash-test results from Global NCAP and Bharat NCAP.
- **Hand-entered data:** CNG and some petrol variants (prices and claimed mileage) were typed in by hand from public brand and price-list pages, as of October 2026. Always confirm prices and features on the brand's official site.
- **Fuel prices and links:** city fuel prices and official model-page links are compiled in `data/manual/`.
- The repo ships only the small `app_data/` files the app needs.
- If you own any of this data and want it removed, please open an issue.

## Planned next

Diesel data for the top models, fixing the official links, a resale-value model, natural-language input and an API. None of these are built yet.