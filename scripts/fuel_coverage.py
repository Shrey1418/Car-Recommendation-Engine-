import sqlite3

import pandas as pd

con = sqlite3.connect("data/processed/car_data.db")
variants = pd.read_sql("SELECT brand, model, variant, fuel, ex_showroom_price, on_sale FROM variants", con)
con.close()

on_sale = variants[variants["on_sale"] == 1]
print("On-sale variants per fuel:")
print(on_sale["fuel"].value_counts().to_string())

combo = on_sale.groupby(["brand", "model"])["fuel"].agg(lambda s: " + ".join(sorted(set(s))))
multi = combo[combo.str.contains(r"\+")]
print()
print("Models with 2+ fuels:", len(multi), "of", len(combo))
print(multi.value_counts().to_string())

print()
print("CNG variants:")
cng = on_sale[on_sale["fuel"] == "CNG"][["brand", "model", "variant", "ex_showroom_price"]]
print(cng.to_string(index=False))