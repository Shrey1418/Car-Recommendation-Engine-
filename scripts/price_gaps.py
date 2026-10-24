import sqlite3

import pandas as pd

con = sqlite3.connect("data/processed/car_data.db")
variants = pd.read_sql("SELECT brand, model, variant, fuel, ex_showroom_price, on_sale FROM variants", con)
con.close()

on_sale = variants[variants["on_sale"] == 1]
print("On-sale variants:", len(on_sale))
print("Without price:", on_sale["ex_showroom_price"].isna().sum())

by_model = on_sale.groupby(["brand", "model"])["ex_showroom_price"].agg(total="size", priced="count").reset_index()
print("On-sale models:", len(by_model))
print("Models with no priced variant at all:", (by_model["priced"] == 0).sum())
print("Models with some but not all priced:", ((by_model["priced"] > 0) & (by_model["priced"] < by_model["total"])).sum())
print(by_model[by_model["priced"] == 0].to_string(index=False))