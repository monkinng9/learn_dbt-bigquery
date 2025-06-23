import polars as pl

table_path = "data/demo_data/processed/product_delta"

delta_table = pl.scan_delta(table_path)

dirpath = "data/demo_data/processed/delta_to_csv"

path = "data/demo_data/processed/product_delta.csv"

delta_table.collect().write_csv(path, separator=",")