import polars as pl

data_path = [
    "/opt/spark/work-dir/data/demo_data/ecommerce_product_20250619T030330.csv",
    "/opt/spark/work-dir/data/demo_data/ecommerce_product_20250620T030330.csv"
]

df = pl.concat([pl.read_csv(path) for path in data_path])

df = df.unique(subset=['product_name', 'ingest_timestamp_utc', 'ecommerce_name'])

df.filter(pl.col("ingest_timestamp_utc") == '2025-06-19 3:03:30').write_csv(
    "/opt/spark/work-dir/data/processed_data/ecommerce_product_20250619T030330.csv"
)

df.filter(pl.col("ingest_timestamp_utc") == '2025-06-20 3:03:30').write_csv(
    "/opt/spark/work-dir/data/processed_data/ecommerce_product_20250620T030330.csv"
)
