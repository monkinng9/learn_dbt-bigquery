{% macro create_external_table(external_table_id, gcs_path) %}
    {% set sql %}
CREATE EXTERNAL TABLE {{ external_table_id }} (
    product_name_sha256 STRING,
    product_name STRING,
    sale_price FLOAT64,
    ingest_timestamp_utc STRING,
    ecommerce_name STRING
)
OPTIONS (
    format = 'CSV',
    uris = ['{{ gcs_path }}'],
    skip_leading_rows = 1,
    description = 'External table for ingesting product data from GCS for run {{ run_started_at.isoformat() }}'
)
    {% endset %}
    {% do return(sql) %}
{% endmacro %}
