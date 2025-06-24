{%- set ingest_date_str = var('ingest_date', run_started_at.strftime('%Y-%m-%d')) -%}
{%- set ingest_timestamp = run_started_at.strftime('%Y%m%dT%H%M') -%}

{%- set source_table_name = 'product' -%}
{%- set external_dataset = 'ecommerce_product' -%}
{%- set external_table_name = 'external_ingest_' ~ ingest_timestamp ~ '_' ~ source_table_name -%}
{%- set external_table_id = '`' ~ external_dataset ~ '.' ~ external_table_name ~ '`' -%}
{%- set gcs_path = 'gs://demo-data-pipeline-beyondbegin/ecommerce_product/raw_data/ecommerce_product_price/ingest_date=' ~ ingest_date_str ~ '/*.csv' -%}

{#-
  Pre-render the complex SQL for the pre-hook into a variable.
  This resolves Jinja scoping issues with the config block.
-#}
{% set create_external_table_sql %}
CREATE OR REPLACE EXTERNAL TABLE {{ external_table_id }} (
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

{#- Pre-render the SQL for the post-hook into a variable. -#}
{% set drop_external_table_sql = "DROP TABLE IF EXISTS " ~ external_table_id %}

{{ config(
    materialized='incremental',
    alias='product_price_raw_data',
    unique_key='product_name_sha256',
    pre_hook=create_external_table_sql,
    post_hook=drop_external_table_sql
) }}

WITH source_data AS (
    SELECT
        CAST(product_name_sha256 AS BYTES) AS product_name_sha256,
        product_name,
        sale_price,
        PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', ingest_timestamp_utc) AS ingest_timestamp_utc,
        ecommerce_name
    FROM
        {{ external_table_id }}
)

SELECT * FROM source_data

{% if is_incremental() %}

  -- filter for new records only
  WHERE ingest_timestamp_utc > (SELECT MAX(ingest_timestamp_utc) FROM {{ this }})

{% endif %}