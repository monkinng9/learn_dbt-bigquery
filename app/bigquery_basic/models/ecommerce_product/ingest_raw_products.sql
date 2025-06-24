{%- set ingest_date_str = var('ingest_date', run_started_at.strftime('%Y-%m-%d')) -%}
{%- set ingest_timestamp = run_started_at.strftime('%Y%m%dT%H%M%S') -%}

{%- set source_table_name = 'product' -%}
{%- set external_dataset = 'ecommerce_product' -%}
{%- set external_table_name = 'external_ingest_' ~ ingest_timestamp ~ '_' ~ source_table_name -%}
{%- set external_table_id = '`' ~ external_dataset ~ '.' ~ external_table_name ~ '`' -%}
{%- set gcs_path = 'gs://demo-data-pipeline-beyondbegin/ecommerce_product/raw_data/ecommerce_product_price/ingest_date=' ~ ingest_date_str ~ '/*.csv' -%}

{% set create_external_table_sql = create_external_table(external_table_id, gcs_path) %}
{% set drop_external_table_sql = drop_external_table(external_table_id) %}

{{ config(
    materialized='incremental',
    alias='product_price_raw_data',
    unique_key='product_name_sha256',
    pre_hook=[create_external_table_sql],
    post_hook=[drop_external_table_sql]
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