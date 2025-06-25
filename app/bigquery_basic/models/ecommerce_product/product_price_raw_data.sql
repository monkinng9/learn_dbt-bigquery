{%- set ingest_date_str = var('ingest_date', run_started_at.strftime('%Y-%m-%d')) -%}

{%- set source_table_name = 'product' -%}
{%- set external_dataset = 'ecommerce_product' -%}
{%- set external_table_name = 'external_ingest_' ~ source_table_name -%}
{%- set external_table_id = '`' ~ external_dataset ~ '.' ~ external_table_name ~ '`' -%}
{%- set gcs_path = 'gs://demo-data-pipeline-beyondbegin/ecommerce_product/raw_data/ecommerce_product_price/ingest_date=' ~ ingest_date_str ~ '/*.csv' -%}

{% set create_external_table_sql = create_external_table(external_table_id, gcs_path) %}

{{ config(
    materialized='incremental',
    alias='product_price_raw_data',
    unique_key=['product_name_sha256','ecommerce_name'],
    pre_hook=[create_external_table_sql]
) }}

WITH source_data AS (
    SELECT
        -- Generate SHA256 hash from the processed product_name
        SHA256(TRIM(REGEXP_REPLACE(LOWER(product_name), r'[^a-z0-9\s\p{Thai}+\-&]', ''))) AS product_name_sha256,
        -- Process product_name:
        -- 1. Convert to lowercase
        -- 2. Remove special characters, preserving English, Thai, numbers, spaces, '+', '-', '&', and parentheses
        -- 3. Trim whitespace
        TRIM(REGEXP_REPLACE(LOWER(product_name), r'[^a-z0-9\s\p{Thai}+\-&()]', '')) AS product_name,
        sale_price,
        ecommerce_name,
        PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', ingest_timestamp_utc) AS ingest_timestamp_utc
    FROM
        {{ external_table_id }}
)
SELECT
    s.* 
FROM 
    source_data s
{% if is_incremental() %}
WHERE ingest_timestamp_utc > (SELECT MAX(ingest_timestamp_utc) FROM {{ this }})
-- To handle cases where a product is updated multiple times in the same batch,
-- we need to select only the latest version of each product.
QUALIFY ROW_NUMBER() OVER (PARTITION BY product_name_sha256 ORDER BY ingest_timestamp_utc DESC) = 1
{% endif %}