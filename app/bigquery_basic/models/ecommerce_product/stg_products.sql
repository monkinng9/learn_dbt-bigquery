{{ config(
    materialized='incremental',
    unique_key='product_name'
) }}

SELECT
    product_name,
    sale_price,
    PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', ingest_timestamp_utc) AS ingest_timestamp_utc,
    ecommerce_name
FROM
    {{ ref('raw_product') }}

{% if is_incremental() %}
  -- This filter will only be applied on incremental runs
  -- It assumes 'ingest_timestamp_utc' can be used to identify new or updated records
  -- compared to the max timestamp already in the target table.
  WHERE ingest_timestamp_utc > (SELECT MAX(ingest_timestamp_utc) FROM {{ this }})
{% endif %}
