{% macro cleanup_run_external_tables() %}
    {%- set ingest_timestamp = run_started_at.strftime('%Y%m%dT%H%M%S') -%}
    {%- set external_dataset = 'ecommerce_product' -%}
    {%- set external_table_prefix = 'external_ingest_' ~ ingest_timestamp -%}

    {% set get_external_tables_sql %}
        SELECT
            table_id
        FROM
            `{{ external_dataset }}`.__TABLES__
        WHERE
            table_id LIKE '{{ external_table_prefix }}%'
            AND type = 3 -- Type 3 indicates an external table
    {% endset %}

    {% set results = run_query(get_external_tables_sql) %}

    {% if execute %}
        {% set external_tables_to_drop = results.columns[0].values() %}

        {% for table_id in external_tables_to_drop %}
            {% set drop_sql %}
                DROP EXTERNAL TABLE IF EXISTS `{{ external_dataset }}`.`{{ table_id }}`;
            {% endset %}
            {% do run_query(drop_sql) %}
            {{ log("Dropped external table: " ~ external_dataset ~ "." ~ table_id, info=True) }}
        {% endfor %}
    {% endif %}

    {% do return('') %}
{% endmacro %}
