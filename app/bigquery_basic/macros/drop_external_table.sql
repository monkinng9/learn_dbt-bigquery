{% macro drop_external_table(external_table_id) %}
    {% set sql %}
DROP TABLE IF EXISTS {{ external_table_id }}
    {% endset %}
    {% do return(sql) %}
{% endmacro %}
