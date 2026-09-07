import json
from typing import Any

from openai import OpenAI

from .config import get_settings
from .datasources import DataSource, execute_read_query, schema_to_prompt
from .schemas import DashboardPlan, DashboardResponse, Insight, WidgetPlan
from .sql_guard import UnsafeQueryError, validate_and_limit_sql
from .visuals import materialize_widget


def build_dashboard(source: DataSource, prompt: str | None = None) -> DashboardResponse:
    settings = get_settings()
    mode = "openai" if settings.openai_api_key else "heuristic"
    objective = prompt or "Create an executive overview dashboard for this database."
    plan = _openai_plan(source, objective) if settings.openai_api_key else _heuristic_plan(source, objective)
    widgets = []
    errors = []

    for index, widget_plan in enumerate(plan.widgets):
        try:
            safe_sql = validate_and_limit_sql(widget_plan.sql, source.schema, source.kind)
            data = execute_read_query(source, safe_sql)
            widgets.append(materialize_widget(widget_plan, data, safe_sql, index))
        except (UnsafeQueryError, Exception) as exc:
            errors.append(f"{widget_plan.title}: {exc}")

    if not widgets:
        detail = "; ".join(errors) if errors else "No widgets could be generated"
        raise ValueError(detail)

    insight = _openai_insight(objective, plan, widgets) if settings.openai_api_key else plan.message
    if errors:
        insight = f"{insight} Some widgets were skipped: {'; '.join(errors[:2])}"

    return DashboardResponse(
        title=plan.title,
        message=insight,
        mode=mode,
        widgets=widgets,
        schema=source.schema,
    )


def _openai_plan(source: DataSource, prompt: str) -> DashboardPlan:
    schema_context = schema_to_prompt(source.schema)
    system = f"""
You are a senior BI Copilot for user-connected {source.kind} databases.
Turn the user's request into a complete dashboard plan using Data Augmented
Generation: reason from the live database schema, generate read-only SQL, then
choose useful visualizations.

Rules:
- Respond in the same language as the user.
- Use only tables and columns from the schema below.
- Every SQL query must be a single read-only SELECT or WITH ... SELECT statement.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY, SET or stored procedure calls.
- Aggregate before visualizing; limit detail tables to concise rows.
- Prefer KPI for one scalar, line/area for time, bar for category comparison,
  pie only for <= 7 parts, scatter for numeric relationships, table for details.
- For KPI queries, select exactly one scalar aliased as value and set value_field to value.
- x_field, y_fields, series_field, and value_field must exactly match SQL aliases.
- Build 3-6 widgets for a dashboard request, or 1-3 focused widgets for a narrow question.
- If schema meaning is ambiguous, choose conservative overview metrics and say so in message.

Schema:
{schema_context}
"""
    response = OpenAI(api_key=get_settings().openai_api_key).responses.parse(
        model=get_settings().openai_model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        text_format=DashboardPlan,
    )
    parsed = getattr(response, "output_parsed", None)
    if isinstance(parsed, DashboardPlan):
        return parsed
    for output in response.output:
        for item in getattr(output, "content", []):
            value = getattr(item, "parsed", None)
            if isinstance(value, DashboardPlan):
                return value
    raise ValueError("The model did not return a dashboard plan")


def _openai_insight(prompt: str, plan: DashboardPlan, widgets: list[Any]) -> str:
    evidence = [
        {
            "title": widget.title,
            "chart_type": widget.chart_type,
            "data": widget.data[:20],
        }
        for widget in widgets
    ]
    response = OpenAI(api_key=get_settings().openai_api_key).responses.parse(
        model=get_settings().openai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "Write a concise BI insight in the same language as the user. "
                    "Ground every claim in the supplied query results. Use 2-4 sentences."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": prompt, "dashboard": plan.title, "widgets": evidence},
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ],
        text_format=Insight,
    )
    parsed = getattr(response, "output_parsed", None)
    if isinstance(parsed, Insight):
        return parsed.message
    return plan.message


def _heuristic_plan(source: DataSource, prompt: str) -> DashboardPlan:
    tables = source.schema.tables
    if not tables:
        raise ValueError("No database tables were found")

    fact = max(tables, key=lambda table: len(table.columns))
    columns = fact.columns
    numeric_columns = [
        column.name
        for column in columns
        if any(token in column.type.lower() for token in ("int", "decimal", "numeric", "float", "double", "real"))
    ]
    date_columns = [
        column.name
        for column in columns
        if any(token in column.type.lower() for token in ("date", "time", "year"))
    ]
    text_columns = [
        column.name
        for column in columns
        if any(token in column.type.lower() for token in ("char", "text", "varchar", "string"))
    ]

    widgets: list[WidgetPlan] = [
        WidgetPlan(
            title=f"Rows in {fact.name}",
            description="Total records in the largest table",
            chart_type="kpi",
            sql=f"SELECT COUNT(*) AS value FROM {fact.name}",
            value_field="value",
            number_format="compact",
        )
    ]

    if numeric_columns:
        metric = numeric_columns[0]
        widgets.append(
            WidgetPlan(
                title=f"Average {metric}",
                description=f"Mean value from {fact.name}",
                chart_type="kpi",
                sql=f"SELECT AVG({metric}) AS value FROM {fact.name}",
                value_field="value",
                number_format="decimal",
            )
        )

    if date_columns and numeric_columns:
        date_col = date_columns[0]
        metric = numeric_columns[0]
        bucket = _date_bucket(source.kind, date_col)
        widgets.append(
            WidgetPlan(
                title=f"{metric} trend",
                description=f"Monthly trend based on {date_col}",
                chart_type="line",
                sql=(
                    f"SELECT {bucket} AS period, SUM({metric}) AS total "
                    f"FROM {fact.name} WHERE {date_col} IS NOT NULL "
                    f"GROUP BY period ORDER BY period"
                ),
                x_field="period",
                y_fields=["total"],
                number_format="compact",
            )
        )

    if text_columns and numeric_columns:
        dimension = text_columns[0]
        metric = numeric_columns[0]
        widgets.append(
            WidgetPlan(
                title=f"Top {dimension}",
                description=f"Highest {metric} by {dimension}",
                chart_type="bar",
                sql=(
                    f"SELECT {dimension} AS category, SUM({metric}) AS total "
                    f"FROM {fact.name} WHERE {dimension} IS NOT NULL "
                    f"GROUP BY {dimension} ORDER BY total DESC LIMIT 10"
                ),
                x_field="category",
                y_fields=["total"],
                number_format="compact",
            )
        )

    widgets.append(
        WidgetPlan(
            title=f"Sample from {fact.name}",
            description="Recent records for quick inspection",
            chart_type="table",
            sql=f"SELECT * FROM {fact.name} LIMIT 20",
            number_format="number",
        )
    )

    return DashboardPlan(
        title="Data Augmented Dashboard",
        message=(
            "Dashboard generated with heuristic planning because OPENAI_API_KEY is not configured. "
            "It uses schema introspection, safe SQL validation and live query results."
        ),
        widgets=widgets[:6],
    )


def _date_bucket(kind: str, column: str) -> str:
    if kind == "postgresql":
        return f"DATE_TRUNC('month', {column})"
    return f"DATE_FORMAT({column}, '%Y-%m-01')"
