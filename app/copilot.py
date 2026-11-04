import json
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .config import get_settings
from .datasources import DataSource, execute_read_query, schema_to_prompt
from .schemas import DashboardPlan, DashboardResponse, Insight, WidgetPlan
from .sql_guard import UnsafeQueryError, validate_and_limit_sql
from .visuals import materialize_widget


class DAGState(TypedDict, total=False):
    question: str
    relevance: str
    sql_query: str
    query_rows: list[dict]
    sql_error: str
    attempts: int
    answer: "Answer"


class Relevance(BaseModel):
    relevance: Literal["relevant", "not_relevant"]


class SQLDraft(BaseModel):
    sql_query: str = Field(min_length=8, max_length=8_000)


class Answer(BaseModel):
    title: str = Field(min_length=4, max_length=90)
    message: str = Field(min_length=10, max_length=700)


MAX_SQL_REPAIR_ATTEMPTS = 2


def build_dashboard(source: DataSource, prompt: str | None = None) -> DashboardResponse:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the LangGraph data-augmented workflow.")
    question = prompt or "Create an executive overview dashboard for this database."
    schema_context = schema_to_prompt(source.schema)

    def llm():
        return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0)

    def check_relevance(state: DAGState) -> DAGState:
        result = llm().with_structured_output(Relevance).invoke(
            f"Determine whether this question can be answered using the database schema. Reply with structured output only.\nSchema:\n{schema_context}\nQuestion: {state['question']}"
        )
        return {"relevance": result.relevance}

    def generate_sql(state: DAGState) -> DAGState:
        result = llm().with_structured_output(SQLDraft).invoke(
            f"Generate one read-only {source.kind} SELECT or WITH...SELECT statement for the question. Use only this schema. Never generate writes, DDL, multiple statements or explanations.\nSchema:\n{schema_context}\nQuestion: {state['question']}"
        )
        return {"sql_query": result.sql_query}

    def execute_sql(state: DAGState) -> DAGState:
        try:
            safe_sql = validate_and_limit_sql(state["sql_query"], source.schema, source.kind)
            return {"sql_query": safe_sql, "query_rows": execute_read_query(source, safe_sql), "sql_error": ""}
        except Exception as exc:
            return {"sql_error": str(exc)}

    def repair_sql(state: DAGState) -> DAGState:
        result = llm().with_structured_output(SQLDraft).invoke(
            f"Repair this failed read-only SQL. Use only the schema and return one SQL statement. Error: {state['sql_error']}\nSchema:\n{schema_context}\nSQL: {state['sql_query']}"
        )
        return {"sql_query": result.sql_query, "attempts": state.get("attempts", 0) + 1}

    def answer(state: DAGState) -> DAGState:
        result = llm().with_structured_output(Answer).invoke(
            f"Write a concise answer in the user's language using only these query rows. Do not invent facts. Question: {state['question']}\nRows: {json.dumps(state.get('query_rows', []), default=str)[:12000]}"
        )
        return {"answer": result}

    def unrelated(_: DAGState) -> DAGState:
        return {"answer": Answer(title="Data Augmented BI Copilot", message="Câu hỏi này không liên quan đến dữ liệu trong database đã kết nối.")}

    graph = StateGraph(DAGState)
    graph.add_node("check_relevance", check_relevance); graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql); graph.add_node("repair_sql", repair_sql)
    graph.add_node("generate_answer", answer); graph.add_node("unrelated", unrelated)
    graph.set_entry_point("check_relevance")
    graph.add_conditional_edges("check_relevance", lambda state: "generate_sql" if state["relevance"] == "relevant" else "unrelated")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_conditional_edges("execute_sql", lambda state: "generate_answer" if not state.get("sql_error") else ("repair_sql" if state.get("attempts", 0) < MAX_SQL_REPAIR_ATTEMPTS else "unrelated"))
    graph.add_edge("repair_sql", "execute_sql"); graph.add_edge("generate_answer", END); graph.add_edge("unrelated", END)
    outcome = graph.compile().invoke({"question": question, "attempts": 0})
    answer_result = outcome["answer"]
    rows = outcome.get("query_rows", [])
    if not rows:
        return DashboardResponse(title=answer_result.title, message=answer_result.message, mode="openai", widgets=[], schema=source.schema)
    fields = list(rows[0]); numeric = next((field for field in fields if isinstance(rows[0].get(field), (int, float))), None)
    chart_type = "kpi" if len(rows) == 1 and len(fields) == 1 else ("bar" if len(fields) >= 2 and numeric else "table")
    plan = WidgetPlan(title="Query result", description="Generated by LangGraph DAG", chart_type=chart_type, sql=outcome["sql_query"], value_field=fields[0] if chart_type == "kpi" else None, x_field=fields[0] if chart_type == "bar" else None, y_fields=[numeric] if chart_type == "bar" and numeric else [], number_format="compact")
    return DashboardResponse(title=answer_result.title, message=answer_result.message, mode="openai", widgets=[materialize_widget(plan, rows, outcome["sql_query"], 0)], schema=source.schema)


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
