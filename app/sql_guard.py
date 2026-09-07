import re

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from .config import get_settings
from .schemas import DatabaseKind, SchemaInfo


FORBIDDEN_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|merge|copy|grant|"
    r"revoke|call|execute|vacuum|analyze|refresh|set|show|use|lock|unlock)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    pass


def validate_and_limit_sql(sql: str, schema: SchemaInfo, dialect: DatabaseKind) -> str:
    candidate = sql.strip().rstrip(";").strip()
    if not candidate:
        raise UnsafeQueryError("SQL is empty")
    if ";" in candidate:
        raise UnsafeQueryError("Only one SQL statement is allowed")
    if FORBIDDEN_PATTERN.search(candidate):
        raise UnsafeQueryError("Only read-only SELECT queries are allowed")

    sqlglot_dialect = "postgres" if dialect == "postgresql" else "mysql"
    try:
        expression = parse_one(candidate, read=sqlglot_dialect)
    except ParseError as exc:
        raise UnsafeQueryError(f"Invalid SQL: {exc}") from exc

    if not isinstance(expression, (exp.Select, exp.Union)):
        raise UnsafeQueryError("The statement must be SELECT or WITH ... SELECT")

    cte_names = {cte.alias_or_name.lower() for cte in expression.find_all(exp.CTE)}
    referenced_tables = {
        table.name.lower()
        for table in expression.find_all(exp.Table)
        if table.name.lower() not in cte_names
    }
    allowed_tables = {table.name.lower() for table in schema.tables}
    unknown_tables = referenced_tables - allowed_tables
    if unknown_tables:
        raise UnsafeQueryError(f"Unknown table(s): {', '.join(sorted(unknown_tables))}")
    if not referenced_tables:
        raise UnsafeQueryError("The query must read from at least one known table")

    max_rows = get_settings().max_query_rows
    limit = expression.args.get("limit")
    if limit is None:
        expression = expression.limit(max_rows)
    else:
        limit_value = limit.expression
        if isinstance(limit_value, exp.Literal) and limit_value.is_int and int(limit_value.this) > max_rows:
            expression.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    return expression.sql(dialect=sqlglot_dialect)
