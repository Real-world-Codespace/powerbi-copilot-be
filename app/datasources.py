from dataclasses import dataclass
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import URL, create_engine, inspect, text
from sqlalchemy.engine import Engine

from .config import get_settings
from .schemas import (
    ColumnInfo,
    DataSourceConnectRequest,
    DataSourceSummary,
    DatabaseKind,
    SchemaInfo,
    TableInfo,
)


@dataclass
class DataSource:
    id: str
    kind: DatabaseKind
    database: str
    schema_name: str | None
    engine: Engine
    schema: SchemaInfo


class DataSourceRegistry:
    def __init__(self) -> None:
        self._items: dict[str, DataSource] = {}

    def add(self, request: DataSourceConnectRequest) -> DataSource:
        engine = create_engine(
            _build_url(request),
            pool_pre_ping=True,
            pool_recycle=900,
            connect_args=_connect_args(request),
        )
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        schema = introspect_schema(
            engine=engine,
            dialect=request.kind,
            database=request.database,
            schema_name=request.schema_name,
        )
        source = DataSource(
            id=str(uuid4()),
            kind=request.kind,
            database=request.database,
            schema_name=request.schema_name,
            engine=engine,
            schema=schema,
        )
        self._items[source.id] = source
        return source

    def get(self, connection_id: str) -> DataSource:
        try:
            return self._items[connection_id]
        except KeyError as exc:
            raise KeyError("Unknown or expired data source connection") from exc

    def list(self) -> list[DataSourceSummary]:
        return [
            DataSourceSummary(
                connection_id=item.id,
                kind=item.kind,
                database=item.database,
                schema_name=item.schema_name,
                table_count=len(item.schema.tables),
            )
            for item in self._items.values()
        ]


registry = DataSourceRegistry()


def _build_url(request: DataSourceConnectRequest) -> URL:
    driver = "postgresql+psycopg" if request.kind == "postgresql" else "mysql+pymysql"
    return URL.create(
        drivername=driver,
        username=request.username,
        password=request.password.get_secret_value(),
        host=request.host,
        port=request.port,
        database=request.database,
    )


def _connect_args(request: DataSourceConnectRequest) -> dict:
    if request.kind == "postgresql":
        args: dict = {"connect_timeout": get_settings().query_timeout_seconds}
        if request.use_ssl:
            args["sslmode"] = "require"
        return args
    args = {
        "connect_timeout": get_settings().query_timeout_seconds,
        "read_timeout": get_settings().query_timeout_seconds,
        "write_timeout": get_settings().query_timeout_seconds,
    }
    if request.use_ssl:
        args["ssl"] = {}
    return args


def introspect_schema(
    engine: Engine,
    dialect: DatabaseKind,
    database: str,
    schema_name: str | None,
) -> SchemaInfo:
    inspector = inspect(engine)
    table_names = inspector.get_table_names(schema=schema_name)
    tables: list[TableInfo] = []

    for table_name in table_names[:80]:
        pk_columns = set(inspector.get_pk_constraint(table_name, schema=schema_name).get("constrained_columns") or [])
        foreign_keys: dict[str, str] = {}
        for fk in inspector.get_foreign_keys(table_name, schema=schema_name):
            referred_table = fk.get("referred_table")
            referred_columns = fk.get("referred_columns") or []
            for column in fk.get("constrained_columns") or []:
                if referred_table and referred_columns:
                    foreign_keys[column] = f"{referred_table}.{referred_columns[0]}"

        columns = [
            ColumnInfo(
                name=column["name"],
                type=str(column["type"]),
                nullable=bool(column.get("nullable", True)),
                primary_key=column["name"] in pk_columns,
                foreign_key=foreign_keys.get(column["name"]),
            )
            for column in inspector.get_columns(table_name, schema=schema_name)
        ]
        tables.append(TableInfo(name=table_name, columns=columns))

    return SchemaInfo(dialect=dialect, database=database, schema_name=schema_name, tables=tables)


def schema_to_prompt(schema: SchemaInfo) -> str:
    lines = [f"{schema.dialect.upper()} database: {schema.database}"]
    if schema.schema_name:
        lines.append(f"Schema namespace: {schema.schema_name}")
    for table in schema.tables:
        lines.append(f"Table: {table.name}")
        for column in table.columns:
            tags = []
            if column.primary_key:
                tags.append("primary key")
            if column.foreign_key:
                tags.append(f"foreign key to {column.foreign_key}")
            suffix = f" ({', '.join(tags)})" if tags else ""
            lines.append(f"- {column.name}: {column.type}{suffix}")
    return "\n".join(lines)


def execute_read_query(source: DataSource, sql: str) -> list[dict]:
    with source.engine.connect() as connection:
        rows = connection.execute(text(sql)).mappings().fetchall()
    return [_json_safe(dict(row)) for row in rows]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    return value


def suggested_questions(schema: SchemaInfo) -> list[str]:
    table_names = [table.name for table in schema.tables[:6]]
    if not table_names:
        return ["Có những bảng dữ liệu nào?", "Tạo dashboard tổng quan từ database này"]
    first_table = table_names[0]
    return [
        f"Tạo dashboard tổng quan cho dữ liệu trong {first_table}",
        f"Top 10 bản ghi quan trọng nhất trong {first_table} là gì?",
        "Có xu hướng nào theo thời gian trong dữ liệu này không?",
        f"So sánh các nhóm/chủng loại chính từ các bảng: {', '.join(table_names[:3])}",
    ]
