import pytest

from app.schemas import ColumnInfo, SchemaInfo, TableInfo
from app.sql_guard import UnsafeQueryError, validate_and_limit_sql


SCHEMA = SchemaInfo(
    dialect="postgresql",
    database="demo",
    tables=[
        TableInfo(
            name="orders",
            columns=[
                ColumnInfo(name="id", type="INTEGER"),
                ColumnInfo(name="amount", type="NUMERIC"),
            ],
        )
    ],
)


def test_allows_select_and_adds_limit():
    sql = validate_and_limit_sql("SELECT id, amount FROM orders", SCHEMA, "postgresql")
    assert "FROM orders" in sql
    assert "LIMIT" in sql


def test_blocks_destructive_sql():
    with pytest.raises(UnsafeQueryError):
        validate_and_limit_sql("DELETE FROM orders", SCHEMA, "postgresql")


def test_blocks_unknown_table():
    with pytest.raises(UnsafeQueryError):
        validate_and_limit_sql("SELECT * FROM users", SCHEMA, "postgresql")
