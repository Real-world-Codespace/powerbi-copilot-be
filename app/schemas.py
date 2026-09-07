from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


DatabaseKind = Literal["postgresql", "mysql"]
ChartType = Literal["kpi", "bar", "line", "area", "pie", "scatter", "table"]
NumberFormat = Literal["number", "integer", "decimal", "percent", "currency", "compact"]


class DataSourceConnectRequest(BaseModel):
    kind: DatabaseKind
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(gt=0, lt=65536)
    database: str = Field(min_length=1, max_length=255)
    username: str = Field(min_length=1, max_length=255)
    password: SecretStr
    schema_name: str | None = Field(default=None, max_length=255)
    use_ssl: bool = True


class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None


class TableInfo(BaseModel):
    name: str
    columns: list[ColumnInfo]


class SchemaInfo(BaseModel):
    dialect: DatabaseKind
    database: str
    schema_name: str | None = None
    tables: list[TableInfo]


class DataSourceConnectResponse(BaseModel):
    connection_id: str
    status: Literal["connected"]
    database_schema: SchemaInfo = Field(alias="schema")
    suggested_questions: list[str]

    model_config = ConfigDict(populate_by_name=True)


class DataSourceSummary(BaseModel):
    connection_id: str
    kind: DatabaseKind
    database: str
    schema_name: str | None
    table_count: int


class WidgetPlan(BaseModel):
    title: str = Field(min_length=2, max_length=90)
    description: str = Field(default="", max_length=240)
    chart_type: ChartType
    sql: str = Field(min_length=8, max_length=8_000)
    x_field: str | None = None
    y_fields: list[str] = Field(default_factory=list, max_length=4)
    series_field: str | None = None
    value_field: str | None = None
    number_format: NumberFormat = "number"

    @model_validator(mode="after")
    def validate_visual_contract(self) -> "WidgetPlan":
        if self.chart_type == "kpi" and not self.value_field:
            self.value_field = "value"
        if self.chart_type not in {"kpi", "table"} and (not self.x_field or not self.y_fields):
            raise ValueError("Chart widgets require x_field and at least one y_field")
        return self


class DashboardPlan(BaseModel):
    title: str = Field(min_length=2, max_length=90)
    message: str = Field(min_length=2, max_length=500)
    widgets: list[WidgetPlan] = Field(min_length=1, max_length=8)


class Insight(BaseModel):
    message: str = Field(min_length=2, max_length=900)


class WidgetResult(BaseModel):
    id: str
    title: str
    description: str = ""
    chart_type: ChartType
    data: list[dict]
    sql: str | None = None
    x_field: str | None = None
    y_fields: list[str] = Field(default_factory=list)
    series_field: str | None = None
    value_field: str | None = None
    number_format: NumberFormat = "number"
    layout: dict[str, int]


class CopilotQueryRequest(BaseModel):
    connection_id: str
    prompt: str = Field(min_length=3, max_length=1_500)


class DashboardRequest(BaseModel):
    connection_id: str
    objective: str | None = Field(default=None, max_length=1_000)


class DashboardResponse(BaseModel):
    title: str
    message: str
    mode: Literal["openai", "heuristic"]
    widgets: list[WidgetResult]
    database_schema: SchemaInfo = Field(alias="schema")

    model_config = ConfigDict(populate_by_name=True)


class MetaResponse(BaseModel):
    app_name: str
    model: str
    copilot_mode: Literal["openai", "heuristic"]
