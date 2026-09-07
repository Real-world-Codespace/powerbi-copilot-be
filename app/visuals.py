from uuid import uuid4

from .schemas import WidgetPlan, WidgetResult


def layout_for(index: int, chart_type: str) -> dict[str, int]:
    if chart_type == "kpi":
        return {"col_span": 3, "row_span": 1}
    if chart_type == "table":
        return {"col_span": 6, "row_span": 2}
    return {"col_span": 6, "row_span": 2}


def materialize_widget(plan: WidgetPlan, data: list[dict], sql: str, index: int) -> WidgetResult:
    value_field = plan.value_field
    if plan.chart_type == "kpi" and data and value_field not in data[0]:
        value_field = next(iter(data[0]))
    return WidgetResult(
        id=str(uuid4()),
        title=plan.title,
        description=plan.description,
        chart_type=plan.chart_type,
        data=data,
        sql=sql,
        x_field=plan.x_field,
        y_fields=plan.y_fields,
        series_field=plan.series_field,
        value_field=value_field,
        number_format=plan.number_format,
        layout=layout_for(index, plan.chart_type),
    )
