import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .copilot import build_dashboard
from .datasources import registry, suggested_questions
from .schemas import (
    CopilotQueryRequest,
    DashboardRequest,
    DashboardResponse,
    DataSourceConnectRequest,
    DataSourceConnectResponse,
    DataSourceSummary,
    MetaResponse,
)


logging.basicConfig(level=logging.INFO)
settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    return MetaResponse(
        app_name=settings.app_name,
        model=settings.openai_model,
        copilot_mode="openai" if settings.openai_api_key else "heuristic",
    )


@app.get("/api/datasources", response_model=list[DataSourceSummary])
def list_datasources() -> list[DataSourceSummary]:
    return registry.list()


@app.post("/api/datasources/connect", response_model=DataSourceConnectResponse)
def connect_datasource(request: DataSourceConnectRequest) -> DataSourceConnectResponse:
    try:
        source = registry.add(request)
    except Exception as exc:
        logging.exception("Data source connection failed")
        raise HTTPException(status_code=422, detail=f"Could not connect to data source: {exc}") from exc
    return DataSourceConnectResponse(
        connection_id=source.id,
        status="connected",
        schema=source.schema,
        suggested_questions=suggested_questions(source.schema),
    )


@app.post("/api/dashboard/generate", response_model=DashboardResponse)
def generate_dashboard(request: DashboardRequest) -> DashboardResponse:
    try:
        source = registry.get(request.connection_id)
        return build_dashboard(source, request.objective)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("Dashboard generation failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/copilot/query", response_model=DashboardResponse)
def copilot_query(request: CopilotQueryRequest) -> DashboardResponse:
    try:
        source = registry.get(request.connection_id)
        return build_dashboard(source, request.prompt)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("Copilot query failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
