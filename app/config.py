from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Data Augmented BI Copilot"
    frontend_origin: str = "http://localhost:3000"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"
    max_query_rows: int = 500
    query_timeout_seconds: int = 20

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
