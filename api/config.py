import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_client_id: str = ""
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24
    gemini_api_key: str = ""
    frontend_url: str = "http://localhost:3000"
    model_path: str = "../saved_models/model_v1.keras"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
