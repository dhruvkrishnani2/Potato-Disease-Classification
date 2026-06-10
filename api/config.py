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
    cors_origins: str = ""
    model_path: str = "../saved_models/model_v1.keras"
    port: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"

    def get_cors_origins(self) -> list[str]:
        if self.cors_origins.strip():
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        origins = {self.frontend_url, "http://localhost:3000", "http://localhost"}
        return [o for o in origins if o]


@lru_cache
def get_settings() -> Settings:
    return Settings()
