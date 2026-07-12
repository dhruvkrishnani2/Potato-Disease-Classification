from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Auth / OAuth
    google_client_id: str = ""
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    # AI advice
    gemini_api_key: str = ""

    # App
    frontend_url: str = "http://localhost:3000"
    port: int = 8000

    # Model path
    model_path: str = str(
        BASE_DIR / "saved_models" / "model_v1.keras"
    )

    cors_origins: str = "http://localhost:3000"

    def get_cors_origins(self):
        return [origin.strip() for origin in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings():
    return Settings()