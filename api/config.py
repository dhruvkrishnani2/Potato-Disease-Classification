from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
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


@lru_cache
def get_settings():
    return Settings()
