from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    """Application settings using Pydantic Settings."""

    # --- Core Application Settings ---
    app_name: str = "Plandrift API"  # Name used in OpenAPI docs
    api_v1_str: str = "/api"  # API prefix
    debug: bool = False  # If True, enables debug logs/SQL echo

    # --- Security & Auth ---
    secret_key: SecretStr = SecretStr("dev-secret-key-change-in-production")
    algorithm: str = "HS256"  # Hashing algo for tokens
    access_token_expire_minutes: int = 15  # Short-lived access tokens (15 minutes)
    refresh_token_expire_minutes: int = (
        60 * 24 * 30
    )  # Long-lived refresh tokens (30 days)
    refresh_token_secret_key: Optional[SecretStr] = (
        None  # Optional separate secret for refresh tokens
    )

    # Google auth
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: str = (
        "https://plandrift.onrender.com/api/auth/google/callback"
    )

    # --- Database ---
    # Default connection string. It will be overwritten by DATABASE_URL in .env
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5433/plandrift"
    )

    # --- External APIs ---
    openrouter_api_key: Optional[str] = (
        None  # API key for OpenRouter (Optional allows app to start without it)
    )

    # --- CORS ---
    frontend_url: str = "http://localhost:3000"  # Allowed origin for browser requests

    # --- Configuration ---
    # Pydantic V2 configuration class
    model_config = SettingsConfigDict(
        env_file=".env",  # Look for a file named .env
        env_file_encoding="utf-8",  # Read it as UTF-8
        extra="ignore",  # Don't crash if .env has extra keys we don't know about
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of Settings.
    The lru_cache decorator ensures that we instantiate the Settings class
    (and read the .env file) only once. Subsequent calls return the same object,
    which improves performance.
    """
    return Settings()
