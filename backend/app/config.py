from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    """Application settings using Pydantic Settings."""

    # --- Core Application Settings ---
    app_name: str = "Planfirst API"  # Name used in OpenAPI docs
    api_v1_str: str = "/api"  # API prefix
    debug: bool = False  # If True, enables debug logs/SQL echo

    # --- Security & Auth ---
    secret_key: SecretStr = SecretStr("dev-secret-key-change-in-production")
    algorithm: str = "HS256"  # Hashing algo for tokens
    access_token_expire_minutes: int = 60  # Access tokens (60 minutes)
    refresh_token_expire_minutes: int = (
        60 * 24 * 30
    )  # Long-lived refresh tokens (30 days)
    refresh_token_secret_key: Optional[SecretStr] = (
        None  # Optional separate secret for refresh tokens
    )

    # Google auth
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/auth/google/callback"

    # --- Database ---
    # Default connection string. It will be overwritten by DATABASE_URL in .env
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5433/planfirst"
    )

    # --- External APIs ---
    openrouter_api_key: Optional[str] = (
        None  # API key for OpenRouter (Optional allows app to start without it)
    )
    openrouter_model: str = "google/gemini-3-flash-preview"
    openrouter_model_fast: Optional[str] = "google/gemini-3-flash-preview"

    # --- CORS ---
    # Comma-separated list of allowed origins. If unset, falls back to frontend_url.
    # Example: "https://yourapp.com,https://staging.yourapp.com"
    frontend_urls: Optional[str] = None
    frontend_url: str = "http://localhost:3000"  # Single allowed origin fallback

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
