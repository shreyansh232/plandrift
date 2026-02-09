"""OAuth configuration for Google authentication.

This module sets up the Authlib OAuth registry for Google OAuth 2.0/OpenID Connect.
"""

from authlib.integrations.starlette_client import OAuth

from app.config import get_settings

# Load settings
settings = get_settings()

# Create OAuth registry
oauth = OAuth()

# Register Google OAuth client
# Uses OpenID Connect discovery URL for automatic configuration
oauth.register(
    name="google",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile",
    },
)
