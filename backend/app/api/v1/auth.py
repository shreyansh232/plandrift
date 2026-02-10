"""Auth API endpoints for user registration and login."""

import logging
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from authlib.integrations.starlette_client import OAuthError

from app.api.deps import get_db, get_current_user
from app.config import get_settings
from app.services.auth import (
    register_user,
    authenticate_user,
    refresh_access_token,
    logout_user,
)
from app.services.google_oauth import GoogleOAuthService
from app.schemas.user import (
    UserCreate,
    UserLogin,
    TokenResponse,
    UserResponse,
    RefreshTokenRequest,
    LogoutRequest,
)
from app.db.models import User
from app.oauth import oauth

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/auth", tags=["auth"])


def get_client_info(request: Request) -> tuple[Optional[str], Optional[str]]:
    """Extract device info and IP address from request."""
    device_info = request.headers.get("user-agent")
    ip_address = request.client.host if request.client else None
    return device_info, ip_address


@router.post("/register", response_model=TokenResponse)
async def register_new_user(
    user_in: UserCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Register a new user."""
    device_info, ip_address = get_client_info(request)
    return await register_user(db, user_in, device_info, ip_address)


@router.post("/login", response_model=TokenResponse)
async def login_user(
    user_in: UserLogin,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate user and return access token."""
    device_info, ip_address = get_client_info(request)
    return await authenticate_user(db, user_in, device_info, ip_address)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Refresh access token using a valid refresh token."""
    device_info, ip_address = get_client_info(request)
    return await refresh_access_token(
        db, refresh_data.refresh_token, device_info, ip_address
    )


@router.post("/logout")
async def logout(
    logout_data: LogoutRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Logout user by revoking refresh token(s).

    If refresh_token is provided, only that token is revoked.
    If logout_all is True, all refresh tokens for the user are revoked.
    """
    logout_all = logout_data.refresh_token is None
    return await logout_user(
        db, str(current_user.id), logout_data.refresh_token, logout_all
    )


@router.get("/profile", response_model=UserResponse)
async def read_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Get current user profile."""
    return UserResponse.model_validate(current_user)


# Google OAuth endpoints
@router.get("/google/login")
async def google_login(request: Request) -> dict:
    """
    Initiate Google OAuth login flow.

    Returns the authorization URL to redirect the user to Google's consent page.
    The frontend should redirect the user to this URL.

    Returns:
        dict: Contains 'authorization_url' and 'state' parameters
    """
    # Generate redirect URI for callback (prefer explicit setting)
    redirect_uri = settings.GOOGLE_REDIRECT_URI or request.url_for(
        "google_callback"
    )

    # Redirect to Google's authorization page
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback")
async def google_callback(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    """
    Handle Google OAuth callback.

    This endpoint is called by Google after the user authorizes the application.
    It exchanges the authorization code for tokens and creates/updates the user.
    On success, redirects to frontend with tokens in URL fragment.
    On failure, redirects to frontend with error parameter.

    Returns:
        RedirectResponse: Redirect to frontend with tokens or error
    """
    frontend_base = settings.frontend_url
    if settings.frontend_urls:
        primary = settings.frontend_urls.split(",")[0].strip()
        if primary:
            frontend_base = primary
    frontend_callback_url = f"{frontend_base}/auth/callback"

    try:
        # Exchange authorization code for tokens and get user info
        token = await oauth.google.authorize_access_token(request)

        # Extract user info from token (Authlib automatically parses id_token)
        user_info = token.get("userinfo")

        if not user_info:
            raise ValueError("No user info returned from Google")

        # Authenticate or create user
        device_info, ip_address = get_client_info(request)
        token_response = await GoogleOAuthService.authenticate_or_create_user(
            db, user_info, device_info, ip_address
        )

        # Redirect to frontend with tokens in URL fragment (more secure than query params)
        # Frontend will extract tokens from fragment and store them
        params = urlencode({
            "access_token": token_response.access_token,
            "refresh_token": token_response.refresh_token,
        })
        return RedirectResponse(url=f"{frontend_callback_url}#{params}")

    except OAuthError as error:
        # Log the actual error for debugging
        logger.error(f"OAuth error during Google callback: {error.error}")
        # Redirect to frontend with generic error
        params = urlencode({"error": "oauth_error", "message": "Authentication was cancelled or failed"})
        return RedirectResponse(url=f"{frontend_callback_url}?{params}")

    except ValueError as error:
        # Known validation errors (email not verified, account exists, etc.)
        logger.warning(f"Validation error during Google OAuth: {str(error)}")
        # For specific user-actionable errors, we can pass a hint
        error_message = str(error)
        if "already exists" in error_message:
            params = urlencode({"error": "account_exists", "message": "An account with this email already exists. Please sign in with your password."})
        elif "not verified" in error_message:
            params = urlencode({"error": "email_not_verified", "message": "Please verify your email with Google first."})
        else:
            params = urlencode({"error": "validation_error", "message": "Authentication failed. Please try again."})
        return RedirectResponse(url=f"{frontend_callback_url}?{params}")

    except Exception as error:
        # Log the actual error for debugging, but don't expose to user
        logger.exception(f"Unexpected error during Google OAuth callback: {str(error)}")
        # Redirect with generic error message
        params = urlencode({"error": "server_error", "message": "Authentication failed. Please try again."})
        return RedirectResponse(url=f"{frontend_callback_url}?{params}")
