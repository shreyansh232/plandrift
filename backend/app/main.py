from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from app.api.v1 import api_router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name, openapi_url=f"{settings.api_v1_str}/openapi.json"
)

# CORS Configuration
# ==================
# TESTING/DEVELOPMENT: allow_origins=["*"] permits requests from any origin.
# This is convenient for local development but INSECURE for production.
#
# PRODUCTION: Replace with specific origins:
#   allow_origins=[settings.frontend_url]
# or multiple origins:
#   allow_origins=["https://yourdomain.com", "https://app.yourdomain.com"]
#
# WARNING: Never use allow_origins=["*"] with allow_credentials=True in production
# as it allows any website to make authenticated requests to your API.
cors_origins: list[str] = []
if settings.debug:
    cors_origins = ["*"]
else:
    if settings.frontend_urls:
        cors_origins = [
            origin.strip()
            for origin in settings.frontend_urls.split(",")
            if origin.strip()
        ]
    elif settings.frontend_url:
        cors_origins = [settings.frontend_url]

    # Always allow local development clients to hit production backend.
    for local_origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
        if local_origin not in cors_origins:
            cors_origins.append(local_origin)

if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Session Middleware for OAuth state management
# Required by Authlib to store OAuth state during the authorization flow
# In production (debug=False): uses secure cookies (HTTPS only, strict same-site)
# In development (debug=True): relaxed settings for localhost testing
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key.get_secret_value(),
    same_site="lax",  # Prevents CSRF while allowing OAuth redirects
    https_only=not settings.debug,  # Require HTTPS in production
)

app.include_router(api_router, prefix=settings.api_v1_str)


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)
