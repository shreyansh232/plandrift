from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.engine.url import make_url
from app.config import get_settings

settings = get_settings()


connect_args: dict = {}
db_url = settings.database_url

# asyncpg does not accept sslmode as a query param; translate it to connect_args.
try:
    url_obj = make_url(db_url)
    if url_obj.drivername.startswith("postgresql+asyncpg"):
        query = dict(url_obj.query)
        sslmode = query.pop("sslmode", None)
        if sslmode and str(sslmode).lower() != "disable":
            connect_args["ssl"] = True
        url_obj = url_obj.set(query=query)
        db_url = url_obj
except Exception:
    # Fall back to raw URL if parsing fails.
    db_url = settings.database_url

engine = create_async_engine(db_url, echo=settings.debug, connect_args=connect_args)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass
