from contextlib import asynccontextmanager

from app.config import get_settings
from app.db import MongoStore
from app.routes import router
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.store = MongoStore(settings)
    await app.state.store.connect()
    yield
    await app.state.store.close()


app = FastAPI(title="GrowthEngine OS API", version="0.1.0", lifespan=lifespan)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Basic hardening headers; CSP still requires a policy review before release."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response
