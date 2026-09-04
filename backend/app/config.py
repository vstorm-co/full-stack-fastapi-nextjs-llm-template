from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongodb_uri: str
    mongodb_database: str = "growthengine"
    cors_origins: str = "http://localhost:3000"
    # Intake routing targets (frontend routes), overridable per environment.
    self_serve_url: str = "/portal/onboarding"
    growth_calendar_url: str = "/booking/growth"
    vip_calendar_url: str = "/booking/vip-fast-track"
    # Optional n8n webhook used by the OmniTrickle/EchoLink sync layer.
    n8n_webhook_url: str | None = None
    # Bearer secret that gates operator endpoints (lead feed, chaser dispatch,
    # telemetry). Empty means operator endpoints fail closed with 503.
    operator_admin_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
