"""Configuration models for project generation."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator


class DatabaseType(str, Enum):
    """Supported database types."""

    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    SQLITE = "sqlite"
    NONE = "none"


class AuthType(str, Enum):
    """Supported authentication types."""

    JWT = "jwt"
    API_KEY = "api_key"
    BOTH = "both"
    NONE = "none"


class BackgroundTaskType(str, Enum):
    """Supported background task systems."""

    NONE = "none"
    CELERY = "celery"
    TASKIQ = "taskiq"
    ARQ = "arq"


class CIType(str, Enum):
    """Supported CI/CD systems."""

    GITHUB = "github"
    GITLAB = "gitlab"
    NONE = "none"


class FrontendType(str, Enum):
    """Supported frontend frameworks."""

    NONE = "none"
    NEXTJS = "nextjs"


class WebSocketAuthType(str, Enum):
    """WebSocket authentication types for AI Agent."""

    NONE = "none"
    JWT = "jwt"
    API_KEY = "api_key"


class AdminEnvironmentType(str, Enum):
    """Admin panel environment restriction types."""

    ALL = "all"  # Available in all environments
    DEV_ONLY = "dev_only"  # Only in development
    DEV_STAGING = "dev_staging"  # Development + Staging (recommended)
    DISABLED = "disabled"  # Disabled everywhere


class OAuthProvider(str, Enum):
    """Supported OAuth2 providers."""

    NONE = "none"
    GOOGLE = "google"


class AIFrameworkType(str, Enum):
    """Supported AI agent frameworks."""

    PYDANTIC_AI = "pydantic_ai"
    LANGCHAIN = "langchain"


class LogfireFeatures(BaseModel):
    """Logfire instrumentation features."""

    fastapi: bool = True
    database: bool = True
    redis: bool = False
    celery: bool = False
    httpx: bool = False


class ProjectConfig(BaseModel):
    """Full project configuration."""

    # Basic info
    project_name: str = Field(..., min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    project_description: str = "A FastAPI project"

    author_name: str = "Your Name"
    author_email: str = "your@email.com"

    # Database
    database: DatabaseType = DatabaseType.POSTGRESQL
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30

    # Authentication
    auth: AuthType = AuthType.JWT
    oauth_provider: OAuthProvider = OAuthProvider.NONE
    enable_session_management: bool = False

    # Observability
    enable_logfire: bool = True
    logfire_features: LogfireFeatures = Field(default_factory=LogfireFeatures)

    # Background tasks
    background_tasks: BackgroundTaskType = BackgroundTaskType.NONE

    # Optional integrations
    enable_redis: bool = False
    enable_caching: bool = False
    enable_rate_limiting: bool = False
    rate_limit_requests: int = 100
    rate_limit_period: int = 60
    enable_pagination: bool = True
    enable_sentry: bool = False
    enable_prometheus: bool = False
    enable_admin_panel: bool = False
    admin_environments: AdminEnvironmentType = AdminEnvironmentType.DEV_STAGING
    admin_require_auth: bool = True
    enable_websockets: bool = False
    enable_file_storage: bool = False
    enable_ai_agent: bool = False
    ai_framework: AIFrameworkType = AIFrameworkType.PYDANTIC_AI
    enable_conversation_persistence: bool = False
    enable_webhooks: bool = False
    websocket_auth: WebSocketAuthType = WebSocketAuthType.NONE
    enable_cors: bool = True
    enable_orjson: bool = True

    # Frontend features
    enable_i18n: bool = False

    # Example CRUD
    include_example_crud: bool = True

    # Dev tools
    enable_pytest: bool = True
    enable_precommit: bool = True
    enable_makefile: bool = True
    enable_docker: bool = True
    ci_type: CIType = CIType.GITHUB
    enable_kubernetes: bool = False
    generate_env: bool = True

    # Python version
    python_version: str = "3.12"

    # Frontend
    frontend: FrontendType = FrontendType.NONE
    frontend_port: int = 3000

    # Backend
    backend_port: int = 8000

    @computed_field
    @property
    def project_slug(self) -> str:
        """Return project slug (underscores instead of hyphens)."""
        return self.project_name.replace("-", "_")

    @model_validator(mode="after")
    def validate_option_combinations(self) -> "ProjectConfig":
        """Validate that option combinations are valid.

        Raises ValueError for invalid combinations:
        - Admin panel requires a database (PostgreSQL or SQLite)
        - Admin panel (SQLAdmin) does not support MongoDB
        - Caching requires Redis to be enabled
        - Session management requires a database
        - Conversation persistence requires a database
        """
        if self.enable_admin_panel and self.database == DatabaseType.NONE:
            raise ValueError("Admin panel requires a database")
        if self.enable_admin_panel and self.database == DatabaseType.MONGODB:
            raise ValueError("Admin panel (SQLAdmin) requires PostgreSQL or SQLite")
        if self.enable_caching and not self.enable_redis:
            raise ValueError("Caching requires Redis to be enabled")
        if self.enable_session_management and self.database == DatabaseType.NONE:
            raise ValueError("Session management requires a database")
        if self.enable_conversation_persistence and self.database == DatabaseType.NONE:
            raise ValueError("Conversation persistence requires a database")
        return self

    def to_cookiecutter_context(self) -> dict[str, Any]:
        """Convert config to cookiecutter context."""
        return {
            "project_name": self.project_name,
            "project_slug": self.project_slug,
            "project_description": self.project_description,
            "author_name": self.author_name,
            "author_email": self.author_email,
            # Database
            "database": self.database.value,
            "use_postgresql": self.database == DatabaseType.POSTGRESQL,
            "use_mongodb": self.database == DatabaseType.MONGODB,
            "use_sqlite": self.database == DatabaseType.SQLITE,
            "use_database": self.database != DatabaseType.NONE,
            "db_pool_size": self.db_pool_size,
            "db_max_overflow": self.db_max_overflow,
            "db_pool_timeout": self.db_pool_timeout,
            # Auth
            "auth": self.auth.value,
            "use_jwt": self.auth in (AuthType.JWT, AuthType.BOTH),
            "use_api_key": self.auth in (AuthType.API_KEY, AuthType.BOTH),
            "use_auth": self.auth != AuthType.NONE,
            # OAuth
            "oauth_provider": self.oauth_provider.value,
            "enable_oauth": self.oauth_provider != OAuthProvider.NONE,
            "enable_oauth_google": self.oauth_provider == OAuthProvider.GOOGLE,
            # Session Management
            "enable_session_management": self.enable_session_management,
            # Logfire
            "enable_logfire": self.enable_logfire,
            "logfire_fastapi": self.logfire_features.fastapi,
            "logfire_database": self.logfire_features.database,
            "logfire_redis": self.logfire_features.redis,
            "logfire_celery": self.logfire_features.celery,
            "logfire_httpx": self.logfire_features.httpx,
            # Background tasks
            "background_tasks": self.background_tasks.value,
            "use_celery": self.background_tasks == BackgroundTaskType.CELERY,
            "use_taskiq": self.background_tasks == BackgroundTaskType.TASKIQ,
            "use_arq": self.background_tasks == BackgroundTaskType.ARQ,
            # Integrations
            "enable_redis": self.enable_redis,
            "enable_caching": self.enable_caching,
            "enable_rate_limiting": self.enable_rate_limiting,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_period": self.rate_limit_period,
            "enable_pagination": self.enable_pagination,
            "enable_sentry": self.enable_sentry,
            "enable_prometheus": self.enable_prometheus,
            "enable_admin_panel": self.enable_admin_panel,
            "admin_environments": self.admin_environments.value,
            "admin_env_all": self.admin_environments == AdminEnvironmentType.ALL,
            "admin_env_dev_only": self.admin_environments == AdminEnvironmentType.DEV_ONLY,
            "admin_env_dev_staging": self.admin_environments == AdminEnvironmentType.DEV_STAGING,
            "admin_env_disabled": self.admin_environments == AdminEnvironmentType.DISABLED,
            "admin_require_auth": self.admin_require_auth,
            "enable_websockets": self.enable_websockets,
            "enable_file_storage": self.enable_file_storage,
            "enable_ai_agent": self.enable_ai_agent,
            "ai_framework": self.ai_framework.value,
            "use_pydantic_ai": self.ai_framework == AIFrameworkType.PYDANTIC_AI,
            "use_langchain": self.ai_framework == AIFrameworkType.LANGCHAIN,
            "enable_conversation_persistence": self.enable_conversation_persistence,
            "enable_webhooks": self.enable_webhooks,
            "websocket_auth": self.websocket_auth.value,
            "websocket_auth_jwt": self.websocket_auth == WebSocketAuthType.JWT,
            "websocket_auth_api_key": self.websocket_auth == WebSocketAuthType.API_KEY,
            "websocket_auth_none": self.websocket_auth == WebSocketAuthType.NONE,
            "enable_cors": self.enable_cors,
            "enable_orjson": self.enable_orjson,
            # Frontend features
            "enable_i18n": self.enable_i18n,
            # Example CRUD
            "include_example_crud": self.include_example_crud,
            # Dev tools
            "enable_pytest": self.enable_pytest,
            "enable_precommit": self.enable_precommit,
            "enable_makefile": self.enable_makefile,
            "enable_docker": self.enable_docker,
            "ci_type": self.ci_type.value,
            "use_github_actions": self.ci_type == CIType.GITHUB,
            "use_gitlab_ci": self.ci_type == CIType.GITLAB,
            "enable_kubernetes": self.enable_kubernetes,
            "generate_env": self.generate_env,
            # Python version
            "python_version": self.python_version,
            # Frontend
            "frontend": self.frontend.value,
            "use_frontend": self.frontend != FrontendType.NONE,
            "use_nextjs": self.frontend == FrontendType.NEXTJS,
            "frontend_port": self.frontend_port,
            # Backend
            "backend_port": self.backend_port,
        }
