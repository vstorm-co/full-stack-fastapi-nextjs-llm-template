"""Tests for fastapi_gen.config module."""

import pytest
from pydantic import ValidationError

from fastapi_gen.config import (
    AuthType,
    BackgroundTaskType,
    CIType,
    DatabaseType,
    LogfireFeatures,
    ProjectConfig,
)


class TestEnums:
    """Tests for configuration enums."""

    def test_database_type_values(self) -> None:
        """Test DatabaseType enum values."""
        assert DatabaseType.POSTGRESQL.value == "postgresql"
        assert DatabaseType.MONGODB.value == "mongodb"
        assert DatabaseType.SQLITE.value == "sqlite"
        assert DatabaseType.NONE.value == "none"

    def test_auth_type_values(self) -> None:
        """Test AuthType enum values."""
        assert AuthType.JWT.value == "jwt"
        assert AuthType.API_KEY.value == "api_key"
        assert AuthType.BOTH.value == "both"
        assert AuthType.NONE.value == "none"

    def test_background_task_type_values(self) -> None:
        """Test BackgroundTaskType enum values."""
        assert BackgroundTaskType.NONE.value == "none"
        assert BackgroundTaskType.CELERY.value == "celery"
        assert BackgroundTaskType.TASKIQ.value == "taskiq"
        assert BackgroundTaskType.ARQ.value == "arq"

    def test_ci_type_values(self) -> None:
        """Test CIType enum values."""
        assert CIType.GITHUB.value == "github"
        assert CIType.GITLAB.value == "gitlab"
        assert CIType.NONE.value == "none"


class TestLogfireFeatures:
    """Tests for LogfireFeatures model."""

    def test_default_values(self) -> None:
        """Test default LogfireFeatures values."""
        features = LogfireFeatures()
        assert features.fastapi is True
        assert features.database is True
        assert features.redis is False
        assert features.celery is False
        assert features.httpx is False

    def test_custom_values(self) -> None:
        """Test LogfireFeatures with custom values."""
        features = LogfireFeatures(
            fastapi=False,
            database=False,
            redis=True,
            celery=True,
            httpx=True,
        )
        assert features.fastapi is False
        assert features.database is False
        assert features.redis is True
        assert features.celery is True
        assert features.httpx is True


class TestProjectConfig:
    """Tests for ProjectConfig model."""

    def test_minimal_config(self) -> None:
        """Test minimal valid configuration."""
        config = ProjectConfig(project_name="myproject")
        assert config.project_name == "myproject"
        assert config.database == DatabaseType.POSTGRESQL
        assert config.auth == AuthType.JWT

    def test_valid_project_names(self) -> None:
        """Test valid project name patterns."""
        valid_names = [
            "myproject",
            "my_project",
            "project123",
            "a",
            "abc_def_123",
        ]
        for name in valid_names:
            config = ProjectConfig(project_name=name)
            assert config.project_name == name

    def test_invalid_project_names(self) -> None:
        """Test invalid project name patterns."""
        invalid_names = [
            "123project",  # starts with number
            "my-project",  # contains hyphen
            "My_Project",  # contains uppercase
            "_project",  # starts with underscore
            "",  # empty
        ]
        for name in invalid_names:
            with pytest.raises(ValidationError):
                ProjectConfig(project_name=name)

    def test_project_slug_conversion(self) -> None:
        """Test project_slug is derived from project_name."""
        config = ProjectConfig(project_name="my_project")
        context = config.to_cookiecutter_context()
        assert context["project_slug"] == "my_project"

    def test_all_fields_present_in_context(self) -> None:
        """Test all expected fields are in cookiecutter context."""
        config = ProjectConfig(project_name="test")
        context = config.to_cookiecutter_context()

        expected_keys = [
            "project_name",
            "project_slug",
            "project_description",
            "author_name",
            "author_email",
            "database",
            "use_postgresql",
            "use_mongodb",
            "use_sqlite",
            "use_database",
            "auth",
            "use_jwt",
            "use_api_key",
            "use_auth",
            "enable_logfire",
            "logfire_fastapi",
            "logfire_database",
            "logfire_redis",
            "logfire_celery",
            "logfire_httpx",
            "background_tasks",
            "use_celery",
            "use_taskiq",
            "use_arq",
            "enable_redis",
            "enable_caching",
            "enable_rate_limiting",
            "enable_pagination",
            "enable_sentry",
            "enable_prometheus",
            "enable_admin_panel",
            "enable_websockets",
            "enable_file_storage",
            "enable_ai_agent",
            "enable_cors",
            "enable_orjson",
            "enable_pytest",
            "enable_precommit",
            "enable_makefile",
            "enable_docker",
            "ci_type",
            "use_github_actions",
            "use_gitlab_ci",
            "enable_kubernetes",
        ]

        for key in expected_keys:
            assert key in context, f"Missing key: {key}"


class TestCookiecutterContext:
    """Tests for to_cookiecutter_context conversion."""

    def test_postgresql_database_flags(self) -> None:
        """Test PostgreSQL sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            database=DatabaseType.POSTGRESQL,
        )
        context = config.to_cookiecutter_context()

        assert context["database"] == "postgresql"
        assert context["use_postgresql"] is True
        assert context["use_mongodb"] is False
        assert context["use_sqlite"] is False
        assert context["use_database"] is True

    def test_mongodb_database_flags(self) -> None:
        """Test MongoDB sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            database=DatabaseType.MONGODB,
        )
        context = config.to_cookiecutter_context()

        assert context["database"] == "mongodb"
        assert context["use_postgresql"] is False
        assert context["use_mongodb"] is True
        assert context["use_sqlite"] is False
        assert context["use_database"] is True

    def test_sqlite_database_flags(self) -> None:
        """Test SQLite sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            database=DatabaseType.SQLITE,
        )
        context = config.to_cookiecutter_context()

        assert context["database"] == "sqlite"
        assert context["use_postgresql"] is False
        assert context["use_mongodb"] is False
        assert context["use_sqlite"] is True
        assert context["use_database"] is True

    def test_no_database_flags(self) -> None:
        """Test no database sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            database=DatabaseType.NONE,
        )
        context = config.to_cookiecutter_context()

        assert context["database"] == "none"
        assert context["use_postgresql"] is False
        assert context["use_mongodb"] is False
        assert context["use_sqlite"] is False
        assert context["use_database"] is False

    def test_jwt_auth_flags(self) -> None:
        """Test JWT auth sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            auth=AuthType.JWT,
        )
        context = config.to_cookiecutter_context()

        assert context["auth"] == "jwt"
        assert context["use_jwt"] is True
        assert context["use_api_key"] is False
        assert context["use_auth"] is True

    def test_api_key_auth_flags(self) -> None:
        """Test API key auth sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            auth=AuthType.API_KEY,
        )
        context = config.to_cookiecutter_context()

        assert context["auth"] == "api_key"
        assert context["use_jwt"] is False
        assert context["use_api_key"] is True
        assert context["use_auth"] is True

    def test_both_auth_flags(self) -> None:
        """Test both auth sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            auth=AuthType.BOTH,
        )
        context = config.to_cookiecutter_context()

        assert context["auth"] == "both"
        assert context["use_jwt"] is True
        assert context["use_api_key"] is True
        assert context["use_auth"] is True

    def test_no_auth_flags(self) -> None:
        """Test no auth sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            auth=AuthType.NONE,
        )
        context = config.to_cookiecutter_context()

        assert context["auth"] == "none"
        assert context["use_jwt"] is False
        assert context["use_api_key"] is False
        assert context["use_auth"] is False

    def test_celery_background_task_flags(self) -> None:
        """Test Celery sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            background_tasks=BackgroundTaskType.CELERY,
        )
        context = config.to_cookiecutter_context()

        assert context["background_tasks"] == "celery"
        assert context["use_celery"] is True
        assert context["use_taskiq"] is False
        assert context["use_arq"] is False

    def test_taskiq_background_task_flags(self) -> None:
        """Test Taskiq sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            background_tasks=BackgroundTaskType.TASKIQ,
        )
        context = config.to_cookiecutter_context()

        assert context["background_tasks"] == "taskiq"
        assert context["use_celery"] is False
        assert context["use_taskiq"] is True
        assert context["use_arq"] is False

    def test_arq_background_task_flags(self) -> None:
        """Test ARQ sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            background_tasks=BackgroundTaskType.ARQ,
        )
        context = config.to_cookiecutter_context()

        assert context["background_tasks"] == "arq"
        assert context["use_celery"] is False
        assert context["use_taskiq"] is False
        assert context["use_arq"] is True

    def test_github_ci_flags(self) -> None:
        """Test GitHub CI sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            ci_type=CIType.GITHUB,
        )
        context = config.to_cookiecutter_context()

        assert context["ci_type"] == "github"
        assert context["use_github_actions"] is True
        assert context["use_gitlab_ci"] is False

    def test_gitlab_ci_flags(self) -> None:
        """Test GitLab CI sets correct flags."""
        config = ProjectConfig(
            project_name="test",
            ci_type=CIType.GITLAB,
        )
        context = config.to_cookiecutter_context()

        assert context["ci_type"] == "gitlab"
        assert context["use_github_actions"] is False
        assert context["use_gitlab_ci"] is True

    def test_logfire_features_in_context(self) -> None:
        """Test Logfire features are correctly mapped."""
        config = ProjectConfig(
            project_name="test",
            enable_logfire=True,
            logfire_features=LogfireFeatures(
                fastapi=True,
                database=False,
                redis=True,
                celery=False,
                httpx=True,
            ),
        )
        context = config.to_cookiecutter_context()

        assert context["enable_logfire"] is True
        assert context["logfire_fastapi"] is True
        assert context["logfire_database"] is False
        assert context["logfire_redis"] is True
        assert context["logfire_celery"] is False
        assert context["logfire_httpx"] is True


class TestOptionCombinationValidation:
    """Tests for invalid option combination validation."""

    def test_admin_panel_requires_database(self) -> None:
        """Test that admin panel cannot be enabled without a database."""
        with pytest.raises(ValidationError, match="Admin panel requires a database"):
            ProjectConfig(
                project_name="test",
                database=DatabaseType.NONE,
                enable_admin_panel=True,
            )

    def test_admin_panel_not_supported_with_mongodb(self) -> None:
        """Test that admin panel (SQLAdmin) does not support MongoDB."""
        with pytest.raises(
            ValidationError, match="Admin panel \\(SQLAdmin\\) requires PostgreSQL or SQLite"
        ):
            ProjectConfig(
                project_name="test",
                database=DatabaseType.MONGODB,
                enable_admin_panel=True,
            )

    def test_caching_requires_redis(self) -> None:
        """Test that caching requires Redis to be enabled."""
        with pytest.raises(ValidationError, match="Caching requires Redis to be enabled"):
            ProjectConfig(
                project_name="test",
                enable_caching=True,
                enable_redis=False,
            )

    def test_session_management_requires_database(self) -> None:
        """Test that session management requires a database."""
        with pytest.raises(ValidationError, match="Session management requires a database"):
            ProjectConfig(
                project_name="test",
                database=DatabaseType.NONE,
                enable_session_management=True,
            )

    def test_conversation_persistence_requires_database(self) -> None:
        """Test that conversation persistence requires a database."""
        with pytest.raises(ValidationError, match="Conversation persistence requires a database"):
            ProjectConfig(
                project_name="test",
                database=DatabaseType.NONE,
                enable_conversation_persistence=True,
            )

    def test_admin_panel_with_postgresql_is_valid(self) -> None:
        """Test that admin panel with PostgreSQL is valid."""
        config = ProjectConfig(
            project_name="test",
            database=DatabaseType.POSTGRESQL,
            enable_admin_panel=True,
        )
        assert config.enable_admin_panel is True
        assert config.database == DatabaseType.POSTGRESQL

    def test_admin_panel_with_sqlite_is_valid(self) -> None:
        """Test that admin panel with SQLite is valid."""
        config = ProjectConfig(
            project_name="test",
            database=DatabaseType.SQLITE,
            enable_admin_panel=True,
        )
        assert config.enable_admin_panel is True
        assert config.database == DatabaseType.SQLITE

    def test_caching_with_redis_is_valid(self) -> None:
        """Test that caching with Redis enabled is valid."""
        config = ProjectConfig(
            project_name="test",
            enable_caching=True,
            enable_redis=True,
        )
        assert config.enable_caching is True
        assert config.enable_redis is True

    def test_session_management_with_database_is_valid(self) -> None:
        """Test that session management with a database is valid."""
        config = ProjectConfig(
            project_name="test",
            database=DatabaseType.POSTGRESQL,
            enable_session_management=True,
        )
        assert config.enable_session_management is True

    def test_conversation_persistence_with_database_is_valid(self) -> None:
        """Test that conversation persistence with a database is valid."""
        config = ProjectConfig(
            project_name="test",
            database=DatabaseType.POSTGRESQL,
            enable_conversation_persistence=True,
        )
        assert config.enable_conversation_persistence is True
