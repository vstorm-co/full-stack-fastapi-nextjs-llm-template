"""Integration tests for generated template code quality.

These tests generate actual projects and run linting/type checking on them.
They are slower but ensure the template produces valid, well-formatted code.
"""

import subprocess
from pathlib import Path

import pytest

from fastapi_gen.config import (
    AuthType,
    BackgroundTaskType,
    CIType,
    DatabaseType,
    FrontendType,
    LogfireFeatures,
    OAuthProvider,
    ProjectConfig,
)
from fastapi_gen.generator import generate_project


@pytest.fixture
def generated_project_minimal(tmp_path: Path) -> Path:
    """Generate a minimal project for testing."""
    config = ProjectConfig(
        project_name="test_minimal",
        database=DatabaseType.NONE,
        auth=AuthType.NONE,
        enable_logfire=False,
        enable_docker=False,
        enable_ai_agent=False,
        ci_type=CIType.NONE,
    )
    return generate_project(config, tmp_path)


@pytest.fixture
def generated_project_full(tmp_path: Path) -> Path:
    """Generate a full-featured project for testing."""
    config = ProjectConfig(
        project_name="test_full",
        project_description="A fully featured test project",
        author_name="Test Author",
        author_email="test@example.com",
        database=DatabaseType.POSTGRESQL,
        auth=AuthType.JWT,
        oauth_provider=OAuthProvider.GOOGLE,
        enable_session_management=True,
        enable_logfire=True,
        logfire_features=LogfireFeatures(
            fastapi=True,
            database=True,
            redis=True,
            celery=True,
            httpx=True,
        ),
        background_tasks=BackgroundTaskType.CELERY,
        enable_redis=True,
        enable_caching=True,
        enable_rate_limiting=True,
        enable_pagination=True,
        enable_sentry=True,
        enable_prometheus=True,
        enable_admin_panel=True,
        enable_websockets=True,
        enable_file_storage=True,
        enable_ai_agent=True,
        enable_webhooks=True,
        enable_cors=True,
        enable_orjson=True,
        enable_pytest=True,
        enable_precommit=True,
        enable_makefile=True,
        enable_docker=True,
        ci_type=CIType.GITHUB,
        enable_kubernetes=True,
        include_example_crud=True,
        frontend=FrontendType.NONE,
    )
    return generate_project(config, tmp_path)


class TestGeneratedTemplateRuff:
    """Test that generated code passes ruff linting."""

    @pytest.mark.slow
    def test_minimal_project_passes_ruff(self, generated_project_minimal: Path) -> None:
        """Test minimal project passes ruff check."""
        backend_path = generated_project_minimal / "backend"
        result = subprocess.run(
            ["uv", "run", "ruff", "check", str(backend_path)],
            capture_output=True,
            text=True,
            cwd=generated_project_minimal,
        )
        assert result.returncode == 0, f"Ruff failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.slow
    def test_full_project_passes_ruff(self, generated_project_full: Path) -> None:
        """Test full project passes ruff check."""
        backend_path = generated_project_full / "backend"
        result = subprocess.run(
            ["uv", "run", "ruff", "check", str(backend_path)],
            capture_output=True,
            text=True,
            cwd=generated_project_full,
        )
        assert result.returncode == 0, f"Ruff failed:\n{result.stdout}\n{result.stderr}"


class TestGeneratedTemplateMypy:
    """Test that generated code passes mypy type checking."""

    @pytest.mark.slow
    def test_minimal_project_passes_mypy(self, generated_project_minimal: Path) -> None:
        """Test minimal project passes mypy check."""
        backend_path = generated_project_minimal / "backend"
        app_path = backend_path / "app"
        result = subprocess.run(
            ["uv", "run", "mypy", str(app_path), "--ignore-missing-imports"],
            capture_output=True,
            text=True,
            cwd=generated_project_minimal,
        )
        assert result.returncode == 0, f"Mypy failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.slow
    def test_full_project_passes_mypy(self, generated_project_full: Path) -> None:
        """Test full project passes mypy check."""
        backend_path = generated_project_full / "backend"
        app_path = backend_path / "app"
        result = subprocess.run(
            ["uv", "run", "mypy", str(app_path), "--ignore-missing-imports"],
            capture_output=True,
            text=True,
            cwd=generated_project_full,
        )
        assert result.returncode == 0, f"Mypy failed:\n{result.stdout}\n{result.stderr}"


class TestGeneratedTemplateAgentsFolder:
    """Test that agents folder is conditionally created based on enable_ai_agent."""

    @pytest.mark.slow
    def test_agents_folder_not_created_when_disabled(self, generated_project_minimal: Path) -> None:
        """Test that agents folder is not present when AI agent is disabled."""
        agents_path = generated_project_minimal / "backend" / "app" / "agents"
        assert not agents_path.exists(), "agents/ folder should not exist when AI is disabled"

    @pytest.mark.slow
    def test_agents_folder_created_when_enabled(self, generated_project_full: Path) -> None:
        """Test that agents folder exists when AI agent is enabled."""
        agents_path = generated_project_full / "backend" / "app" / "agents"
        assert agents_path.exists(), "agents/ folder should exist when AI is enabled"
        assert (agents_path / "__init__.py").exists()
        assert (agents_path / "assistant.py").exists()

    @pytest.mark.slow
    def test_agent_route_not_created_when_disabled(self, generated_project_minimal: Path) -> None:
        """Test that agent API route is not present when AI agent is disabled."""
        agent_route = (
            generated_project_minimal / "backend" / "app" / "api" / "routes" / "v1" / "agent.py"
        )
        assert not agent_route.exists(), "agent.py route should not exist when AI is disabled"


class TestGeneratedTemplateSyntax:
    """Test that generated Python files have valid syntax."""

    @pytest.mark.slow
    def test_minimal_project_valid_python_syntax(self, generated_project_minimal: Path) -> None:
        """Test all Python files in minimal project have valid syntax."""
        backend_path = generated_project_minimal / "backend"
        python_files = list(backend_path.rglob("*.py"))

        for py_file in python_files:
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(py_file)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Syntax error in {py_file}:\n{result.stderr}"

    @pytest.mark.slow
    def test_full_project_valid_python_syntax(self, generated_project_full: Path) -> None:
        """Test all Python files in full project have valid syntax."""
        backend_path = generated_project_full / "backend"
        python_files = list(backend_path.rglob("*.py"))

        for py_file in python_files:
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(py_file)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Syntax error in {py_file}:\n{result.stderr}"
