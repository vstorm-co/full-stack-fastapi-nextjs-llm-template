# CLAUDE.md

## Project Overview

**{{ cookiecutter.project_name }}** - FastAPI application generated with [Full-Stack FastAPI + Next.js Template](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template).

**Stack:** FastAPI + Pydantic v2
{%- if cookiecutter.use_postgresql %}, PostgreSQL (async){%- endif %}
{%- if cookiecutter.use_mongodb %}, MongoDB (async){%- endif %}
{%- if cookiecutter.use_sqlite %}, SQLite{%- endif %}
{%- if cookiecutter.use_jwt %}, JWT auth{%- endif %}
{%- if cookiecutter.enable_redis %}, Redis{%- endif %}
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_pydantic_ai %}, PydanticAI{%- endif %}
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_langchain %}, LangChain{%- endif %}
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_langgraph %}, LangGraph{%- endif %}
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_crewai %}, CrewAI{%- endif %}
{%- if cookiecutter.use_celery %}, Celery{%- endif %}
{%- if cookiecutter.use_taskiq %}, Taskiq{%- endif %}
{%- if cookiecutter.use_frontend %}, Next.js 15{%- endif %}

## Commands

```bash
# Backend
cd backend
uv run uvicorn app.main:app --reload --port {{ cookiecutter.backend_port }}
pytest
ruff check . --fix && ruff format .
{%- if cookiecutter.use_postgresql or cookiecutter.use_sqlite %}

# Database
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "Description"
{%- endif %}
{%- if cookiecutter.use_frontend %}

# Frontend
cd frontend
bun dev
bun test
{%- endif %}
{%- if cookiecutter.enable_docker %}

# Docker
docker compose up -d
{%- endif %}
```

## Project Structure

```
backend/app/
├── api/routes/v1/    # HTTP endpoints
├── services/         # Business logic
├── repositories/     # Data access
├── schemas/          # Pydantic models
├── db/models/        # Database models
├── core/config.py    # Settings
{%- if cookiecutter.enable_ai_agent %}
├── agents/           # AI agents
{%- endif %}
└── commands/         # CLI commands
```

## Key Conventions

- Use `db.flush()` in repositories (not `commit`)
- Services raise domain exceptions (`NotFoundError`, `AlreadyExistsError`)
- Schemas: separate `Create`, `Update`, `Response` models
- Commands auto-discovered from `app/commands/`

## Where to Find More Info

Before starting complex tasks, read relevant docs:
- **Architecture details:** `docs/architecture.md`
- **Adding features:** `docs/adding_features.md`
- **Testing guide:** `docs/testing.md`
- **Code patterns:** `docs/patterns.md`

## Environment Variables

Key variables in `.env`:
```bash
ENVIRONMENT=local
{%- if cookiecutter.use_postgresql %}
POSTGRES_HOST=localhost
POSTGRES_PASSWORD=secret
{%- endif %}
{%- if cookiecutter.use_jwt %}
SECRET_KEY=change-me-use-openssl-rand-hex-32
{%- endif %}
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_openai %}
OPENAI_API_KEY=sk-...
{%- endif %}
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_anthropic %}
ANTHROPIC_API_KEY=sk-ant-...
{%- endif %}
{%- if cookiecutter.enable_logfire %}
LOGFIRE_TOKEN=your-token
{%- endif %}
```
