# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.12] - 2026-01-02

### Added

#### CrewAI Multi-Agent Framework Improvements

- **Full type annotations** for all CrewAI event handlers in `crewai_assistant.py`
- **Comprehensive event queue listener** with handlers for:
  - `crew_started`, `crew_completed`, `crew_failed`
  - `agent_started`, `agent_completed`
  - `task_started`, `task_completed`
  - `tool_started`, `tool_finished`
  - `llm_started`, `llm_completed`
- **Improved stream method** with proper thread and queue handling:
  - Natural completion path when receiving None sentinel
  - Race condition handling for thread death scenarios
  - Defensive code with `# pragma: no cover` for edge cases
- **100% test coverage** for CrewAI assistant module

### Fixed

#### Backend Fixes

- **Type annotations** - All mypy errors fixed across the codebase:
  - Added `Any` types where needed in `logfire_setup.py`
  - Fixed `Callable` types in `commands/__init__.py`
  - Added proper types to versioning middleware
  - Full type coverage for CrewAI event handlers
- **WebSocket disconnect handling** - Proper logging and cleanup when client disconnects during agent processing (lines 241-242 in `agent.py`)
- **Health endpoint edge cases** - Added `# pragma: no cover` for unreachable 503 response path (checks dict is always empty)
- **Abstract method coverage** - Added `# pragma: no cover` for abstract `run()` method in `BasePipeline`

#### Frontend Fixes

- **Timeline connector lines** for grouped messages now display correctly
- **Message grouping** visual indicators properly connect related messages

### Tests Added

- **100% code coverage achieved** (720 statements, 0 missing)
- Tests for all 11 CrewAI event handlers:
  - `test_crew_started_handler`, `test_crew_completed_handler`, `test_crew_failed_handler`
  - `test_agent_started_handler`, `test_agent_completed_handler`
  - `test_task_started_handler`, `test_task_completed_handler`
  - `test_tool_started_handler`, `test_tool_finished_handler`
  - `test_llm_started_handler`, `test_llm_completed_handler`
- Tests for CrewAI stream method edge cases:
  - `test_stream_complete_flow` - natural completion path
  - `test_stream_empty_queue_break` - queue empty handling
  - `test_stream_with_error` - error event handling
- Tests for WebSocket disconnect during processing:
  - `test_websocket_disconnect_during_stream`
  - `test_websocket_disconnect_during_processing`
- Tests for health endpoint edge cases:
  - `test_readiness_probe_503_unit` - 503 response logic

## [0.1.11] - 2026-01-02

### Added

#### LangGraph ReAct Agent Support

- **LangGraph as third AI framework option** alongside PydanticAI and LangChain
- New `--ai-framework langgraph` CLI option for project creation
- Interactive prompt includes "LangGraph (ReAct agent)" choice
- **ReAct (Reasoning + Acting) agent pattern** with graph-based architecture:
  - Agent node for LLM reasoning and tool decision
  - Tools node for executing tool calls
  - Conditional edges for tool execution loop
  - Memory-based checkpointing for conversation continuity
- **Full WebSocket streaming support** using `astream()` with dual modes:
  - `messages` mode for token-level LLM streaming
  - `updates` mode for node state changes (tool calls/results)
- **Tool result correlation** - proper `tool_call_id` matching between calls and results
- New template files:
  - `app/agents/langgraph_assistant.py` - LangGraphAssistant class with run() and stream()
  - WebSocket route implementation in `app/api/routes/v1/agent.py`
- New cookiecutter variable: `use_langgraph`
- Dependencies for LangGraph projects:
  - `langchain-core>=0.3.0`
  - `langchain-openai>=0.3.0` or `langchain-anthropic>=0.3.0`
  - `langgraph>=0.2.0`
  - `langgraph-checkpoint>=2.0.0`

### Changed

- **`AIFrameworkType` enum** extended with `LANGGRAPH` value
- **AI framework prompt** now shows three options: PydanticAI, LangChain, LangGraph
- **LLM provider validation** - OpenRouter not supported with LangGraph (same as LangChain)
- **`VARIABLES.md`** updated with `use_langgraph` documentation
- **Template `CLAUDE.md`** includes LangGraph in stack section

## [0.1.10] - 2025-12-27

### Added

#### Nginx Reverse Proxy Support

- **Nginx as alternative to Traefik** with two configuration modes:
  - `nginx_included`: Full Nginx setup in docker-compose.prod.yml
  - `nginx_external`: Nginx config template only, for external Nginx
- **Nginx configuration template** (`nginx/nginx.conf`) with:
  - Reverse proxy for backend API (api.DOMAIN)
  - Reverse proxy for frontend (DOMAIN) - conditional
  - Reverse proxy for Flower dashboard (flower.DOMAIN) - conditional
  - WebSocket support for `/ws` endpoint
  - Security headers (X-Frame-Options, X-Content-Type-Options, HSTS, etc.)
  - HTTP to HTTPS redirect
  - SSL/TLS configuration with modern cipher suites
  - Let's Encrypt ACME challenge support
- **SSL certificate directory** (`nginx/ssl/`) with setup instructions
- New cookiecutter variables:
  - `include_nginx_service`: Include Nginx container in docker-compose
  - `include_nginx_config`: Generate nginx configuration files
  - `use_nginx`: Using Nginx (included or external)
  - `use_traefik`: Using Traefik (included or external)

### Changed

- **Reverse proxy prompt** now offers 5 options:
  - Traefik (included in docker-compose) - default
  - Traefik (external, shared between projects)
  - Nginx (included in docker-compose)
  - Nginx (external, config template only)
  - None (expose ports directly)
- **`ReverseProxyType` enum** extended with `NGINX_INCLUDED` and `NGINX_EXTERNAL`
- **docker-compose.prod.yml** updated:
  - Added nginx service definition
  - Services use backend-internal network when nginx is selected
  - No ports exposed on backend/frontend when nginx handles traffic
- **`.env.prod.example`** includes DOMAIN variable for nginx configuration
- **`post_gen_project.py`** removes nginx/ folder when nginx is not selected

### Tests Added

- Tests for `NGINX_INCLUDED` and `NGINX_EXTERNAL` enum values
- Tests for cookiecutter context generation with all reverse proxy options
- Tests for `prompt_reverse_proxy()` with nginx choices

## [0.1.9] - 2025-12-26

### Added

#### SQLModel Support

- **Optional SQLModel ORM** as alternative to SQLAlchemy for PostgreSQL and SQLite
- New `--orm` CLI option: `--orm sqlalchemy` (default) or `--orm sqlmodel`
- Interactive prompt for ORM library selection when using `fastapi-fullstack new`
- SQLModel provides simplified syntax combining SQLAlchemy and Pydantic:
  ```python
  from sqlmodel import SQLModel, Field

  class User(SQLModel, table=True):
      id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
      email: str = Field(max_length=255, unique=True)
      is_active: bool = Field(default=True)
  ```
- Full Alembic compatibility maintained with SQLModel
- SQLAdmin works seamlessly with SQLModel models
- All database models updated with SQLModel variants:
  - `User`, `Item`, `Conversation`, `Message`, `Session`, `Webhook`, `WebhookDelivery`
- `VARIABLES.md` updated with new ORM variables: `orm_type`, `use_sqlalchemy`, `use_sqlmodel`

### Changed

- Database model templates now support conditional SQLModel/SQLAlchemy syntax
- `alembic/env.py` uses `SQLModel.metadata` when SQLModel is selected
- Repositories remain unchanged (SQLModel uses same AsyncSession and methods)

### Tests Added

- Tests for `OrmType` enum values
- Tests for `use_sqlalchemy` and `use_sqlmodel` computed fields
- Tests for SQLModel validation (requires PostgreSQL or SQLite)
- Tests for `prompt_orm_type()` function
- Updated `run_interactive_prompts()` tests with `prompt_orm_type` mocks

## [0.1.7] - 2025-12-23

### Added

#### Docker & Production

- **Optional Traefik reverse proxy** with three configuration modes:
  - `traefik_included`: Full Traefik setup in docker-compose.prod.yml (default)
  - `traefik_external`: Traefik labels only, for shared Traefik instances
  - `none`: No reverse proxy, ports exposed directly
- **`.env.prod.example` template** for production secrets management:
  - Conditional sections for PostgreSQL, Redis, JWT, Traefik, Flower
  - Required variable validation using `${VAR:?error}` syntax
  - Setup instructions in docker-compose.prod.yml header
- **Unique Traefik router names** using `project_slug` prefix for multi-tenant support:
  - `{project_slug}-api`, `{project_slug}-frontend`, `{project_slug}-flower`
  - Prevents conflicts when running multiple projects on same server

#### AI Agent Support

- **`AGENTS.md`** file for non-Claude AI agents (Codex, Copilot, Cursor, Zed, OpenCode)
- **Progressive disclosure documentation** in generated projects:
  - `docs/architecture.md` - layered architecture details
  - `docs/adding_features.md` - how to add endpoints, commands, tools
  - `docs/testing.md` - testing guide and examples
  - `docs/patterns.md` - DI, service, repository patterns
- **README.md** updated with "AI-Agent Friendly" section

### Changed

- **Template `CLAUDE.md` refactored** from 384 to ~80 lines following [progressive disclosure best practices](https://humanlayer.dev/blog/writing-a-good-claude-md)
- **Main project `CLAUDE.md`** updated with "Where to Find More Info" section
- **docker-compose.prod.yml** now uses `env_file: .env.prod` instead of inline defaults
- **Removed hardcoded credentials** (`changeme`) from docker-compose.prod.yml

### Security

- Production credentials no longer have insecure defaults
- `.env.prod` added to `.gitignore` to prevent committing secrets
- Required environment variables fail fast with descriptive error messages

## [0.1.6] - 2025-12-22

### Added

#### Multi-LLM Provider Support
- **Multiple LLM providers** for AI agents: OpenAI, Anthropic, and OpenRouter
- PydanticAI supports all three providers (OpenAI, Anthropic, OpenRouter)
- LangChain supports OpenAI and Anthropic
- New `--llm-provider` CLI option and interactive prompt
- Provider-specific API key configuration in `.env` and `config.py`

#### CLI Enhancements
- **`make create-admin` command** for quick admin user creation
- **Comprehensive CLI options** for `fastapi-fullstack create` command:
  - `--redis`, `--caching`, `--rate-limiting`
  - `--admin-panel`, `--websockets`
  - `--task-queue` (none/celery/taskiq/arq)
  - `--oauth-google`, `--session-management`
  - `--kubernetes`, `--ci` (github/gitlab/none)
  - `--sentry`, `--prometheus`
  - `--file-storage`, `--webhooks`
  - `--python-version` (3.11/3.12/3.13)
  - `--i18n`
- **Configuration presets** for common use cases:
  - `--preset production`: Full production setup with Redis, Sentry, K8s, Prometheus
  - `--preset ai-agent`: AI agent with WebSocket streaming and conversation persistence
- **Interactive rate limit configuration** when rate limiting is enabled:
  - Requests per period (default: 100)
  - Period in seconds (default: 60)
  - Storage backend (memory or Redis)

#### Documentation
- **Improved CLI documentation** in README explaining project CLI naming convention (`uv run <project_slug>`)
- **Makefile shortcuts** documented with `make help` command

#### Template Improvements
- **Generator version metadata** in generated projects (`pyproject.toml`):
  ```toml
  [tool.fastapi-fullstack]
  generator_version = "0.1.6"
  generated_at = "2025-12-22T10:30:00+00:00"
  ```
- **Centralized agent prompts** module (`app/agents/prompts.py`) for easier maintenance
- **Template variables documentation** (`template/VARIABLES.md`) with 88+ variables documented

#### Validation
- **Email validation** for `author_email` field using Pydantic's `EmailStr`
- **Tests for OpenRouter + LangChain** validation (combination is rejected)
- **Tests for agents folder** conditional creation

### Changed

#### Configuration Validation
- **Improved option combination validation** in `ProjectConfig`:
  - Admin panel requires PostgreSQL or SQLite (not MongoDB)
  - Caching requires Redis to be enabled
  - Session management requires a database
  - Conversation persistence requires a database
  - Rate limiting with Redis storage requires Redis enabled
  - OpenRouter is only available with PydanticAI (not LangChain)

#### Database Support
- **Admin panel prompt** now appears for both PostgreSQL and SQLite (previously only PostgreSQL)
- **Database-specific post-generation messages**:
  - PostgreSQL: `make docker-db` + migration commands
  - SQLite: Auto-creation note + migration commands (no Docker)
  - MongoDB: `make docker-mongo` (no migrations)
- **Added `close_db()` function** for SQLite database consistency

#### Project Name Handling
- **Unified project name validation** between `prompts.py` and `config.py`
- Extracted validation into `_validate_project_name()` function with clear error messages
- Shows converted project name to user when it differs from input

### Fixed

#### Backend Fixes
- **Conversation list API response format**: Changed `/conversations` and `/conversations/{id}/messages` endpoints to return paginated response `{ items: [...], total: N }` instead of raw array, fixing frontend conversation list not loading after page refresh
- **Database session handling**: Split `get_db_session` into async generator for FastAPI `Depends()` and `@asynccontextmanager` for manual use (WebSocket handlers)
- **WebSocket authentication**:
  - Update `deps.py` to use `get_db_context` for WebSocket auth
  - Add cookie-based authentication support for WebSocket (`access_token` cookie)
  - Now accepts token via query parameter OR cookie for flexibility
- **WebSocket exception handling**: Fix `AttributeError` when exception occurs on WebSocket connection (`request.method` doesn't exist for WebSocket)
- **WebSocket conversation persistence**:
  - Fix `get_db_session` vs `get_db_context` usage (async generator vs async context manager)
  - Fix event name mismatch: backend now sends `conversation_created` to match frontend expectation
- **Docker Compose**: Fix `env_file` path from `.env` to `./backend/.env`
- **ValidationInfo typing**: Add proper type hints to all field validators in `config.py`

#### Frontend Fixes
- **ThemeToggle hydration mismatch**: Add mounted state to prevent SSR/client mismatch
- **Button component**: Extract `asChild` prop to prevent DOM warning
- **ConversationList**: Add default value for conversations to prevent undefined error
- **New Chat button**:
  - Create conversation in database immediately (eager creation)
  - Clear messages properly when switching conversations
  - Fix message appending issue when switching between conversations
- **Conversation store**: Add defensive checks for undefined state

#### CLI Fixes
- **Consistent package name**: Changed from `fastapi-gen` to `fastapi-fullstack` in version option
- **Makefile**: Always generated now (removed from optional dev tools)

#### Template/Generator Fixes
- **Ruff dependency in hooks**: Graceful handling when ruff is not installed:
  - Check PATH for ruff binary
  - Fall back to `uvx ruff` if uv is available
  - Fall back to `python -m ruff` if available as module
  - Show friendly warning if ruff is not available
- **Dynamic generator version**: Replaced hardcoded version with `DYNAMIC` placeholder
- **Unused files cleanup**: Improved post-generation hook to remove:
  - AI agent files based on framework selection
  - Example CRUD files when disabled
  - Conversation, webhook, session files when features disabled
  - Worker directory when no background tasks selected
  - Empty directories automatically
- **`.env` file location**: Move `.env.example` from root to `backend/`

### Tests Added

- Tests for all configuration validation combinations
- Tests for project name validation edge cases
- Tests for `new` command `--output` option
- Tests for OpenRouter + LangChain validation
- Tests for admin panel prompt with SQLite
- Tests for agents folder conditional creation
- Tests for email validation (config and prompts)
- Tests for rate limit configuration prompts
