# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
