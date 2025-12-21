.PHONY: install format lint test run clean help db-init

# === Setup ===
install:
	uv sync --directory backend --dev
{%- if cookiecutter.enable_precommit %}
	@if git rev-parse --git-dir > /dev/null 2>&1; then \
		uv run --directory backend pre-commit install; \
	else \
		echo "⚠️  Not a git repository - skipping pre-commit install"; \
		echo "   Run 'git init && make install' to set up pre-commit hooks"; \
	fi
{%- endif %}
	@echo ""
	@echo "✅ Installation complete!"
	@echo ""
	@echo "Next steps:"
{%- if not cookiecutter.generate_env %}
	@echo "  • cd backend && cp .env.example .env"
	@echo "  • Edit backend/.env with your settings"
{%- endif %}
{%- if cookiecutter.use_postgresql %}
	@echo "  • make docker-db        # Start PostgreSQL"
	@echo "  • make db-upgrade       # Apply migrations"
{%- endif %}
	@echo "  • make run              # Start development server"
{%- if cookiecutter.generate_env %}
	@echo ""
	@echo "Note: backend/.env is pre-configured for development"
{%- endif %}

# === Code Quality ===
format:
	uv run --directory backend ruff format app tests cli
	uv run --directory backend ruff check app tests cli --fix

lint:
	uv run --directory backend ruff check app tests cli
	uv run --directory backend ruff format app tests cli --check
	uv run --directory backend mypy app

# === Testing ===
test:
	uv run --directory backend pytest tests/ -v

test-cov:
	uv run --directory backend pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

{%- if cookiecutter.use_postgresql or cookiecutter.use_sqlite %}

# === Database ===
{%- if cookiecutter.use_postgresql and cookiecutter.enable_docker %}
db-init: docker-db
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 3
	uv run --directory backend {{ cookiecutter.project_slug }} db upgrade
	@echo ""
	@echo "✅ Database initialized!"
{%- else %}
db-init:
	uv run --directory backend {{ cookiecutter.project_slug }} db upgrade
	@echo ""
	@echo "✅ Database initialized!"
{%- endif %}

db-migrate:
	@read -p "Migration message: " msg; \
	uv run --directory backend {{ cookiecutter.project_slug }} db migrate -m "$$msg"

db-upgrade:
	uv run --directory backend {{ cookiecutter.project_slug }} db upgrade

db-downgrade:
	uv run --directory backend {{ cookiecutter.project_slug }} db downgrade

db-current:
	uv run --directory backend {{ cookiecutter.project_slug }} db current

db-history:
	uv run --directory backend {{ cookiecutter.project_slug }} db history
{%- endif %}

# === Server ===
run:
	uv run --directory backend {{ cookiecutter.project_slug }} server run --reload

run-prod:
	uv run --directory backend {{ cookiecutter.project_slug }} server run --host 0.0.0.0 --port 8000

routes:
	uv run --directory backend {{ cookiecutter.project_slug }} server routes

{%- if cookiecutter.use_jwt %}

# === Users ===
user-create:
	uv run --directory backend {{ cookiecutter.project_slug }} user create

user-list:
	uv run --directory backend {{ cookiecutter.project_slug }} user list
{%- endif %}

{%- if cookiecutter.use_celery %}

# === Celery ===
celery-worker:
	uv run --directory backend {{ cookiecutter.project_slug }} celery worker

celery-beat:
	uv run --directory backend {{ cookiecutter.project_slug }} celery beat

celery-flower:
	uv run --directory backend {{ cookiecutter.project_slug }} celery flower
	@echo ""
	@echo "✅ Flower started at http://localhost:5555"
{%- endif %}

{%- if cookiecutter.use_taskiq %}

# === Taskiq ===
taskiq-worker:
	uv run --directory backend {{ cookiecutter.project_slug }} taskiq worker

taskiq-scheduler:
	uv run --directory backend {{ cookiecutter.project_slug }} taskiq scheduler
{%- endif %}

{%- if cookiecutter.enable_docker %}

# === Docker: Backend (Development) ===
docker-up:
	docker-compose up -d
	@echo ""
	@echo "✅ Backend services started!"
	@echo "   API: http://localhost:{{ cookiecutter.backend_port }}"
	@echo "   Docs: http://localhost:{{ cookiecutter.backend_port }}/docs"
{%- if cookiecutter.use_celery %}
	@echo "   Flower: http://localhost:5555"
{%- endif %}
{%- if cookiecutter.use_postgresql %}
	@echo "   PostgreSQL: localhost:5432"
{%- endif %}
{%- if cookiecutter.enable_redis %}
	@echo "   Redis: localhost:6379"
{%- endif %}

docker-down:
	docker-compose down
{%- if cookiecutter.use_frontend %}
	docker-compose -f docker-compose.frontend.yml down 2>/dev/null || true
{%- endif %}

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

docker-shell:
	docker-compose exec app /bin/bash

{%- if cookiecutter.use_frontend %}

# === Docker: Frontend (Development) ===
docker-frontend:
	docker-compose -f docker-compose.frontend.yml up -d
	@echo ""
	@echo "✅ Frontend started!"
	@echo "   URL: http://localhost:{{ cookiecutter.frontend_port }}"
	@echo ""
	@echo "Note: Backend must be running (make docker-up)"

docker-frontend-down:
	docker-compose -f docker-compose.frontend.yml down

docker-frontend-logs:
	docker-compose -f docker-compose.frontend.yml logs -f

docker-frontend-build:
	docker-compose -f docker-compose.frontend.yml build
{%- endif %}

# === Docker: Production (with Traefik) ===
docker-prod:
	docker-compose -f docker-compose.prod.yml up -d
	@echo ""
	@echo "✅ Production services started with Traefik!"
	@echo ""
	@echo "Endpoints (replace DOMAIN with your domain):"
{%- if cookiecutter.use_frontend %}
	@echo "   Frontend: https://$$DOMAIN"
{%- endif %}
	@echo "   API: https://api.$$DOMAIN"
{%- if cookiecutter.use_celery %}
	@echo "   Flower: https://flower.$$DOMAIN"
{%- endif %}
	@echo "   Traefik: https://traefik.$$DOMAIN"

docker-prod-down:
	docker-compose -f docker-compose.prod.yml down

docker-prod-logs:
	docker-compose -f docker-compose.prod.yml logs -f

docker-prod-build:
	docker-compose -f docker-compose.prod.yml build

{%- if cookiecutter.use_postgresql %}

# === Docker: Individual Services ===
docker-db:
	docker-compose up -d db
	@echo ""
	@echo "✅ PostgreSQL started on port 5432"
	@echo "   Connection: postgresql://postgres:postgres@localhost:5432/{{ cookiecutter.project_slug }}"

docker-db-stop:
	docker-compose stop db
{%- endif %}

{%- if cookiecutter.enable_redis %}

docker-redis:
	docker-compose up -d redis
	@echo ""
	@echo "✅ Redis started on port 6379"

docker-redis-stop:
	docker-compose stop redis
{%- endif %}
{%- endif %}

# === Cleanup ===
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml

# === Help ===
help:
	@echo ""
	@echo "{{ cookiecutter.project_name }} - Available Commands"
	@echo "======================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies + pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make run           Start dev server (with hot reload)"
	@echo "  make test          Run tests"
	@echo "  make lint          Check code quality"
	@echo "  make format        Auto-format code"
	@echo ""
{%- if cookiecutter.use_postgresql or cookiecutter.use_sqlite %}
	@echo "Database:"
	@echo "  make db-init       Initialize database (start + migrate)"
	@echo "  make db-migrate    Create new migration"
	@echo "  make db-upgrade    Apply migrations"
	@echo "  make db-downgrade  Rollback last migration"
	@echo "  make db-current    Show current migration"
	@echo ""
{%- endif %}
{%- if cookiecutter.use_jwt %}
	@echo "Users:"
	@echo "  make user-create   Create new user (interactive)"
	@echo "  make user-list     List all users"
	@echo ""
{%- endif %}
{%- if cookiecutter.use_celery %}
	@echo "Celery:"
	@echo "  make celery-worker  Start Celery worker"
	@echo "  make celery-beat    Start Celery beat scheduler"
	@echo "  make celery-flower  Start Flower monitoring UI"
	@echo ""
{%- endif %}
{%- if cookiecutter.use_taskiq %}
	@echo "Taskiq:"
	@echo "  make taskiq-worker     Start Taskiq worker"
	@echo "  make taskiq-scheduler  Start Taskiq scheduler"
	@echo ""
{%- endif %}
{%- if cookiecutter.enable_docker %}
	@echo "Docker (Development):"
	@echo "  make docker-up            Start backend services"
	@echo "  make docker-down          Stop all services"
	@echo "  make docker-logs          View backend logs"
	@echo "  make docker-build         Build backend images"
{%- if cookiecutter.use_frontend %}
	@echo "  make docker-frontend      Start frontend (separate)"
	@echo "  make docker-frontend-down Stop frontend"
{%- endif %}
{%- if cookiecutter.use_postgresql %}
	@echo "  make docker-db            Start only PostgreSQL"
{%- endif %}
{%- if cookiecutter.enable_redis %}
	@echo "  make docker-redis         Start only Redis"
{%- endif %}
	@echo ""
	@echo "Docker (Production with Traefik):"
	@echo "  make docker-prod          Start production stack"
	@echo "  make docker-prod-down     Stop production stack"
	@echo "  make docker-prod-logs     View production logs"
	@echo ""
{%- endif %}
	@echo "Other:"
	@echo "  make routes        Show all API routes"
	@echo "  make clean         Clean cache files"
	@echo ""
