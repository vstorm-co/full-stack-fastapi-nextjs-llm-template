# GrowthEngine OS

Autonomous marketing-firm infrastructure: a master intake gateway that triages inbound demand across a fleet of owned engines — video ads (Influencer Echo), print collateral (Print-For-You), email drip (OmniTrickle / EchoLink), and local attribution (OmniLocal OS) — with an operator console and approval-first automation boundaries.

## Structure

- `backend/` — FastAPI service (Motor + MongoDB). Intake triage, lead feed, chaser dispatch, telemetry, plus the GrowthEngine domain routes (products, campaigns, tasks, outreach assets, analytics).
- `frontend/` — Next.js 15 app (dark glassmorphic theme, Space Grotesk / DM Mono).

## Routes

| Route | Purpose |
|---|---|
| `/` | Public landing + 3-question dynamic gatekeeper (intake widget) |
| `/portal/onboarding` | Cabinet Suite — self-serve fleet consoles (Echo / Print / OmniTrickle / OmniLocal) |
| `/booking/growth` | Core Growth Pod booking intake (logs under `tier_growth_pod`) |
| `/booking/vip-fast-track` | White-glove booking intake (logs under `tier_white_glove_agency`) |
| `/privacy` · `/terms` | Legal pages (data handling + terms for automated marketing) |
| `/admin/operator` | Operator command center (Bearer-gated by operator key) |

## Backend API

| Endpoint | Purpose |
|---|---|
| `POST /api/leads/intake` | Triage: bottleneck + weekly budget + requested modules → routing verdict + `target_url` |
| `GET /api/leads?tier=…` | Operator lead feed (`under_250` / `250_to_1250` / `over_1250`) — Bearer `OPERATOR_ADMIN_KEY` |
| `POST /api/leads/{id}/chaser` | Dispatch SMS + email chaser sequence marker — Bearer `OPERATOR_ADMIN_KEY` |
| `GET /api/telemetry` | Subsystem health: API, MongoDB, n8n webhook — Bearer `OPERATOR_ADMIN_KEY` |
| `GET /api/health`, `/api/dashboard/summary` | Health + pipeline summary |

Triage matrix: `< $250/wk → self-serve` (`/portal/onboarding`), `$250–$1,250/wk → growth pod` (`/booking/growth`), `$1,250+/wk or full-operating-stack bottleneck → white-glove` (`/booking/vip-fast-track`).

## Environment

Secrets live in the runtime environment (Keys / API keys UI) — never commit credentials.

- `MONGODB_URI` — required by the backend (database defaults to `growthengine`).
- `NEXT_PUBLIC_API_BASE_URL` — frontend → backend base (default `http://localhost:8000`).
- `OPERATOR_ADMIN_KEY` — backend Bearer secret gating `GET /api/leads`, chaser dispatch, and telemetry (fail-closed: unset returns 503). Keep public intake endpoints reachable.
- `NEXT_PUBLIC_OPERATOR_ADMIN_KEY` — operator console gate and Bearer header; must match `OPERATOR_ADMIN_KEY`.
- `N8N_WEBHOOK_URL` — optional n8n sync endpoint surfaced in telemetry.
- `SELF_SERVE_URL` / `GROWTH_CALENDAR_URL` / `VIP_CALENDAR_URL` — override intake routing targets.

## Run locally

```bash
# backend (from repo root, with MONGODB_URI set)
uv run uvicorn backend.main:app --app-dir backend --host 0.0.0.0 --port 8000

# frontend
cd frontend
npm install
npm run dev
```

## Operating boundary

Tests: `uv run pytest backend/tests` (intake triage matrix) alongside the generator suite in `tests/`.

The system routes, drafts, and coordinates relentlessly — but external sends, publishing, contact enrichment, and paid ad spend remain explicit approval actions with audit trails, opt-outs, and suppression lists. "Autonomous" describes continuous planning and iteration, never unsupervised spend or messaging.
