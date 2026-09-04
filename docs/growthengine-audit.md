# GrowthEngine OS — Full Audit (Fleet Build)

Scope: FastAPI + Motor backend, Next.js 15 frontend, routes `/`, `/portal/onboarding`, `/booking/vip-fast-track`, `/admin/operator`. Audit date: current fleet build. Each item: ✅ verified working · 🔶 partial/needs work · ❌ missing.

## Product

- **Solves the problem** ✅ — The master intake captures a real signal (bottleneck + weekly spend + requested fleet modules) and routes to distinct outcomes instead of a generic "contact us." The operator console closes the loop: leads land in Mongo, chaser dispatch is a real endpoint, and telemetry reports subsystem state.
- **Core loop completeness** 🔶 — Booking destinations for the mid tier (`/booking/growth`) still 404: the triage matrix points there but only the VIP page exists. Create the growth booking page (mirror of `vip-fast-track` with `weekly_budget: "250_to_1250"`, `requested_modules` selectable) before relying on end-to-end routing.
- **Truthfulness of claims** 🔶 — Copy says "self-serve tools at $300/mo" and "$4k/mo retainer"; no billing exists. Either ship payments or label these as target pricing.

## Design

- **Visual polish** ✅ — Consistent matte-black glassmorphic system (`#08080A`, violet/fuchsia glow, Space Grotesk + DM Mono), central CSS tokens, coherent sections on all four pages.
- **Design consistency** ✅ — New pages reuse the shared token/class system rather than inventing palettes.
- **Iconography** ✅ — Lucide icons used uniformly.
- **Deferred** 🔶 — No design-token audit tooling; token drift is possible as pages grow.

## Mobile & Tablet

- **Responsive layouts** ✅ — All four pages collapse to single-column at ≤900px and ≤560px; the operator lead table becomes stacked cards.
- **Touch targets** ✅ — Primary CTAs/buttons ≥40px; click-to-advance cards are large.
- **Untested on-device** 🔶 — No real-device pass (iOS Safari, Android Chrome) has been run; the console/gate flows should be smoke-tested for zoom/scroll issues and `100vh` on mobile browsers.

## Performance

- **Bundle** ✅ — Each route ~104–113 kB first-load JS; static prerendering for all pages; no client-side data on public pages.
- **Fonts** ✅ (fixed this session) — Google Fonts moved from render-blocking CSS `@import` to `<link preconnect>` + stylesheet in the document head.
- **Icons** ✅ — Lucide is tree-shaken per-import.
- **Backend latency** 🔶 — Every operator action makes direct fetches; no caching layer for the lead feed; telemetry pings Mongo per render. Fine at prototype scale; add a cache (or polling interval) before scale.
- **Images/media** 🔶 — No real ad-video/print-asset rendering exists yet; when added, size/compress and stream via CDN.

## Documentation

- **README** ✅ (updated this session) — Now documents architecture, all routes, the API table, triage matrix, env keys, run steps, and the approval boundary.
- **Audit doc** ✅ — This file.
- **Missing** 🔶 — No user-facing help/FAQ inside the portal, no API reference page beyond the README table, no changelog for the fleet additions.

## Engineering & Scale

### Code quality
- ✅ Ruff-clean backend, typed Pydantic schemas, typed TSX, ESLint-free but strict `tsc` via Next build.
- 🔶 Backend `backend/app/` is not covered by the repo's lint/format config at the root (ruff invoked manually); wire it into `pyproject.toml` so CI enforces it. Frontend has no lint script at all — add `eslint` or rely on `tsc`.

### Accessibility
- ✅ (fixed this session) — Real `<label for>` on the new booking form; `aria-label` on intake inputs and operator passcode; global `:focus-visible` rings; `prefers-reduced-motion` disables animation.
- ✅ Semantic landmarks (`header/nav/main/section/footer`), native `<details>`, `role=tablist` on creative tabs.
- 🔶 No automated axe testing, no keyboard-only walkthrough recorded, no color-contrast measurement (muted grays on near-black should be verified ≥4.5:1 for body text), creative tab buttons should use `role="tab"` + arrow-key handling.

### Scalability & reliability
- ✅ Indexes on `leads` (email, weekly_budget+created_at, stack+created_at), tasks, campaigns, analytics; reads sorted on `created_at`.
- 🔶 Single-process Motor client, no pooling knobs tuned, no queue/workers — the chaser endpoint only stamps MongoDB state (no real SMS/email worker yet). In-memory state only; rate limits absent on `/api/leads/intake` (spam risk). A surge of intake → unbounded inserts; add a limiter and queue the chaser handoff.

### Error handling
- ✅ 404s for missing products/tasks/leads; intake UI has sending/error/retry states; telemetry degrades gracefully to "offline."
- 🔶 No global exception handler on the API (DB-down raises generic 500s), no structured logging, no timeout/retry on outbound providers. Chaser dispatch errors surface only in the UI.

### Database
- ✅ Motor lifecycle via lifespan; startup ping + index creation; `verification_state`, `stack`, `provisioned_subsystem` persisted per lead; booking fields (`preferred_call_time`, `target_start_date`) added this session.
- 🔶 No migrations/versioning for schema drift (Mongo documents are flexible — pin a `schema_version` field), no unique constraints on email (duplicate intakes allowed by design — confirm that's intended), no TTL/retention policy, and the operator page mislabels nothing — but there is no data deletion API for GDPR erasure.

### Test coverage & QA
- ✅ 119 generator tests pass; `ruff` clean; Next build type-checks every page.
- ❌ **Zero tests for the GrowthEngine backend routes** — the triage matrix (3 tiers + full-stack override), chaser dispatch, telemetry, and 404 paths are untested. Highest-value addition: tests with a mocked store asserting tier → stack/action/URL for all four bottleneck/budget combinations, and that `/api/leads/{id}/chaser` marks the doc.
- ❌ No frontend component/interaction tests (intake widget steps, gate unlock, chaser button states).

### Integrations
- 🔶 Deliberately not yet wired: Ayrshare (Echo deployment), print renderer, email/SMS provider, POS/KDS sync, n8n webhook (only reflected in telemetry). Each must be an isolated adapter behind an interface with approval/opt-out/rate-limit checks — documented as the next build layer.

### Cloud cost / backend efficiency
- 🔶 No metrics on spend. Immediate wins: cap the lead feed query (already `limit(200)`), add an index-backed pagination cursor instead of repeated full scans, batch telemetry instead of per-render pings, and add connection pooling/monitoring before any paid hosting.

## Security & Access

### Security
- ✅ (fixed this session) — Security headers middleware added (`nosniff`, `DENY` framing, referrer policy, permissions policy); secrets only via env; no credentials in source.
- ✅ Server-side validation on all inputs (Pydantic enums/lengths); intake is not an open redirect — `target_url` comes from server settings, never client input.
- 🔶 No CSP yet (inline JSON-LD scripts block a strict policy — needs a reviewed allowlist), no rate limiting, no request-size cap documented, MongoDB should be on a private network with Atlas IP allowlist in any real deployment.

### Identity & access
- ❌ No real auth anywhere. The operator console gate is a **client-side** key check (anyone can read it from the bundle) — the page itself says real authorization must live on the API. Leads are world-readable via `GET /api/leads` if the API is exposed. Before any live deployment: API-key/Bearer auth on operator endpoints + row-level ownership when multi-tenant.

### Dependency & supply chain
- 🔶 Live `npm audit` could not complete in the sandbox (registry timeout); `--offline` against the local advisory cache reports 0 vulnerabilities. An earlier session reported 1 moderate + 1 high after install — re-run `npm audit` against the registry before release and remediate.
- 🔶 Backend pins are minimum-version ranges (`>=`), not locks — generate `uv.lock`/pinned requirements for reproducible deploys.

## Marketing & Revenue

### SEO
- ✅ Metadata + OpenGraph + `metadataBase` + Organization and FAQPage JSON-LD; theme-color viewport added this session.
- 🔶 No sitemap.xml, robots.txt, canonical per-page URLs, or OG images; `/booking/vip-fast-track` and `/portal/onboarding` share the generic title — add per-page metadata.

### Landing page optimization
- ✅ Strong single CTA loop: hero → 3-question gate → routed destination; comparison, proof, pricing brackets, FAQ support the decision.
- 🔶 No analytics/heatmap on the landing page to measure gate drop-off by step — add event capture at each intake step before optimizing.

### Copy & content
- ✅ Distinct, concrete value copy; honest no-guarantee framing retained; owned-stack messaging clear.
- 🔶 Guarantee-adjacent language ("never fails", "every deployment verified as revenue") was correctly avoided — keep it that way; consumer-protection regulators treat revenue promises as claims.

### Branding
- ✅ Cohesive tokens across all four pages; brand mark/voice consistent.
- 🔶 No brand guide/lockup spec; favicon/OG assets missing.

### Internationalization
- ❌ No i18n. All copy hardcoded in English, `<html lang="en">`, no locale routing or string extraction. Needed only once non-English markets are targeted — but the intake API stores free-text fields with English enum values, so plan the locale boundary before then.

### Billing & tax
- ❌ No billing. Stated prices ($300/mo engines, $4k/mo retainer, ad spend pass-through) have no subscription, invoicing, tax/VAT, or spend-cap implementation. Ad-spend approvals are UI copy only. Before selling: payment provider + tax handling + per-channel budget caps enforced server-side.

## Legal & Compliance

### Legal
- ❌ No LICENSE file in this workspace, no Terms of Service, no client agreement covering retainer/performance claims, and no cancellation policy. The pages make service promises — ship legal pages before real clients sign.
- 🔶 Third-party fonts loaded from Google Fonts CDN without a privacy note.

### Privacy
- ❌ The intake and booking forms collect name, work email, phone, company, and ad-spend data with only a microcopy "consent-aware follow-up" line — no privacy policy, no retention statement, no erasure/export mechanism, no suppression-list handling documented.
- 🔶 Telemetry is minimal and separate from lead data (good); formalize the data-flow boundary in a written policy before processing real leads.

## What changed in this audit pass

1. New `/booking/vip-fast-track` route with labeled form → logs white-glove leads (full-stack bottleneck, `over_1250`, all four modules, call time, start date persisted).
2. Backend: `preferred_call_time` / `target_start_date` fields persisted; security-headers middleware.
3. Frontend: fonts de-render-blocked, `metadataBase` + robots + viewport/theme-color, `aria-label`s, `:focus-visible`, `prefers-reduced-motion`.
4. README rewritten for the fleet architecture; this audit doc refreshed.

## Top 5 priorities before real clients

1. **API auth + rate limiting** on intake and operator endpoints (currently world-readable when exposed).
2. **Backend tests** for the triage matrix and chaser/telemetry routes.
3. **Privacy policy + data deletion/opt-out API** — forms collect PII today.
4. **Billing** so published prices are enforceable; enforce spend caps server-side.
5. **Remaining booking route** `/booking/growth` + live provider adapters behind approval gates.
