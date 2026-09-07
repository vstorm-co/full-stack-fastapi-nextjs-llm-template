"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ArrowLeft,
  Database,
  Inbox,
  Lock,
  MessageSquareText,
  Radio,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const ADMIN_KEY = process.env.NEXT_PUBLIC_OPERATOR_ADMIN_KEY ?? "";

// Operator endpoints require the same key as a Bearer token (backend
// OPERATOR_ADMIN_KEY). Client-side gate and API gate must share the value.
const AUTH_HEADERS: HeadersInit = ADMIN_KEY ? { Authorization: `Bearer ${ADMIN_KEY}` } : {};

type WeeklyBudgetTier = "under_250" | "250_to_1250" | "over_1250";
type TierFilter = "all" | WeeklyBudgetTier;

type LeadRow = {
  id: string;
  name: string;
  email: string;
  phone: string;
  company_name: string;
  primary_bottleneck: string;
  weekly_budget: WeeklyBudgetTier;
  requested_modules: string[];
  stack: string;
  provisioned_subsystem: string | null;
  trigger_chaser: boolean;
  routing_action: string;
  target_url: string | null;
  assigned_agent: string | null;
  verification_state: string;
  chaser_dispatched_at: string | null;
  created_at: string;
};

type SubsystemStatus = {
  subsystem: string;
  status: "online" | "degraded" | "offline" | "not_configured";
  detail: string;
};

type Telemetry = {
  backend: SubsystemStatus;
  mongodb: SubsystemStatus;
  n8n_webhook: SubsystemStatus;
  generated_at: string;
};

const tierTabs: { key: TierFilter; label: string }[] = [
  { key: "all", label: "All tiers" },
  { key: "under_250", label: "Self-serve" },
  { key: "250_to_1250", label: "Growth pod" },
  { key: "over_1250", label: "White-glove" },
];

const bottleneckLabels: Record<string, string> = {
  lead_generation: "Lead generation",
  operational_overhead: "Operational overhead",
  local_foot_traffic: "Local foot traffic",
  retention_and_churn: "Retention & churn",
  full_operating_stack: "Full operating stack",
};

function tagClass(stack: string) {
  if (stack.includes("growth")) return "lead-tag growth";
  if (stack.includes("white_glove")) return "lead-tag white-glove";
  return "lead-tag self-serve";
}

function telStateClass(status: SubsystemStatus["status"]) {
  if (status === "online") return "tel-state ok";
  if (status === "not_configured" || status === "degraded") return "tel-state off";
  return "tel-state down";
}

const telemetryCards: {
  key: "backend" | "mongodb" | "n8n_webhook";
  label: string;
  icon: typeof Zap;
}[] = [
  { key: "backend", label: "FastAPI backend", icon: Zap },
  { key: "mongodb", label: "MongoDB connection", icon: Database },
  { key: "n8n_webhook", label: "N8N webhook sync", icon: MessageSquareText },
];

export default function OperatorConsole() {
  const [passcode, setPasscode] = useState("");
  const [unlocked, setUnlocked] = useState(false);
  const [gateError, setGateError] = useState(false);

  // Console state
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [leads, setLeads] = useState<LeadRow[] | null>(null);
  const [filter, setFilter] = useState<TierFilter>("all");
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [dispatched, setDispatched] = useState<Record<string, string>>({});

  useEffect(() => {
    if (typeof window !== "undefined" && window.sessionStorage.getItem("ge-operator-unlocked") === "1") {
      setUnlocked(true);
    }
  }, []);

  const loadTelemetry = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/telemetry`, { headers: AUTH_HEADERS });
      if (!res.ok) throw new Error(`telemetry failed: ${res.status}`);
      setTelemetry((await res.json()) as Telemetry);
    } catch {
      setTelemetry(null);
    }
  }, []);

  const loadLeads = useCallback(
    async (tier: TierFilter) => {
      setLoading(true);
      setApiError(null);
      try {
        const qs = tier === "all" ? "" : `?tier=${tier}`;
        const res = await fetch(`${API_BASE}/api/leads${qs}`, { headers: AUTH_HEADERS });
        if (!res.ok) throw new Error(`leads failed: ${res.status}`);
        const data = (await res.json()) as LeadRow[];
        setLeads(data);
      } catch (err) {
        setApiError(err instanceof Error ? err.message : "API unreachable");
        setLeads(null);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    if (!unlocked) return;
    void loadTelemetry();
    void loadLeads(filter);
  }, [unlocked, filter, loadTelemetry, loadLeads]);

  async function dispatchChaser(lead: LeadRow) {
    setDispatched((prev) => ({ ...prev, [lead.id]: "sending" }));
    try {
      const res = await fetch(`${API_BASE}/api/leads/${lead.id}/chaser`, { method: "POST", headers: AUTH_HEADERS });
      if (!res.ok) throw new Error(`chaser failed: ${res.status}`);
      const data = (await res.json()) as { dispatched_at: string };
      setDispatched((prev) => ({ ...prev, [lead.id]: data.dispatched_at }));
      void loadTelemetry();
    } catch {
      setDispatched((prev) => ({ ...prev, [lead.id]: "error" }));
    }
  }

  function submitGate(e: React.FormEvent) {
    e.preventDefault();
    if (passcode === ADMIN_KEY) {
      setUnlocked(true);
      setGateError(false);
      window.sessionStorage.setItem("ge-operator-unlocked", "1");
    } else {
      setGateError(true);
    }
  }

  if (!ADMIN_KEY) {
    return (
      <main className="operator-shell">
        <nav className="portal-nav">
          <a className="back-link" href="/">
            <ArrowLeft size={15} /> growthengine OS
          </a>
        </nav>
        <section className="operator-gate">
          <ShieldCheck size={26} style={{ color: "#c084fc" }} />
          <h1>Operator access not configured</h1>
          <p>
            Set <span className="mono">NEXT_PUBLIC_OPERATOR_ADMIN_KEY</span> in the environment (Keys / API keys tab) to
            arm the command center. The key is read at build time for this client-side gate.
          </p>
        </section>
      </main>
    );
  }

  if (!unlocked) {
    return (
      <main className="operator-shell">
        <nav className="portal-nav">
          <a className="back-link" href="/">
            <ArrowLeft size={15} /> growthengine OS
          </a>
        </nav>
        <section className="operator-gate">
          <Lock size={26} style={{ color: "#c084fc" }} />
          <h1>Operator command center</h1>
          <p>Restricted view. Enter the operator key to open the live lead feed, chaser dispatch, and fleet telemetry.</p>
          <form onSubmit={submitGate}>
            <input
              type="password"
              aria-label="Operator key"
              value={passcode}
              onChange={(e) => {
                setPasscode(e.target.value);
                setGateError(false);
              }}
              placeholder="Operator key"
              autoComplete="off"
            />
            <button type="submit" className="console-action">
              <ShieldCheck size={15} /> Unlock console
            </button>
            {gateError && <p className="gate-error">Invalid operator key.</p>}
          </form>
        </section>
      </main>
    );
  }

  return (
    <main className="operator-shell">
      <header className="operator-header">
        <div>
          <div className="eyebrow">
            <Radio size={12} /> operator command center / live
          </div>
          <h1>GrowthEngine fleet control</h1>
        </div>
        <a className="back-link" href="/portal/onboarding">
          <ArrowLeft size={15} /> back to the cabinet
        </a>
      </header>

      <section className="telemetry-grid">
        {telemetryCards.map(({ key, label, icon: Icon }) => {
          const status = telemetry?.[key] ?? null;
          return (
            <div className="telemetry-card" key={key}>
              <span className="tel-name">
                <Icon size={12} style={{ marginRight: 6, verticalAlign: -2 }} /> {label}
              </span>
              <div className={telStateClass(status?.status ?? "offline")}>
                {(status?.status ?? "offline").replace("_", " ")}
              </div>
              <p className="tel-detail">{status ? status.detail : "API unreachable — start the backend with MONGODB_URI"}</p>
            </div>
          );
        })}
      </section>

      <section className="lead-feed-head">
        <h2>
          Live MongoDB lead feed <span className="mono" style={{ color: "#6f6788", fontSize: 12 }}>
            {leads ? `· ${leads.length} shown` : ""}
          </span>
        </h2>
        <div className="tier-filters">
          {tierTabs.map((t) => (
            <button
              key={t.key}
              type="button"
              className={filter === t.key ? "tier-filter active" : "tier-filter"}
              onClick={() => setFilter(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </section>

      {apiError && (
        <div className="operator-empty">
          <Inbox size={28} />
          <p>
            The lead feed is offline: <span className="mono">{apiError}</span>. Start the FastAPI backend with{" "}
            <span className="mono">MONGODB_URI</span> set and retry.
          </p>
          <button type="button" className="console-action" style={{ marginTop: 14 }} onClick={() => void loadLeads(filter)}>
            <RefreshCw size={15} /> Retry
          </button>
        </div>
      )}

      {!apiError && loading && (
        <div className="operator-empty">
          <RefreshCw size={28} style={{ animation: "spin 1.2s linear infinite" }} />
          <p>Pulling the feed…</p>
        </div>
      )}

      {!apiError && !loading && leads !== null && leads.length === 0 && (
        <div className="operator-empty">
          <Inbox size={28} />
          <p>No leads in this tier yet. New intake submissions land here the moment the gatekeeper routes them.</p>
        </div>
      )}

      {!apiError && !loading && leads !== null && leads.length > 0 && (
        <div className="lead-table">
          {leads.map((lead) => (
            <div className="lead-row" key={lead.id}>
              <div>
                <div className="lead-name">{lead.name}</div>
                <div className="lead-sub">
                  {lead.company_name} · {lead.email}
                </div>
              </div>
              <div className="lead-cell">
                <span className="mono">{lead.phone}</span>
              </div>
              <div className="lead-cell">{bottleneckLabels[lead.primary_bottleneck] ?? lead.primary_bottleneck}</div>
              <div className="lead-cell">
                <span className={tagClass(lead.stack)}>
                  {lead.stack.includes("white_glove")
                    ? "white-glove"
                    : lead.stack.includes("growth")
                      ? "growth pod"
                      : "self-serve"}
                </span>
                <div className="lead-sub" style={{ marginTop: 6 }}>
                  {lead.weekly_budget} / wk
                </div>
              </div>
              <div className="lead-cell">
                {lead.requested_modules.length > 0 ? lead.requested_modules.join(" · ") : "—"}
                {lead.trigger_chaser && (
                  <div className="lead-sub" style={{ color: "var(--lime)" }}>
                    chaser armed
                  </div>
                )}
              </div>
              <div style={{ textAlign: "right" }}>
                {dispatched[lead.id] === "error" ? (
                  <span className="chaser-sent" style={{ color: "#f87171" }}>
                    dispatch failed
                  </span>
                ) : dispatched[lead.id] && dispatched[lead.id] !== "sending" ? (
                  <span className="chaser-sent">
                    <Sparkles size={11} style={{ verticalAlign: -1 }} /> SMS + email sent
                  </span>
                ) : (
                  <button
                    type="button"
                    className="chaser-btn"
                    disabled={dispatched[lead.id] === "sending"}
                    onClick={() => void dispatchChaser(lead)}
                  >
                    <Zap size={13} /> {dispatched[lead.id] === "sending" ? "Dispatching…" : "Dispatch chaser"}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      <p className="operator-note">
        Operator endpoints are bearer-gated server-side via <span className="mono">OPERATOR_ADMIN_KEY</span>; the key
        below must match it. Chaser dispatch marks the lead in MongoDB and is the hook point for the live SMS/email
        sequence.
      </p>
    </main>
  );
}