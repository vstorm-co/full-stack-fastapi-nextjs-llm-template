"use client";

import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BadgeCheck,
  CalendarCheck,
  Clock,
  Globe,
  Radar,
  Rocket,
  ShieldCheck,
  Zap,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type PrimaryBottleneck =
  | "lead_generation"
  | "operational_overhead"
  | "local_foot_traffic"
  | "retention_and_churn";

const bottleneckOptions: { value: PrimaryBottleneck; label: string }[] = [
  { value: "lead_generation", label: "Lead generation" },
  { value: "operational_overhead", label: "Operational overhead" },
  { value: "local_foot_traffic", label: "Local foot traffic" },
  { value: "retention_and_churn", label: "Retention & churn" },
];

const included = [
  "10 hooks × 8 types per package, refreshed monthly across three creative archetypes",
  "AI chaser on SMS + email — zero-latency, stateful until booked or disqualified",
  "Funnel routing before a human ever sees a lead — the rest never hits your calendar",
  "Weekly deployment reports with measured next-best-action",
  "Ad spend billed directly to your ad accounts — never marked up, never spent without approval",
];

const nextSteps = [
  { time: "Same day", text: "Strategist triages your intake and confirms the call slot" },
  { time: "Call", text: "Growth scoping: offer, creative archetypes, funnel, chaser cadence" },
  { time: "Day 1", text: "Growth Pod provisioned — creative batches, funnel, and chaser armed" },
];

type BookingForm = {
  name: string;
  email: string;
  phone: string;
  company_name: string;
  primary_bottleneck: PrimaryBottleneck | null;
  preferred_call_time: string;
  target_start_date: string;
};

type IntakeResponse = {
  lead_id: string;
  tier: string;
  routing_action: string;
  target_url: string | null;
  provisioned_subsystem: string | null;
  assigned_agent: string | null;
};

const emptyForm: BookingForm = {
  name: "",
  email: "",
  phone: "",
  company_name: "",
  primary_bottleneck: null,
  preferred_call_time: "",
  target_start_date: "",
};

const callTimes = ["Morning (8–11am)", "Midday (11am–2pm)", "Afternoon (2–5pm)", "Evening (5–8pm)"];

export default function GrowthPodBooking() {
  const [form, setForm] = useState<BookingForm>(emptyForm);
  const [status, setStatus] = useState<"idle" | "sending" | "done" | "error">("idle");
  const [verdict, setVerdict] = useState<IntakeResponse | null>(null);

  const set = (key: keyof BookingForm) => (value: string) => setForm((prev) => ({ ...prev, [key]: value }));

  const valid =
    form.name.trim().length >= 2 &&
    /\S+@\S+\.\S+/.test(form.email) &&
    form.phone.trim().length >= 7 &&
    form.company_name.trim().length >= 1 &&
    form.primary_bottleneck !== null &&
    form.preferred_call_time !== "" &&
    form.target_start_date !== "";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("sending");
    try {
      const res = await fetch(`${API_BASE}/api/leads/intake`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: form.name,
          email: form.email,
          phone: form.phone,
          company_name: form.company_name,
          // The Growth gate always routes under the mid-tier pod.
          primary_bottleneck: form.primary_bottleneck,
          weekly_budget: "250_to_1250",
          requested_modules: [],
          preferred_call_time: form.preferred_call_time,
          target_start_date: form.target_start_date,
        }),
      });
      if (!res.ok) throw new Error(`intake failed: ${res.status}`);
      const data: IntakeResponse = await res.json();
      setVerdict(data);
      setStatus("done");
    } catch {
      setStatus("error");
    }
  }

  if (status === "done" && verdict) {
    return (
      <main className="portal-shell">
        <nav className="portal-nav">
          <a className="back-link" href="/portal/onboarding">
            <ArrowLeft size={15} /> growthengine OS
          </a>
        </nav>
        <section className="booking-confirm">
          <div className="booking-confirm-mark">
            <BadgeCheck size={30} />
          </div>
          <div className="eyebrow">growth pod / request received</div>
          <h1>Your pod slot is reserved.</h1>
          <p>
            Lead <span className="mono">{verdict.lead_id}</span> was routed to{" "}
            <b>{verdict.provisioned_subsystem ?? "managed_growth_pod"}</b>
            {verdict.assigned_agent ? (
              <>
                {" "}
                and handed to <b>{verdict.assigned_agent}</b>
              </>
            ) : null}
            . The chaser is armed — SMS + email follow-up starts the moment you confirm the call slot. A strategist
            confirms your preferred time within one business day.
          </p>
          <div className="booking-next">
            {nextSteps.map((step) => (
              <div key={step.time}>
                <span className="mono">{step.time}</span>
                <p>{step.text}</p>
              </div>
            ))}
          </div>
          <a className="upgrade-cta" href="/portal/onboarding" style={{ marginTop: 6 }}>
            Back to the Cabinet Suite <ArrowRight size={15} />
          </a>
        </section>
      </main>
    );
  }

  if (status === "error") {
    return (
      <main className="portal-shell">
        <nav className="portal-nav">
          <a className="back-link" href="/portal/onboarding">
            <ArrowLeft size={15} /> growthengine OS
          </a>
        </nav>
        <section className="booking-confirm">
          <div className="booking-confirm-mark" style={{ color: "var(--amber)" }}>
            <Clock size={30} />
          </div>
          <div className="eyebrow">growth pod / routing offline</div>
          <h1>The intake API did not respond.</h1>
          <p>
            Your request was not lost — it simply did not reach the routing engine. Retry, or email{" "}
            <span className="mono">hello@growthengine.local</span> and a strategist will triage you manually.
          </p>
          <button type="button" className="upgrade-cta" style={{ border: "none" }} onClick={() => setStatus("idle")}>
            Retry booking <ArrowRight size={15} />
          </button>
        </section>
      </main>
    );
  }

  return (
    <main className="portal-shell">
      <nav className="portal-nav">
        <a className="back-link" href="/portal/onboarding">
          <ArrowLeft size={15} /> growthengine OS
        </a>
        <a className="back-link mono" href="/booking/vip-fast-track">
          white-glove fast track →
        </a>
      </nav>

      <header className="booking-hero">
        <div className="eyebrow">
          <Rocket size={12} /> growth pod / managed pipeline
        </div>
        <h1>
          Core Growth Pod
          <span className="booking-divider">|</span>
          <em>Scale Infrastructure</em>
        </h1>
        <p>
          Structured multi-channel acquisition for operators spending $250–$1,250/wk on media. One offer, ten tested
          hooks, a funnel that sorts before a human looks, and an AI chaser that holds every lead until the slot is
          booked.
        </p>
      </header>

      <section className="booking-layout">
        <div className="booking-scope">
          <div className="booking-included">
            <h2>What the Growth Pod runs</h2>
            <ul>
              {included.map((item) => (
                <li key={item}>
                  <BadgeCheck size={15} /> {item}
                </li>
              ))}
            </ul>
          </div>
          <div className="booking-included">
            <h2>What happens next</h2>
            <div className="booking-next">
              {nextSteps.map((step) => (
                <div key={step.time}>
                  <span className="mono">{step.time}</span>
                  <p>{step.text}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="booking-trust">
            <span>
              <ShieldCheck size={14} /> Spend approval gates on every channel
            </span>
            <span>
              <Globe size={14} /> Owned stack — no third-party markup
            </span>
            <span>
              <Zap size={14} /> Zero-latency chaser until booked
            </span>
          </div>
        </div>

        <form className="booking-panel" onSubmit={submit}>
          <div className="booking-panel-head">
            <span className="mono">calendar booking / growth pod</span>
            <span className="live-dot">
              <CalendarCheck size={12} /> Growth slots open
            </span>
          </div>
          <div className="booking-availability" aria-hidden="true">
            <Radar size={14} /> Calendar placeholder — connect your Cal.com/Calendly provider to go live
          </div>

          <div className="field">
            <label htmlFor="gk-name">Full name</label>
            <input id="gk-name" value={form.name} onChange={(e) => set("name")(e.target.value)} placeholder="Alex Rivera" autoComplete="name" />
          </div>
          <div className="field">
            <label htmlFor="gk-email">Work email</label>
            <input id="gk-email" type="email" value={form.email} onChange={(e) => set("email")(e.target.value)} placeholder="alex@company.com" autoComplete="email" />
          </div>
          <div className="field">
            <label htmlFor="gk-phone">Phone</label>
            <input id="gk-phone" type="tel" value={form.phone} onChange={(e) => set("phone")(e.target.value)} placeholder="+1 555 010 2030" autoComplete="tel" />
          </div>
          <div className="field">
            <label htmlFor="gk-company">Company</label>
            <input id="gk-company" value={form.company_name} onChange={(e) => set("company_name")(e.target.value)} placeholder="River & Co" autoComplete="organization" />
          </div>

          <div className="field" role="group" aria-labelledby="gk-bottleneck">
            <label id="gk-bottleneck">Primary bottleneck</label>
            <div className="call-time-grid">
              {bottleneckOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  aria-pressed={form.primary_bottleneck === option.value}
                  className={form.primary_bottleneck === option.value ? "call-time selected" : "call-time"}
                  onClick={() => setForm((prev) => ({ ...prev, primary_bottleneck: option.value }))}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div className="field">
            <label htmlFor="gk-call-time">Preferred call time</label>
            <div className="call-time-grid">
              {callTimes.map((time) => (
                <button
                  key={time}
                  type="button"
                  className={form.preferred_call_time === time ? "call-time selected" : "call-time"}
                  onClick={() => set("preferred_call_time")(time)}
                >
                  {time}
                </button>
              ))}
            </div>
          </div>
          <div className="field">
            <label htmlFor="gk-start">Target start date</label>
            <input id="gk-start" type="date" value={form.target_start_date} onChange={(e) => set("target_start_date")(e.target.value)} />
          </div>

          <button type="submit" className="console-action booking-submit" disabled={!valid || status === "sending"}>
            {status === "sending" ? "Routing to the pod…" : "Reserve my Growth Pod slot"}
            <ArrowRight size={15} />
          </button>
          <span className="intake-legal">Consent-aware follow-up · opt out anytime · no spend without approval</span>
        </form>
      </section>

      <footer className="legal-footer">
        <span>growthengine OS · growth pod</span>
        <span>
          <a href="/privacy">Privacy</a> · <a href="/terms">Terms</a> · <a href="/">Home</a>
        </span>
      </footer>
    </main>
  );
}
