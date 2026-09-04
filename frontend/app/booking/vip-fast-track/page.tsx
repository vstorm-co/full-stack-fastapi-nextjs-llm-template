"use client";

import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BadgeCheck,
  CalendarCheck,
  Clock,
  Crown,
  Globe,
  Radar,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const fullFleetModules = [
  "video_ads",
  "print_collateral",
  "email_retention",
  "local_pos_attribution",
];

const included = [
  "Dedicated strategist + full fleet orchestration (Video · Print · Email · POS · Reputation)",
  "Pass-through ad spend — billed directly to your ad accounts, zero markup",
  "VIP fast-track calendar: skip the queue, speak to the firm this week",
  "Weekly deployment reports with measured next-best-action",
];

const nextSteps = [
  { time: "Same day", text: "Strategist triages your intake and confirms the call slot" },
  { time: "Call", text: "White-glove scoping: stack, creative archetypes, budgets, start date" },
  { time: "Day 1", text: "Full fleet provisioned — creative, print, drip, and local engines armed" },
];

type BookingForm = {
  name: string;
  email: string;
  phone: string;
  company_name: string;
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
  preferred_call_time: "",
  target_start_date: "",
};

const callTimes = ["Morning (8–11am)", "Midday (11am–2pm)", "Afternoon (2–5pm)", "Evening (5–8pm)"];

export default function VipFastTrackBooking() {
  const [form, setForm] = useState<BookingForm>(emptyForm);
  const [status, setStatus] = useState<"idle" | "sending" | "done" | "error">("idle");
  const [verdict, setVerdict] = useState<IntakeResponse | null>(null);

  const set = (key: keyof BookingForm) => (value: string) => setForm((prev) => ({ ...prev, [key]: value }));

  const valid =
    form.name.trim().length >= 2 &&
    /\S+@\S+\.\S+/.test(form.email) &&
    form.phone.trim().length >= 7 &&
    form.company_name.trim().length >= 1 &&
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
          ...form,
          // The fast-track gate always routes under the white-glove pod.
          primary_bottleneck: "full_operating_stack",
          weekly_budget: "over_1250",
          requested_modules: fullFleetModules,
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
          <a className="back-link mono" href="/admin/operator" style={{ font: "11px 'DM Mono', monospace" }}>
            operator console →
          </a>
        </nav>
        <section className="booking-confirm">
          <div className="booking-confirm-mark">
            <BadgeCheck size={30} />
          </div>
          <div className="eyebrow">white-glove pod / request received</div>
          <h1>You are on the fast track.</h1>
          <p>
            Lead <span className="mono">{verdict.lead_id}</span> was routed to{" "}
            <b>{verdict.provisioned_subsystem ?? "full_fleet_white_glove"}</b>
            {verdict.assigned_agent ? (
              <>
                {" "}
                and handed to <b>{verdict.assigned_agent}</b>
              </>
            ) : null}
            . A strategist confirms your preferred call time within one business day — the chaser keeps the thread warm
            until the slot is locked.
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
          <div className="eyebrow">white-glove pod / routing offline</div>
          <h1>The intake API did not respond.</h1>
          <p>
            Your request was not lost — it simply did not reach the routing engine. Retry, or email{" "}
            <span className="mono">hello@growthengine.local</span> and a strategist will fast-track you manually.
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
        <a className="back-link mono" href="/admin/operator" style={{ font: "11px 'DM Mono', monospace" }}>
          operator console →
        </a>
      </nav>

      <header className="booking-hero">
        <div className="eyebrow">
          <Crown size={12} /> white-glove pod / executive route
        </div>
        <h1>
          Fast-Track Executive Onboarding
          <span className="booking-divider">|</span>
          <em>White-Glove Pod</em>
        </h1>
        <p>
          Full fleet orchestration ($4k/mo retainer). Pass-through ad spend is billed directly to your ad accounts with
          zero markup. One intake, four engines armed, one strategist who owns your outcome.
        </p>
      </header>

      <section className="booking-layout">
        <div className="booking-scope">
          <div className="booking-included">
            <h2>What the retainer runs</h2>
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
            <span className="mono">calendar booking / white-glove</span>
            <span className="live-dot">
              <CalendarCheck size={12} /> VIP slots open
            </span>
          </div>
          <div className="booking-availability" aria-hidden="true">
            <Radar size={14} /> Calendar placeholder — connect your Cal.com/Calendly provider to go live
          </div>

          <div className="field">
            <label htmlFor="bk-name">Full name</label>
            <input id="bk-name" value={form.name} onChange={(e) => set("name")(e.target.value)} placeholder="Alex Rivera" autoComplete="name" />
          </div>
          <div className="field">
            <label htmlFor="bk-email">Work email</label>
            <input id="bk-email" type="email" value={form.email} onChange={(e) => set("email")(e.target.value)} placeholder="alex@company.com" autoComplete="email" />
          </div>
          <div className="field">
            <label htmlFor="bk-phone">Phone</label>
            <input id="bk-phone" type="tel" value={form.phone} onChange={(e) => set("phone")(e.target.value)} placeholder="+1 555 010 2030" autoComplete="tel" />
          </div>
          <div className="field">
            <label htmlFor="bk-company">Company</label>
            <input id="bk-company" value={form.company_name} onChange={(e) => set("company_name")(e.target.value)} placeholder="River & Co" autoComplete="organization" />
          </div>
          <div className="field">
            <label htmlFor="bk-call-time">Preferred call time</label>
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
            <label htmlFor="bk-start">Target start date</label>
            <input id="bk-start" type="date" value={form.target_start_date} onChange={(e) => set("target_start_date")(e.target.value)} />
          </div>

          <button type="submit" className="console-action booking-submit" disabled={!valid || status === "sending"}>
            {status === "sending" ? "Routing to the pod…" : "Request fast-track onboarding"}
            <ArrowRight size={15} />
          </button>
          <span className="intake-legal">Consent-aware follow-up · opt out anytime · no spend without approval</span>
        </form>
      </section>

      <footer className="legal-footer">
        <span>growthengine OS · white-glove pod</span>
        <span>
          <a href="/privacy">Privacy</a> · <a href="/terms">Terms</a> · <a href="/">Home</a>
        </span>
      </footer>
    </main>
  );
}