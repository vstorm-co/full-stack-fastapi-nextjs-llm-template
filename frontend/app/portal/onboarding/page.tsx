"use client";

import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BadgeCheck,
  BarChart3,
  CalendarCheck,
  Clapperboard,
  Factory,
  Mail,
  Printer,
  RefreshCw,
  Send,
  Sparkles,
  Store,
  Wand2,
  X,
  Zap,
} from "lucide-react";

type ModuleId = "echo" | "print" | "trickle" | "local";

const cabinetModules: {
  id: ModuleId;
  icon: typeof Clapperboard;
  name: string;
  tag: string;
  blurb: string;
  points: string[];
}[] = [
  {
    id: "echo",
    icon: Clapperboard,
    name: "Influencer Echo Studio",
    tag: "video engine",
    blurb:
      "Hook generator, video scriptwriter, and automated ad reels deployed through Ayrshare to every major social surface.",
    points: ["8 hook types, 3 archetypes", "50-word offer engine", "One-click social deployment"],
  },
  {
    id: "print",
    icon: Printer,
    name: "Print Studio",
    tag: "print-for-you",
    blurb:
      "Automated vector print collateral: in-store signage, table tents, QR menus, and downloadable PDFs — branded and ready to ship.",
    points: ["Vector PDF generator", "Menu & signage templates", "Local print + ship queue"],
  },
  {
    id: "trickle",
    icon: Mail,
    name: "OmniTrickle Mail Engine",
    tag: "echolink / smart-send",
    blurb:
      "SmartSend drip cadence, n8n-triggered workflows, and gratitude loops that keep every customer in the conversation.",
    points: ["3 / 5 / 7-touch cadences", "Gratitude loop automations", "N8N webhook wiring"],
  },
  {
    id: "local",
    icon: Store,
    name: "OmniLocal Hub",
    tag: "local brand brain",
    blurb:
      "Merchant attribution, POS/KDS sync, and local review defense — the operating brain for brick-and-mortar brands.",
    points: ["Ad spend → foot traffic ROI", "POS / KDS live sync", "Review defense queue"],
  },
];

const echoHooks: Record<string, string[]> = {
  narrative: [
    "They watched your competitor take your market. Here is the 30-second reversal.",
    "Every founder has a villain. Ours is the spreadsheet that eats your team's week.",
    "The lead you lost last quarter — the ad that gets it back in one offer.",
  ],
  humor: [
    "Nobody admits the invoice spreadsheet. The machine just deleted it for them.",
    "Your staff is not slow. Your follow-up process is a rotary phone.",
    "The QR menu no one scans — until the table tent gives them a reason.",
  ],
  vsl: [
    "Why local operators lose 31% of booked revenue to no-shows — and the fix.",
    "The 4 failure points of AI marketing, and the deployment loop that avoids them.",
    "How a $1,250/week pod out-books a $10k retainer agency.",
  ],
};

const printJobs = [
  { label: "Menu — vector PDF", detail: "QR menu · 2-sided · print-ready" },
  { label: "Table tent — double-sided", detail: "4.25×6 in · 14pt cardstock" },
  { label: "Window signage — storefront", detail: "24×36 in · exterior vinyl" },
  { label: "QR standee — countertop", detail: "Acrylic A5 · scannable landing" },
];

const dripCadences: Record<number, string[]> = {
  3: ["Day 0 — welcome + value", "Day 3 — case study", "Day 7 — offer + reply lane"],
  5: ["Day 0 — welcome + value", "Day 2 — proof", "Day 5 — objection flip", "Day 8 — case study", "Day 12 — offer + reply lane"],
  7: [
    "Day 0 — welcome + value",
    "Day 2 — proof",
    "Day 4 — objection flip",
    "Day 6 — social proof",
    "Day 9 — case study",
    "Day 12 — gratitude loop",
    "Day 15 — offer + reply lane",
  ],
};

const localStats = [
  { label: "Attributed foot traffic", value: "+38%" },
  { label: "ROI per ad dollar", value: "4.2×" },
  { label: "Review defense queue", value: "3 open" },
  { label: "POS / KDS last sync", value: "2 min ago" },
];

function EchoConsole() {
  const [archetype, setArchetype] = useState<"narrative" | "humor" | "vsl">("narrative");
  const [hooks, setHooks] = useState<string[] | null>(null);
  const [queued, setQueued] = useState(false);
  return (
    <>
      <div className="console-options">
        {(["narrative", "humor", "vsl"] as const).map((a) => (
          <button
            key={a}
            type="button"
            className={archetype === a ? "console-option selected" : "console-option"}
            onClick={() => {
              setArchetype(a);
              setHooks(null);
              setQueued(false);
            }}
          >
            {a === "narrative" ? "Narrative drama" : a === "humor" ? "Pain-point humor" : "High-intent VSL"}
          </button>
        ))}
      </div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <button type="button" className="console-action" onClick={() => setHooks(echoHooks[archetype])}>
          <Wand2 size={15} /> Generate hooks
        </button>
        <button
          type="button"
          className="console-action"
          disabled={!hooks}
          style={{ opacity: hooks ? 1 : 0.45, cursor: hooks ? "pointer" : "not-allowed" }}
          onClick={() => setQueued(true)}
        >
          <Send size={15} /> Queue to Ayrshare deployment
        </button>
      </div>
      {hooks && (
        <div className="console-output">
          <h4>generated hooks · {archetype}</h4>
          <ul className="out-list">
            {hooks.map((h) => (
              <li key={h}>{h}</li>
            ))}
          </ul>
        </div>
      )}
      {queued && (
        <p className="queued-note">
          <CalendarCheck size={14} /> Deployment queued — scheduled across Instagram, TikTok, YouTube Shorts, and Facebook Reels
        </p>
      )}
    </>
  );
}

function PrintConsole() {
  const [selected, setSelected] = useState<string | null>(null);
  const [queued, setQueued] = useState(false);
  return (
    <>
      <div className="console-options">
        {printJobs.map((j) => (
          <button
            key={j.label}
            type="button"
            className={selected === j.label ? "console-option selected" : "console-option"}
            onClick={() => {
              setSelected(j.label);
              setQueued(false);
            }}
          >
            {j.label}
          </button>
        ))}
      </div>
      <button
        type="button"
        className="console-action"
        disabled={!selected}
        style={{ opacity: selected ? 1 : 0.45, cursor: selected ? "pointer" : "not-allowed" }}
        onClick={() => setQueued(true)}
      >
        <Printer size={15} /> Queue vector PDF
      </button>
      {selected && queued && (
        <div className="console-output">
          <h4>print job queued</h4>
          <div className="sync-row">
            <span>{selected}</span>
            <span className="mono" style={{ color: "var(--lime)" }}>
              render → proof → ship
            </span>
          </div>
          <p className="queued-note">
            <RefreshCw size={14} /> Vector PDF rendering — branded with your menu, logo, and QR landing
          </p>
        </div>
      )}
    </>
  );
}

function TrickleConsole() {
  const [cadence, setCadence] = useState<number>(3);
  return (
    <>
      <div className="console-options">
        {[3, 5, 7].map((n) => (
          <button
            key={n}
            type="button"
            className={cadence === n ? "console-option selected" : "console-option"}
            onClick={() => setCadence(n)}
          >
            {n}-touch cadence
          </button>
        ))}
      </div>
      <button type="button" className="console-action">
        <Zap size={15} /> SmartSend picker
      </button>
      <div className="console-output">
        <h4>drip schedule · gratitude loop enabled</h4>
        <div className="drip-timeline">
          {dripCadences[cadence].map((step) => (
            <div className="drip-touch" key={step}>
              <Mail size={13} /> {step}
            </div>
          ))}
        </div>
        <p className="queued-note">
          <Sparkles size={14} /> N8N webhook wired — replies re-enter the loop, no human babysitting
        </p>
      </div>
    </>
  );
}

function LocalConsole() {
  const [syncedAt, setSyncedAt] = useState("2 min ago");
  return (
    <>
      <div className="local-stats">
        {localStats.map((s) => (
          <div className="local-stat" key={s.label}>
            <strong>{s.value}</strong>
            <span>{s.label}</span>
          </div>
        ))}
      </div>
      <button type="button" className="console-action" onClick={() => setSyncedAt("just now")}>
        <RefreshCw size={15} /> Sync POS / KDS now
      </button>
      <div className="sync-row">
        <span>Last merchant sync</span>
        <span className="mono" style={{ color: "var(--lime)" }}>
          {syncedAt}
        </span>
      </div>
      <p className="queued-note">
        <BarChart3 size={14} /> Attribution feed live — ad spend mapped to foot traffic and ticket size
      </p>
    </>
  );
}

function ModuleConsole({ module }: { module: ModuleId }) {
  if (module === "echo") return <EchoConsole />;
  if (module === "print") return <PrintConsole />;
  if (module === "trickle") return <TrickleConsole />;
  return <LocalConsole />;
}

export default function OnboardingPortal() {
  const [openModule, setOpenModule] = useState<ModuleId | null>(null);
  const active = cabinetModules.find((m) => m.id === openModule) ?? null;

  return (
    <main className="portal-shell">
      <nav className="portal-nav">
        <a className="back-link" href="/">
          <ArrowLeft size={15} /> growthengine OS
        </a>
        <a className="back-link mono" href="/admin/operator" style={{ font: "11px 'DM Mono', monospace" }}>
          operator console →
        </a>
      </nav>

      <section className="upgrade-banner">
        <div>
          <h2>Upgrade to White-Glove — $4k/mo retainer</h2>
          <p>
            Let our firm run every one of these four engines for you end-to-end: strategy, creative, deployment, follow-up,
            and reporting. You approve budgets; the fleet does the rest.
          </p>
        </div>
        <a className="upgrade-cta" href="/booking/vip-fast-track">
          Fast-track the full fleet <ArrowRight size={15} />
        </a>
      </section>

      <section className="cabinet-heading">
        <div className="eyebrow">
          <Factory size={12} /> self-serve suite / the cabinet
        </div>
        <h1>
          Your growth engines,
          <br />
          <span className="mono" style={{ color: "#c084fc" }}>
            one cabinet.
          </span>
        </h1>
        <p>
          Every subsystem we run for our white-glove clients is available here as a self-serve tool at $300/mo each. Pick an
          engine, launch its console, and it connects back to the same intake, chaser, and analytics loop the firm uses.
        </p>
      </section>

      <section className="cabinet-grid">
        {cabinetModules.map(({ id, icon: Icon, name, tag, blurb, points }) => (
          <article className="cabinet-card" key={id}>
            <div className="cabinet-icon">
              <Icon size={20} />
            </div>
            <span className="cabinet-status">operational</span>
            <h3>{name}</h3>
            <span className="cabinet-tag">{tag}</span>
            <p>{blurb}</p>
            <ul>
              {points.map((p) => (
                <li key={p}>
                  <BadgeCheck size={13} /> {p}
                </li>
              ))}
            </ul>
            <button type="button" className="cabinet-launch" onClick={() => setOpenModule(openModule === id ? null : id)}>
              {openModule === id ? "Close console" : "Launch console"} <ArrowRight size={14} />
            </button>
          </article>
        ))}
      </section>

      {active && (
        <section className="cabinet-console">
          <div className="console-head">
            <h3>
              {active.name} <span className="mono" style={{ color: "#c084fc", fontSize: 12 }}>
                / {active.tag}
              </span>
            </h3>
            <button type="button" className="console-close" onClick={() => setOpenModule(null)}>
              <X size={13} /> close
            </button>
          </div>
          <ModuleConsole module={active.id} />
        </section>
      )}

      <footer className="landing-footer" style={{ marginTop: 70 }}>
        <span>growthengine OS</span>
        <span>Cabinet Suite · self-serve fleet</span>
        <span className="mono">$300/mo per engine · white-glove from $4k/mo</span>
      </footer>
    </main>
  );
}