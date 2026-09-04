"use client";

import type { FormEvent } from "react";
import { useEffect, useState } from "react";
import {
  ArrowRight,
  BadgeCheck,
  BookOpen,
  CalendarCheck,
  CircleDollarSign,
  Clock,
  Factory,
  FileVideo,
  Flame,
  Laugh,
  Mail,
  MessageSquareText,
  Phone,
  Radar,
  ShieldCheck,
  Sparkles,
  Target,
  X,
  Zap,
} from "lucide-react";

type WeeklyBudgetTier = "under_250" | "250_to_1250" | "over_1250";

type PrimaryBottleneck =
  | "lead_generation"
  | "operational_overhead"
  | "local_foot_traffic"
  | "retention_and_churn"
  | "full_operating_stack";

type RequestedModule = "video_ads" | "print_collateral" | "email_retention" | "local_pos_attribution";

type IntakePayload = {
  name: string;
  email: string;
  phone: string;
  company_name: string;
  primary_bottleneck: PrimaryBottleneck;
  weekly_budget: WeeklyBudgetTier;
  requested_modules: RequestedModule[];
};

type IntakeResponse = {
  lead_id: string;
  tier: WeeklyBudgetTier;
  routing_action: "self_serve_redirect" | "book_call" | "vip_fast_track";
  target_url: string | null;
  provisioned_subsystem: string | null;
  assigned_agent: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const pillars = [
  {
    index: "01",
    title: "Offer + Ads",
    tagline: "The ad does the finding.",
    body: "Ten hooks per package, eight hook types, one offer written in fifty words. Every campaign targets a different lead.",
    icon: Target,
  },
  {
    index: "02",
    title: "Funnel",
    tagline: "The funnel does the sorting.",
    body: "Three qualifying questions on the intake form tell your team who to speak with first. The rest never touches a human calendar.",
    icon: Radar,
  },
  {
    index: "03",
    title: "AI Follow-Up",
    tagline: "AI does the chasing.",
    body: "A lead that fills a form at midnight gets answered at midnight. Stateful follow-up keeps going until the slot is confirmed.",
    icon: Zap,
  },
];

const creativeFormats = [
  {
    id: "narrative",
    icon: Flame,
    label: "Hyper-Targeted Narrative",
    archetype: "Retaliation drama",
    logline: "A founder watches a rival eat their market — then flips the story in 30 seconds.",
    beats: ["Cold open on the loss", "The moment of realization", "The reversal", "One CTA, fifty words"],
    stats: ["Best for cold B2B", "8–15s cutdown", "Hook type: villain origin"],
  },
  {
    id: "humor",
    icon: Laugh,
    label: "Pain-Point Humor",
    archetype: "Workplace comedy",
    logline: "The spreadsheet nobody admits they maintain — until the machine takes it over.",
    beats: ["Relatable chaos", "Exaggerated stakes", "Product as punchline", "CTA lands on the laugh"],
    stats: ["Best for mid-funnel", "15–30s cutdown", "Hook type: pattern break"],
  },
  {
    id: "vsl",
    icon: FileVideo,
    label: "High-Intent VSL",
    archetype: "Long-form education",
    logline: "A structured teardown that earns the calendar before it ever asks for it.",
    beats: ["Market diagnosis", "Cost of inaction", "The mechanism", "Proof stack + booking"],
    stats: ["Best for hot leads", "4–9 min runtime", "Hook type: authority claim"],
  },
];

const metrics = [
  { label: "hooks per package", value: "10" },
  { label: "hook types tested", value: "8" },
  { label: "questions to route you", value: "3" },
];

const proofCases = [
  {
    industry: "Financial services",
    result: "$30 → $1.50 per lead",
    detail:
      "Equipment financing is a bidding war. Instead of outspending incumbents, batches of hooks were tested against each other until winners emerged, then scaled hard.",
    stat: "−95% cost per lead",
  },
  {
    industry: "Luxury yacht brokerage",
    result: "Months of dead leads → booked viewings",
    detail:
      "Fractional-ownership buyers need trust before they talk. Educational long-form creative re-qualified a stale list without a single cold call.",
    stat: "Stale list revived",
  },
  {
    industry: "Payment processing",
    result: "Skeptical CFOs → signed audits",
    detail:
      "A savings audit only works if the owner opens it. Pain-point humor got the pitch past the gatekeeper's spam filter and into the calendar.",
    stat: "Gatekeepers passed",
  },
];

const comparison = [
  [
    "Speed-to-Lead Response",
    "3 to 24-hour lag. Leads sit in email inboxes or CRM spreadsheets while sales reps manually dial. (400% drop in close rates).",
    "Sub-60-second autonomous engagement. Stateful voice & two-way SMS engages 24/7/365 while the buyer is still on-site.",
  ],
  [
    "Fulfillment Engine",
    "The Junior Coordinator Mill. Pitched by senior founders, then handed off to a 23-year-old managing 30–40 accounts simultaneously.",
    "Zero human bottleneck. Direct programmatic API logic with stateful conversation loops. No sick days, no account churn.",
  ],
  [
    "The Software Reality",
    "Markup on Rented SaaS. Dashboard wrappers over off-the-shelf tools (Looker Studio, Zapier, GHL) marked up 400–500%.",
    "100% Owned Fleet. Native AI video reels, vector print collateral, automated SmartSend sequences, and direct database attribution.",
  ],
  [
    "Commercial Alignment",
    "The Spend-Markup Trap. Agencies charge 15–20% on top of your ad spend, financially incentivizing them to waste your budget.",
    "0% Ad Markup. Flat transparent pod pricing. You pay ad platforms directly. Strategy is anchored strictly on qualified pipeline.",
  ],
  [
    "Asset Ownership",
    "Rented & Ephemeral. If you cancel the retainer, your automations, dashboards, and reporting pipelines evaporate.",
    "Permanent Capital Asset. You own your code, your database, your workflows, and your customer records forever.",
  ],
];

const pricingTiers = [
  {
    tier: "under_250",
    name: "Self-Serve Suite",
    range: "< $250 / week ad spend",
    points: ["Automated qualifying funnel", "Hook library + creative templates", "AI follow-up on managed channels", "Upgrade path to Growth Pod"],
  },
  {
    tier: "250_to_1250",
    name: "Core Growth Pod",
    range: "$250 – $1,250 / week ad spend",
    points: ["Everything in Self-Serve", "10 hooks × 8 types, refreshed monthly", "AI chaser on SMS + email", "Weekly deployment reports"],
  },
  {
    tier: "over_1250",
    name: "White-Glove Pod",
    range: "$1,250+ / week ad spend",
    points: ["Everything in Growth Pod", "VIP fast-track calendar + strategist", "Voice-agent speed-to-lead", "Executive reporting + spend approval console"],
  },
];

const industries = [
  "Financial services",
  "Equipment financing",
  "Yacht brokerage",
  "Payment processing",
  "Roofing & home services",
  "Medical practices",
  "Legal services",
  "SaaS & dev tools",
  "Franchises",
  "Real estate",
];

const faqs = [
  {
    q: "How is GrowthEngine priced?",
    a: "Pods are calibrated on weekly ad spend, not monthly retainers. Under $250/week runs the self-serve suite, $250–$1,250/week enters the Core Growth Pod, and $1,250+/week gets the white-glove pipeline. Agency fees and ad spend are always separate line items — never bundled or marked up.",
  },
  {
    q: "Does GrowthEngine guarantee a number of leads?",
    a: "No responsible agency can. Leads are a function of market, budget, and creative. What we guarantee is process: ten hooks tested against each other, every deployment followed by a measured report, and follow-up that runs until booked or disqualified.",
  },
  {
    q: "What does the AI actually do in the workflow?",
    a: "Specific things: generating and batching hook variants across three archetypes, running the 3-question qualifying funnel, holding SMS/email conversations with full thread state, and reading funnel metrics to queue the next experiment. Every external send or spend passes an approval gate.",
  },
  {
    q: "How fast does follow-up start?",
    a: "A form fill at midnight gets answered at midnight — zero latency on SMS and email in parallel. The chaser keeps state across objections and reschedules until the slot is confirmed or the lead is explicitly disqualified.",
  },
];

const fieldNotes = [
  { title: "Why AI voice agents and video ads work together", note: "Video earns the click. Voice earns the sale. One system, not two vendors." },
  { title: "AI marketing for local businesses", note: "How regional operators win their market before the franchises do." },
  { title: "Why AI marketing campaigns fail", note: "The four failure points we see — and the deployment loop that prevents them." },
];

const bottleneckOptions: { value: PrimaryBottleneck; label: string; hint: string }[] = [
  { value: "lead_generation", label: "Lead generation", hint: "A predictable stream of new prospects" },
  { value: "operational_overhead", label: "Operational overhead", hint: "Manual busywork between lead and sale" },
  { value: "local_foot_traffic", label: "Local foot traffic", hint: "More people through the door" },
  { value: "retention_and_churn", label: "Retention & churn", hint: "Customers we win, then lose too fast" },
  { value: "full_operating_stack", label: "Full operating stack", hint: "Run the whole growth engine end-to-end" },
];

const moduleOptions: { value: RequestedModule; label: string; hint: string }[] = [
  { value: "video_ads", label: "Video ads & reels", hint: "Influencer Echo Studio" },
  { value: "print_collateral", label: "Print collateral", hint: "Print-For-You Studio" },
  { value: "email_retention", label: "Email retention", hint: "OmniTrickle Mail Engine" },
  { value: "local_pos_attribution", label: "Local POS attribution", hint: "OmniLocal Hub" },
];

const routingLabels: Record<IntakeResponse["routing_action"], string> = {
  self_serve_redirect: "Self-serve suite",
  book_call: "Growth Pod calendar",
  vip_fast_track: "Executive fast-track",
};

function SpeedToLeadDemo() {
  const [phone, setPhone] = useState("");
  const [seconds, setSeconds] = useState<number | null>(null);
  const [status, setStatus] = useState<"idle" | "sending" | "running" | "done" | "error">("idle");

  useEffect(() => {
    if (seconds === null || seconds <= 0) {
      if (seconds === 0) setStatus("done");
      return;
    }
    const timer = window.setInterval(() => setSeconds((value) => (value === null ? null : value - 1)), 1000);
    return () => window.clearInterval(timer);
  }, [seconds]);

  async function triggerDemo(event: FormEvent) {
    event.preventDefault();
    if (phone.trim().length < 7 || status === "sending" || status === "running") return;
    setStatus("sending");
    try {
      const response = await fetch(`${API_BASE}/api/leads/speed-to-lead-demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone }),
      });
      if (!response.ok) throw new Error(`demo failed: ${response.status}`);
      setSeconds(59);
      setStatus("running");
    } catch {
      setStatus("error");
    }
  }

  return (
    <div className="speed-demo">
      <div className="speed-demo-copy">
        <div className="eyebrow"><Phone size={12} /> the 60-second proof</div>
        <h2>Feel the response before you buy the system.</h2>
        <p>
          Enter a mobile number and start a timed qualification simulation. The API records the verification request and
          shows the response window; live SMS delivery stays disabled until a provider and consent are configured.
        </p>
      </div>
      <form className="speed-demo-form" onSubmit={triggerDemo}>
        <label htmlFor="speed-demo-phone">Mobile number</label>
        <div className="speed-demo-input-row">
          <input
            id="speed-demo-phone"
            type="tel"
            value={phone}
            onChange={(event) => {
              setPhone(event.target.value);
              if (status === "error") setStatus("idle");
            }}
            placeholder="+1 555 010 2030"
            autoComplete="tel"
          />
          <button type="submit" className="speed-demo-button" disabled={phone.trim().length < 7 || status === "sending" || status === "running"}>
            {status === "sending" ? "Starting…" : status === "running" ? "Running" : "Trigger demo"}
            <Zap size={15} />
          </button>
        </div>
        <div className="speed-demo-status" aria-live="polite">
          {status === "running" && <><strong>00:{String(seconds).padStart(2, "0")}</strong> qualification window active</>}
          {status === "done" && <><BadgeCheck size={15} /> Demo window complete — the response event is recorded.</>}
          {status === "error" && "The verification service is unavailable. Try again shortly."}
          {status === "idle" && "No message is sent until you explicitly opt in to a live provider."}
        </div>
      </form>
    </div>
  );
}

function IntakeWidget() {
  const [step, setStep] = useState(0);
  const [bottleneck, setBottleneck] = useState<PrimaryBottleneck | null>(null);
  const [budget, setBudget] = useState<WeeklyBudgetTier | null>(null);
  const [modules, setModules] = useState<RequestedModule[]>([]);
  const [contact, setContact] = useState({ name: "", email: "", phone: "", company_name: "" });
  const [status, setStatus] = useState<"idle" | "sending" | "done" | "error">("idle");
  const [verdict, setVerdict] = useState<IntakeResponse | null>(null);

  const canAdvance =
    (step === 0 && bottleneck !== null) ||
    (step === 1 && budget !== null) ||
    (step === 2 && contact.name.trim() && /\S+@\S+\.\S+/.test(contact.email) && contact.phone.trim().length >= 7 && contact.company_name.trim());

  function toggleModule(module: RequestedModule) {
    setModules((prev) => (prev.includes(module) ? prev.filter((m) => m !== module) : [...prev, module]));
  }

  async function submit() {
    if (bottleneck === null || budget === null) return;
    setStatus("sending");
    const payload: IntakePayload = {
      ...contact,
      primary_bottleneck: bottleneck,
      weekly_budget: budget,
      requested_modules: modules,
    };
    try {
      const res = await fetch(`${API_BASE}/api/leads/intake`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`intake failed: ${res.status}`);
      const data: IntakeResponse = await res.json();
      setVerdict(data);
      setStatus("done");
      if (data.target_url) window.location.href = data.target_url;
    } catch {
      setStatus("error");
    }
  }

  if (status === "done" && verdict) {
    const priorityMessage: Record<PrimaryBottleneck, string> = {
      lead_generation: "We flagged your priority as Lead Generation Recovery. Your pod will build and test fresh acquisition angles.",
      operational_overhead: "We flagged your priority as Speed-to-Lead Remediation. Your pod will configure sub-60-second autonomous booking during your sprint.",
      local_foot_traffic: "We flagged your priority as Local Demand Capture. Your pod will connect creative, attribution, and the local operating loop.",
      retention_and_churn: "We flagged your priority as Retention Recovery. Your pod will configure stateful follow-up and customer gratitude loops.",
      full_operating_stack: "We flagged your priority as Full-Fleet Orchestration. Your white-glove pod will coordinate every engine end-to-end.",
    };
    return (
      <div className="intake-done">
        <BadgeCheck size={30} />
        <h3>Verified and routed.</h3>
        <p>{priorityMessage[bottleneck ?? "lead_generation"]}</p>
        <p>
          Tier <b>{verdict.tier}</b> · {routingLabels[verdict.routing_action]}
          {verdict.provisioned_subsystem ? <> · subsystem <b>{verdict.provisioned_subsystem}</b></> : null}
        </p>
        <span>Lead {verdict.lead_id} is on the board.</span>
        {verdict.target_url && (
          <a className="intake-primary intake-route-link" href={verdict.target_url}>
            Continue to your route <ArrowRight size={15} />
          </a>
        )}
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="intake-done error">
        <Clock size={30} />
        <h3>Routing service unreachable.</h3>
        <p>Your answers are intact. The intake API did not respond — retry shortly or email hello@growthengine.local and a strategist will triage manually.</p>
        <button className="intake-back" onClick={() => setStatus("idle")}>
          Retry intake
        </button>
      </div>
    );
  }

  return (
    <div className="intake-widget">
      <div className="intake-steps">
        {["Bottleneck", "Weekly spend", "Modules & contact"].map((label, i) => (
          <span key={label} className={i === step ? "active" : i < step ? "past" : ""}>
            <i>{i < step ? "✓" : `0${i + 1}`}</i>
            {label}
          </span>
        ))}
      </div>

      {step === 0 && (
        <div className="intake-step">
          <h3>What is the primary bottleneck in your pipeline right now?</h3>
          <div className="tier-grid">
            {bottleneckOptions.map((option) => (
              <button
                key={option.value}
                className={bottleneck === option.value ? "tier-card selected" : "tier-card"}
                onClick={() => setBottleneck(option.value)}
                type="button"
              >
                <strong>{option.label}</strong>
                <span>{option.hint}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {step === 1 && (
        <div className="intake-step">
          <h3>What is your weekly ad spend?</h3>
          <p className="intake-hint">This calibrates your pod. Nothing is charged by GrowthEngine at intake.</p>
          <div className="tier-grid">
            {pricingTiers.map((tier) => (
              <button
                key={tier.tier}
                className={budget === tier.tier ? "tier-card selected" : "tier-card"}
                onClick={() => setBudget(tier.tier as WeeklyBudgetTier)}
                type="button"
              >
                <strong>{tier.range}</strong>
                <span>{tier.name}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="intake-step">
          <h3>Which fleet engines do you want provisioned — and where do we reach you?</h3>
          <p className="intake-hint">Pick any that apply. Every engine below is ours — no third-party markup.</p>
          <div className="module-grid">
            {moduleOptions.map((option) => (
              <button
                key={option.value}
                className={modules.includes(option.value) ? "module-chip selected" : "module-chip"}
                onClick={() => toggleModule(option.value)}
                type="button"
              >
                {option.label}
                <span>{option.hint}</span>
              </button>
            ))}
          </div>
          <div className="contact-grid">
            <input aria-label="Full name" value={contact.name} onChange={(e) => setContact({ ...contact, name: e.target.value })} placeholder="Full name" autoComplete="name" />
            <input aria-label="Work email" value={contact.email} onChange={(e) => setContact({ ...contact, email: e.target.value })} placeholder="Work email" type="email" autoComplete="email" />
            <input aria-label="Phone" value={contact.phone} onChange={(e) => setContact({ ...contact, phone: e.target.value })} placeholder="Phone" type="tel" autoComplete="tel" />
            <input aria-label="Company" value={contact.company_name} onChange={(e) => setContact({ ...contact, company_name: e.target.value })} placeholder="Company" autoComplete="organization" />
          </div>
        </div>
      )}

      <div className="intake-actions">
        {step > 0 && (
          <button className="intake-ghost" onClick={() => setStep(step - 1)} type="button">
            Back
          </button>
        )}
        {step < 2 ? (
          <button className="intake-primary" disabled={!canAdvance} onClick={() => setStep(step + 1)} type="button">
            Continue <ArrowRight size={15} />
          </button>
        ) : (
          <button className="intake-primary" disabled={!canAdvance || status === "sending"} onClick={submit} type="button">
            {status === "sending" ? "Routing…" : "Run the gatekeeper"}
          </button>
        )}
      </div>
      <span className="intake-legal">Consent-aware follow-up · opt out anytime · no spend without approval</span>
    </div>
  );
}

export default function Home() {
  const [activeFormat, setActiveFormat] = useState(creativeFormats[0].id);
  const format = creativeFormats.find((f) => f.id === activeFormat) ?? creativeFormats[0];

  return (
    <main className="landing-shell">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "FAQPage",
            mainEntity: faqs.map((f) => ({
              "@type": "Question",
              name: f.q,
              acceptedAnswer: { "@type": "Answer", text: f.a },
            })),
          }),
        }}
      />

      <nav className="landing-nav">
        <a className="brand" href="#top">
          <span className="brand-mark">
            <Radar size={20} />
          </span>
          growthengine <span className="brand-muted">OS</span>
        </a>
        <div className="nav-links">
          <a href="#pillars">The engine</a>
          <a href="#creative">Creative</a>
          <a href="#pricing">Pricing</a>
          <a href="#faq">FAQ</a>
          <a href="/portal/onboarding">Portal</a>
        </div>
        <a className="nav-cta" href="#intake">
          Book the pod <ArrowRight size={15} />
        </a>
      </nav>

      <section className="landing-hero" id="top">
        <div className="hero-copy">
          <div className="eyebrow">
            <CircleDollarSign size={12} /> appointment infrastructure / always on
          </div>
          <h1>
            AI marketing firm filling <em>calendars</em> for high-performing sales teams.
          </h1>
          <p className="hero-lede">
            Three engines run as one firm: creative that finds the lead, a gatekeeper that sorts it, and an AI chaser that
            holds the conversation until your calendar fills. No retainers for busywork — pods calibrated on weekly ad
            spend.
          </p>
          <div className="hero-actions">
            <a className="hero-button" href="#intake">
              Run the 3-question gate <ArrowRight size={17} />
            </a>
            <a className="quiet-button" href="#pillars">
              See the engine
            </a>
          </div>
          <div className="trust-line">
            <BadgeCheck size={15} /> Zero-latency follow-up <span>·</span> Stateful until booked <span>·</span> Approved budgets only
          </div>
        </div>
        <div className="hero-visual">
          <div className="glass-panel chase-panel">
            <div className="panel-head">
              <span className="mono">chaser / live session</span>
              <span className="live-dot">ACTIVE</span>
            </div>
            <div className="chase-thread">
              <div className="chase-msg inbound">
                <span className="chase-meta">
                  <Mail size={12} /> 12:04 AM · inbound form fill
                </span>
                <p>&ldquo;Saw the ad — can you take over our outbound?&rdquo;</p>
              </div>
              <div className="chase-msg outbound">
                <span className="chase-meta">
                  <Zap size={12} /> 12:04 AM · AI chaser
                </span>
                <p>Qualified on bottleneck, offered two slots, held the thread.</p>
              </div>
              <div className="chase-msg inbound">
                <span className="chase-meta">
                  <MessageSquareText size={12} /> 12:19 AM · SMS reply
                </span>
                <p>&ldquo;Thursday works.&rdquo;</p>
              </div>
              <div className="chase-booked">
                <CalendarCheck size={15} /> Appointment confirmed 12:21 AM — calendar updated
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="metrics-strip">
        {metrics.map((m) => (
          <div key={m.label}>
            <strong>{m.value}</strong>
            <span>{m.label}</span>
          </div>
        ))}
        <div className="proof-note">
          Deployed across finance, brokerage, home services, medical, legal, and SaaS.
          <span> Calibrated on weekly ad spend, not monthly retainers.</span>
        </div>
      </section>

      <section className="section-block" id="proof">
        <div className="section-heading">
          <div>
            <div className="eyebrow">deployment reports</div>
            <h2>
              Every campaign ends
              <br />
              <span>in a number.</span>
            </h2>
          </div>
          <p>
            Before/after economics from the playbook the agents run — hook batches tested against each other, winners
            scaled, losers retired.
          </p>
        </div>
        <div className="proof-grid">
          {proofCases.map((c) => (
            <article className="proof-card" key={c.industry}>
              <span className="proof-industry mono">{c.industry}</span>
              <div className="proof-result">{c.result}</div>
              <p>{c.detail}</p>
              <div className="proof-stat">
                <Sparkles size={14} /> {c.stat}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="section-block" id="pillars">
        <div className="section-heading">
          <div>
            <div className="eyebrow">the three-pillar revenue engine</div>
            <h2>
              One firm.
              <br />
              <span>Three engines.</span>
            </h2>
          </div>
          <p>
            Every campaign runs the same institutional playbook: one offer, engineered ads, a funnel that qualifies before
            a human ever looks, and an AI that never lets a lead go cold.
          </p>
        </div>
        <div className="pillar-grid">
          {pillars.map(({ index, title, tagline, body, icon: Icon }) => (
            <article className="pillar-card" key={index}>
              <div className="pillar-top">
                <span className="pillar-index">{index}</span>
                <span className="pillar-icon">
                  <Icon size={19} />
                </span>
              </div>
              <h3>{title}</h3>
              <p className="pillar-tagline">{tagline}</p>
              <p className="pillar-body">{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section-block" id="compare">
        <div className="section-heading">
          <div>
            <div className="eyebrow">architectural breakdown</div>
            <h2>
              Operational truth.
              <br />
              <span>Not feature theater.</span>
            </h2>
          </div>
          <p>
            The business model determines the outcome. This is the operating difference between renting an agency team and
            deploying owned acquisition infrastructure.
          </p>
        </div>
        <div className="compare-table" aria-label="Operational comparison between a legacy agency retainer and GrowthEngine OS">
          <div className="compare-head">
            <span>operational vector</span>
            <span>legacy agency retainer ($3k–$8k/mo)</span>
            <span>growthengine OS (owned infrastructure)</span>
          </div>
          {comparison.map(([label, legacy, growthEngine]) => (
            <div className="compare-row" key={label}>
              <span className="compare-label">{label}</span>
              <span className="compare-theirs">
                <X size={14} /> {legacy}
              </span>
              <span className="compare-ours">
                <BadgeCheck size={14} /> {growthEngine}
              </span>
            </div>
          ))}
        </div>
        <div className="comparison-callout">
          <Sparkles size={20} />
          <p>
            Stop paying high-cost retainers for an agency to learn on your dime. We built the actual tools so you can run on
            owned infrastructure.
          </p>
        </div>
        <a className="teardown-link" href="/vs/agencies">
          Read the full operational agency teardown <ArrowRight size={15} />
        </a>
      </section>

      <section className="section-block" id="speed-demo">
        <SpeedToLeadDemo />
      </section>

      <section className="section-block creative-section" id="creative">
        <div className="section-heading">
          <div>
            <div className="eyebrow">the creative engine</div>
            <h2>
              Ten hooks. Eight types.
              <br />
              <span>One fifty-word offer.</span>
            </h2>
          </div>
          <p>
            Production spans three archetypes so every campaign meets its buyer in a different register. Select a format
            to see how the ad does the finding.
          </p>
        </div>
        <div className="creative-tabs" role="tablist" aria-label="Creative formats">
          {creativeFormats.map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              role="tab"
              aria-selected={activeFormat === id}
              className={activeFormat === id ? "creative-tab active" : "creative-tab"}
              onClick={() => setActiveFormat(id)}
              type="button"
            >
              <Icon size={16} /> {label}
            </button>
          ))}
        </div>
        <div className="creative-stage">
          <div className="stage-copy">
            <span className="stage-archetype mono">{format.archetype}</span>
            <h3>{format.logline}</h3>
            <ol>
              {format.beats.map((beat) => (
                <li key={beat}>{beat}</li>
              ))}
            </ol>
          </div>
          <div className="stage-meta">
            {format.stats.map((stat) => (
              <span key={stat}>
                <Sparkles size={13} /> {stat}
              </span>
            ))}
          </div>
        </div>
      </section>

      <section className="section-block" id="pricing">
        <div className="section-heading">
          <div>
            <div className="eyebrow">transparent brackets</div>
            <h2>
              Priced on weekly ad spend.
              <br />
              <span>Never a black box.</span>
            </h2>
          </div>
          <p>
            Your weekly ad budget sets the pod — agency fees and ad spend are separate line items, always. The gatekeeper
            below routes you to the matching bracket in three questions.
          </p>
        </div>
        <div className="pricing-grid">
          {pricingTiers.map((tier) => (
            <article className={tier.tier === "250_to_1250" ? "pricing-card featured" : "pricing-card"} key={tier.tier}>
              {tier.tier === "250_to_1250" && <span className="pricing-flag">most common</span>}
              <span className="mono pricing-range">{tier.range}</span>
              <h3>{tier.name}</h3>
              <ul>
                {tier.points.map((p) => (
                  <li key={p}>
                    <BadgeCheck size={14} /> {p}
                  </li>
                ))}
              </ul>
              <a className="pricing-link" href="#intake">
                Route me here <ArrowRight size={14} />
              </a>
            </article>
          ))}
        </div>
        <div className="owned-banner">
          <span className="owned-mark">
            <Factory size={22} />
          </span>
          <div>
            <h3>Owned stack. No rented SaaS with a 500% markup.</h3>
            <p>
              We don&apos;t rent third-party SaaS and pass the markup to you. We own the video engine, the print studio, the
              email infrastructure, and the local operating system — every dollar you spend goes into the machine that
              sells for you.
            </p>
          </div>
        </div>
      </section>

      <section className="section-block chaser-section" id="chaser">
        <div className="section-heading">
          <div>
            <div className="eyebrow">the persistent chaser</div>
            <h2>
              Midnight form fill.
              <br />
              <span>Midnight answer.</span>
            </h2>
          </div>
          <p>
            Zero-latency SMS and email follow-up that keeps state. The chaser continues until the slot is confirmed or
            the lead is explicitly disqualified — nothing sits in an inbox purgatory.
          </p>
        </div>
        <div className="chaser-grid">
          <article>
            <Clock size={18} />
            <h3>0s response</h3>
            <p>Every form fill is engaged the moment it lands, on SMS and email in parallel.</p>
          </article>
          <article>
            <MessageSquareText size={18} />
            <h3>Stateful threads</h3>
            <p>Objections, reschedules, and no-shows carry context — the chaser never restarts cold.</p>
          </article>
          <article>
            <CalendarCheck size={18} />
            <h3>Booked or disqualified</h3>
            <p>The loop exits on exactly two outcomes: a confirmed slot, or a logged disqualification.</p>
          </article>
        </div>
      </section>

      <section className="industries-strip">
        <span className="industries-label mono">industries deployed</span>
        <div className="industries-list">
          {industries.map((i) => (
            <span key={i}>{i}</span>
          ))}
        </div>
      </section>

      <section className="intake-section" id="intake">
        <div className="intake-copy">
          <div className="eyebrow">the dynamic gatekeeper</div>
          <h2>
            Three questions.
            <br />
            <span>One routing decision.</span>
          </h2>
          <p>
            The gatekeeper reads your bottleneck, your weekly ad spend, and your timeline — then routes you to the pod
            that matches. Under $250 a week lands in the self-serve suite. Core spend enters the Growth Pod. High-ticket
            spend goes straight to the white-glove fast track.
          </p>
          <ul className="intake-points">
            <li>
              <Phone size={15} /> Straight to the right calendar — no forms into a void
            </li>
            <li>
              <Target size={15} /> Weekly spend calibration, not monthly retainers
            </li>
            <li>
              <ShieldCheck size={15} /> The chaser takes over the second you submit
            </li>
          </ul>
        </div>
        <IntakeWidget />
      </section>

      <section className="section-block" id="faq">
        <div className="section-heading">
          <div>
            <div className="eyebrow">straight answers</div>
            <h2>
              Questions serious buyers
              <br />
              <span>should ask.</span>
            </h2>
          </div>
          <p>
            Pricing models, guarantees, and what the AI actually does — answered before you ever get on a call.
          </p>
        </div>
        <div className="faq-list">
          {faqs.map((f) => (
            <details className="faq-item" key={f.q}>
              <summary>{f.q}</summary>
              <p>{f.a}</p>
            </details>
          ))}
        </div>
      </section>

      <section className="section-block notes-section" id="notes">
        <div className="section-heading">
          <div>
            <div className="eyebrow">field notes</div>
            <h2>
              Playbooks in public.
              <br />
              <span>No secrets, no fluff.</span>
            </h2>
          </div>
          <p>
            The same operating logic our agents run, written out — vertical teardowns, failure analysis, and system
            design.
          </p>
        </div>
        <div className="notes-grid">
          {fieldNotes.map((n) => (
            <article key={n.title}>
              <BookOpen size={16} />
              <h3>{n.title}</h3>
              <p>{n.note}</p>
            </article>
          ))}
        </div>
      </section>

      <footer className="landing-footer">
        <span>growthengine OS</span>
        <span>Appointments, engineered.</span>
        <span className="mono">
          v0.5 / remediation build · <a href="/privacy">privacy</a> · <a href="/terms">terms</a> ·{" "}
          <a href="/admin/operator">operator console</a>
        </span>
      </footer>
    </main>
  );
}
