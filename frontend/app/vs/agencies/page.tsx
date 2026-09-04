import { ArrowLeft, ArrowRight, BadgeCheck, Building2, CircleDollarSign, FileSearch, ShieldCheck, X } from "lucide-react";

const battlecards = [
  {
    id: "klientboost",
    name: "KlientBoost",
    context: "Performance creative and paid media retainer",
    legacy: [
      "Paid media optimization still depends on account-manager handoffs and scheduled reporting cycles.",
      "Creative testing is packaged as a service layer, so the operating system remains rented from the agency.",
      "The client carries the cost of learning while media and retainer economics stay separate from qualified pipeline.",
    ],
    growthengine: [
      "Sub-60-second stateful SMS and voice qualification keeps a new lead moving while intent is highest.",
      "Owned creative, funnel, and attribution services keep experiments connected from hook to booked call.",
      "Flat pod pricing and direct platform billing align optimization to qualified appointments, not media volume.",
    ],
  },
  {
    id: "webfx",
    name: "WebFX",
    context: "Large full-service digital marketing organization",
    legacy: [
      "A broad service catalog creates more coordination points between strategy, production, media, and sales.",
      "Leads can wait for a human response after the form submission, especially outside business hours.",
      "Dashboards and workflow configuration are typically part of the ongoing relationship rather than a permanent asset.",
    ],
    growthengine: [
      "The gatekeeper routes by bottleneck and weekly spend before a human calendar is touched.",
      "The chaser owns the thread across follow-up states until booked or explicitly disqualified.",
      "Your codebase, database, workflows, and customer records remain a capital asset you control.",
    ],
  },
  {
    id: "disruptive-hawke",
    name: "Disruptive / Hawke",
    context: "Specialist growth and commerce agency models",
    legacy: [
      "Specialist expertise can still be delivered through junior account layers managing many brands at once.",
      "Ad-spend percentage models may reward more media activity even when the qualified pipeline is flat.",
      "Campaign assets, automations, and reporting often depend on an active retainer relationship.",
    ],
    growthengine: [
      "Direct programmatic workflows remove the coordinator bottleneck from repetitive fulfillment and follow-up.",
      "Zero media markup with transparent pod pricing makes budget approval and performance review explicit.",
      "Native video, vector print, SmartSend, and local attribution are a connected owned fleet, not dashboard wrappers.",
    ],
  },
];

const vectors = [
  ["Speed-to-lead", "3–24 hour response windows", "Sub-60-second engagement"],
  ["Fulfillment", "Coordinator handoffs", "Programmatic stateful loops"],
  ["Commercials", "Retainer + media markup", "Flat pod + direct media billing"],
  ["Assets", "Rented workflows", "Permanent owned infrastructure"],
];

export const metadata = {
  title: "Agency Teardown | GrowthEngine OS",
  description: "An operational comparison of legacy agency retainers and GrowthEngine OS owned acquisition infrastructure.",
};

export default function AgencyTeardownPage() {
  return (
    <main className="teardown-shell">
      <nav className="teardown-nav">
        <a className="brand" href="/">
          <span className="brand-mark"><FileSearch size={20} /></span>
          growthengine <span className="brand-muted">OS</span>
        </a>
        <a className="back-link" href="/">
          <ArrowLeft size={15} /> back home
        </a>
      </nav>

      <header className="teardown-hero">
        <div className="eyebrow"><CircleDollarSign size={12} /> operational teardown / 2026 field note</div>
        <h1>The agency question is not who runs your ads. It is who owns the machine.</h1>
        <p>
          A practical breakdown for operators comparing GrowthEngine OS with KlientBoost, WebFX, Disruptive, Hawke, and
          the traditional retainer model. We compare failure modes, response time, economics, and what remains when the
          engagement ends.
        </p>
        <div className="teardown-actions">
          <a className="hero-button" href="/#intake">Run the gatekeeper <ArrowRight size={16} /></a>
          <a className="quiet-button" href="/#compare">See the short matrix</a>
        </div>
      </header>

      <section className="teardown-vectors" aria-label="Operational vectors">
        {vectors.map(([label, legacy, ours]) => (
          <div className="teardown-vector" key={label}>
            <span className="mono">{label}</span>
            <p><X size={14} /> {legacy}</p>
            <p><BadgeCheck size={14} /> {ours}</p>
          </div>
        ))}
      </section>

      <section className="teardown-intro">
        <div className="eyebrow"><ShieldCheck size={12} /> the standard we hold ourselves to</div>
        <h2>Use the battlecard to make a better buying decision.</h2>
        <p>
          This is not a claim that every legacy agency produces poor work. It is a structural comparison: where does a
          lead wait, where does budget leak, and which systems belong to you after the campaign is over?
        </p>
      </section>

      <section className="battlecard-list">
        {battlecards.map((card) => (
          <article className="battlecard" id={card.id} key={card.id}>
            <div className="battlecard-head">
              <div>
                <span className="mono">battlecard / {card.id}</span>
                <h2>GrowthEngine OS vs. {card.name}</h2>
                <p>{card.context}</p>
              </div>
              <Building2 size={24} />
            </div>
            <div className="battlecard-columns">
              <div>
                <h3><X size={15} /> Legacy retainer pattern</h3>
                <ul>{card.legacy.map((point) => <li key={point}>{point}</li>)}</ul>
              </div>
              <div>
                <h3><BadgeCheck size={15} /> Owned infrastructure pattern</h3>
                <ul>{card.growthengine.map((point) => <li key={point}>{point}</li>)}</ul>
              </div>
            </div>
            <a className="teardown-inline-link" href="/#intake">See how your pipeline routes <ArrowRight size={14} /></a>
          </article>
        ))}
      </section>

      <section className="teardown-callout">
        <div className="eyebrow">the kitchen test</div>
        <p>
          Hiring a traditional agency is like hiring a waiter who takes your order, drops the ticket on the kitchen counter,
          and disappears for 4 hours. By the time someone checks the ticket, the customer has walked out. GrowthEngine OS is
          the autonomous kitchen that cooks, serves, and books the next table in under 60 seconds.
        </p>
      </section>

      <footer className="legal-footer">
        <span>growthengine OS · operational truth over agency theater</span>
        <span><a href="/privacy">Privacy</a> · <a href="/terms">Terms</a> · <a href="/">Home</a></span>
      </footer>
    </main>
  );
}
