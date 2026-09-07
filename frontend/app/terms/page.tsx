import { ArrowLeft, FileText } from "lucide-react";

export const metadata = {
  title: "Terms of Service | GrowthEngine OS",
  description:
    "Terms governing use of GrowthEngine OS automated marketing services, self-serve engines, managed pods, budgets, approvals, and acceptable use.",
  robots: { index: true, follow: true },
};

export default function TermsPage() {
  return (
    <main className="legal-shell">
      <nav className="portal-nav">
        <a className="brand" href="/">
          <span className="brand-mark">
            <FileText size={20} />
          </span>
          growthengine <span className="brand-muted">OS</span>
        </a>
        <a className="back-link" href="/">
          <ArrowLeft size={15} /> back home
        </a>
      </nav>

      <header className="legal-hero">
        <div className="eyebrow">legal / terms of service</div>
        <h1>Terms of Service</h1>
        <p className="legal-meta">Last updated: September 4, 2026</p>
        <p>
          These terms govern your use of GrowthEngine OS, its website, self-serve engines, managed marketing pods, and
          any automated follow-up or campaign services provisioned under your account. By submitting an intake form or
          using the platform you agree to these terms.
        </p>
      </header>

      <div className="legal-body">
        <section className="legal-section">
          <h2>1. The service</h2>
          <p>
            GrowthEngine OS provides automated marketing infrastructure: video and hook creative generation, qualifying
            funnels, AI follow-up across SMS and email, print collateral generation, email drip sequences, and local
            attribution tooling. Services are delivered through self-serve engines, managed pods (Core Growth Pod,
            White-Glove Pod), and operator-run campaigns.
          </p>
        </section>

        <section className="legal-section">
          <h2>2. Tiers, fees, and ad spend</h2>
          <ul>
            <li>
              Pod selection is calibrated on your weekly ad spend: under $250/week (self-serve suite), $250–$1,250/week
              (Core Growth Pod), and $1,250+/week or full-stack operations (White-Glove Pod, currently a $4,000/month
              retainer).
            </li>
            <li>
              Agency fees and ad spend are always separate line items. Ad spend is billed directly to your own ad
              accounts with zero markup — GrowthEngine never marks up media.
            </li>
            <li>
              No budget is spent on telemarketing, influencer partnerships, social promotion, or paid advertising until
              you explicitly approve the amount for that action. Automated spend stops at your approved caps.
            </li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>3. No guaranteed results</h2>
          <p>
            Marketing outcomes depend on your market, offer, budget, and timing. GrowthEngine OS commits to process —
            testing creative batches, measuring every deployment, and iterating until a measurable acquisition path is
            found — but does not and cannot guarantee specific lead counts, appointments, or revenue. We will never
            claim a &ldquo;guaranteed sale&rdquo; because no responsible operator can.
          </p>
        </section>

        <section className="legal-section">
          <h2>4. Acceptable use and compliance</h2>
          <ul>
            <li>You may only market products and services you own or are authorized to represent.</li>
            <li>
              You agree to comply with all applicable laws, including CAN-SPAM, GDPR, TCPA, and platform terms of
              service for SMS, email, social, and advertising channels.
            </li>
            <li>
              Outreach targets consent-aware, permission-based audiences. You may not use the platform to send
              unsolicited bulk messages, purchase or scrape lists in violation of a provider&apos;s terms, impersonate
              people, or make deceptive claims.
            </li>
            <li>Opt-outs and suppression requests must be honored; the platform enforces them across channels.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>5. Your responsibilities</h2>
          <p>
            You are responsible for the accuracy of the product information, pricing, and claims you supply, for
            obtaining any rights needed for creative or collateral (including likeness rights for influencer content),
            and for the products you sell. You must keep account credentials confidential and review deployment reports
            we produce for your campaigns.
          </p>
        </section>

        <section className="legal-section">
          <h2>6. Intellectual property</h2>
          <p>
            The platform, its interfaces, and its underlying technology are owned by GrowthEngine OS. Creative,
            collateral, copy, and sequences generated for your campaigns are licensed to you for use in marketing those
            campaigns. You retain all rights in the products, branding, and materials you provide.
          </p>
        </section>

        <section className="legal-section">
          <h2>7. Disclaimers and limitation of liability</h2>
          <p>
            The service is provided &ldquo;as is&rdquo; without warranties of any kind, express or implied. To the
            maximum extent permitted by law, GrowthEngine OS is not liable for indirect, incidental, or consequential
            damages, lost profits, or lost revenue arising from use of the service, and its total liability for any
            claim is limited to the fees you paid in the twelve months preceding the claim. Nothing in these terms
            limits liability that cannot be limited by law.
          </p>
        </section>

        <section className="legal-section">
          <h2>8. Termination</h2>
          <p>
            You may stop using the service at any time. We may suspend or terminate access for breach of these terms,
            non-payment, or conduct that threatens the platform or its users. On termination, pending automated
            follow-up ceases, outstanding invoices remain due, and your data is handled per the Privacy Policy.
          </p>
        </section>

        <section className="legal-section">
          <h2>9. Changes and governing law</h2>
          <p>
            We may update these terms with notice on this page. Continued use after changes constitutes acceptance.
            These terms are governed by the laws of the jurisdiction where GrowthEngine OS is established, without
            regard to conflict-of-law rules.
          </p>
        </section>

        <section className="legal-section">
          <h2>10. Contact</h2>
          <p>
            Questions about these terms:{" "}
            <a href="mailto:legal@growthengine.local" className="legal-link">
              legal@growthengine.local
            </a>
            .
          </p>
        </section>
      </div>

      <footer className="legal-footer">
        <span>growthengine OS · appointments, engineered</span>
        <span>
          <a href="/privacy">Privacy Policy</a> · <a href="/">Home</a>
        </span>
      </footer>
    </main>
  );
}
