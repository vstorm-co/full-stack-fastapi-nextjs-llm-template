import { ArrowLeft, ShieldCheck } from "lucide-react";

export const metadata = {
  title: "Privacy Policy | GrowthEngine OS",
  description:
    "How GrowthEngine OS collects, uses, stores, and deletes your data across its automated marketing engines, intake forms, and follow-up channels.",
  robots: { index: true, follow: true },
};

export default function PrivacyPage() {
  return (
    <main className="legal-shell">
      <nav className="portal-nav">
        <a className="brand" href="/">
          <span className="brand-mark">
            <ShieldCheck size={20} />
          </span>
          growthengine <span className="brand-muted">OS</span>
        </a>
        <a className="back-link" href="/">
          <ArrowLeft size={15} /> back home
        </a>
      </nav>

      <header className="legal-hero">
        <div className="eyebrow">legal / privacy policy</div>
        <h1>Privacy Policy</h1>
        <p className="legal-meta">Last updated: September 4, 2026</p>
        <p>
          GrowthEngine OS operates automated marketing engines — video creative, qualifying funnels, AI follow-up
          (SMS/email), print collateral, and local attribution tools. This policy explains what data we collect when
          you use the site or submit intake forms, why we collect it, and how you can access or delete it.
        </p>
      </header>

      <div className="legal-body">
        <section className="legal-section">
          <h2>1. Information we collect</h2>
          <ul>
            <li>
              <strong>Intake and booking data</strong> — name, work email, phone number, company name, your primary
              marketing bottleneck, weekly ad-spend bracket, requested service modules, preferred call time, and target
              start date when you complete a form.
            </li>
            <li>
              <strong>Communication data</strong> — replies you send to our follow-up messages and the channels you use
              (SMS, email), so the AI chaser can keep thread state.
            </li>
            <li>
              <strong>Usage data</strong> — pages visited, referring pages, device and browser information, and
              timestamps, used to improve the product and measure campaign deployments.
            </li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>2. How we use your information</h2>
          <ul>
            <li>Route your intake through the correct pod (self-serve, Growth Pod, or White-Glove).</li>
            <li>Run consent-aware automated follow-up across SMS and email until a slot is booked or you opt out.</li>
            <li>Provision the self-serve engines you select and generate campaign creative, collateral, and drip sequences.</li>
            <li>Produce deployment reports and measure conversion outcomes for campaigns you run through the platform.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>3. Consent, opt-out, and suppression</h2>
          <p>
            Follow-up messages are only sent where you have provided consent or where an applicable legal basis exists.
            Every outreach message includes a clear way to stop contact: reply <strong>STOP</strong>, use the
            unsubscribe link, or email{" "}
            <a href="mailto:privacy@growthengine.local" className="legal-link">
              privacy@growthengine.local
            </a>
            . Opt-outs are recorded on a suppression list and honored across every channel we operate. When a lead
            opts out, the automated chaser stops permanently for that contact.
          </p>
        </section>

        <section className="legal-section">
          <h2>4. Automated processing disclosure</h2>
          <p>
            Conversations, follow-up messages, and intake routing may be handled by automated software agents rather
            than human staff. If you prefer to speak with a person, reply with the word <strong>HUMAN</strong> or email
            us — a team member will take over.
          </p>
        </section>

        <section className="legal-section">
          <h2>5. Sharing and third parties</h2>
          <p>
            We do not sell your personal data. Data is shared only with service providers that run the platform
            (hosting, messaging, analytics) under data-processing terms, and only as needed to deliver the services you
            request. Ad spend is billed directly to your own ad accounts; we never silently resell your data to
            third-party brokers.
          </p>
        </section>

        <section className="legal-section">
          <h2>6. Security and retention</h2>
          <p>
            Data is encrypted in transit, stored behind access controls, and retained only as long as needed to run your
            campaigns and satisfy legal obligations. Lead and booking records that go stale or are disputed are
            suppressed or deleted on request.
          </p>
        </section>

        <section className="legal-section">
          <h2>7. Your rights</h2>
          <p>
            You may request access to, correction of, or deletion of the personal data we hold about you. Email{" "}
            <a href="mailto:privacy@growthengine.local" className="legal-link">
              privacy@growthengine.local
            </a>{" "}
            from the address you used at intake; we respond within 30 days. Where required by law (for example GDPR or
            CCPA/CPRA), deletion requests extend to all copies we control.
          </p>
        </section>

        <section className="legal-section">
          <h2>8. Changes to this policy</h2>
          <p>
            We may update this policy as the platform evolves. Material changes are announced on this page with a new
            &ldquo;last updated&rdquo; date. Continued use after a change means you accept the updated policy.
          </p>
        </section>
      </div>

      <footer className="legal-footer">
        <span>growthengine OS · appointments, engineered</span>
        <span>
          <a href="/terms">Terms of Service</a> · <a href="/">Home</a>
        </span>
      </footer>
    </main>
  );
}
