import "./globals.css";

export const metadata = {
  metadataBase: new URL("https://growthengine.local"),
  title: "GrowthEngine OS | AI Marketing Firm Filling Calendars for Sales Teams",
  description:
    "Autonomous AI marketing firm: engineered video ad creative, a 3-question qualifying funnel, and zero-latency AI follow-up that holds every lead until the appointment is booked.",
  keywords: [
    "AI marketing firm",
    "AI marketing agency",
    "appointment setting",
    "speed to lead",
    "AI follow-up",
    "qualified inbound leads",
  ],
  openGraph: {
    title: "GrowthEngine OS — AI Marketing Firm Filling Calendars",
    description:
      "Creative that finds the lead. A funnel that sorts it. AI that holds the conversation until the calendar fills.",
    type: "website",
    siteName: "GrowthEngine OS",
    url: "https://growthengine.local",
  },
  robots: { index: true, follow: true },
};

export const viewport = {
  themeColor: "#f8f9fa",
  width: "device-width",
  initialScale: 1,
};

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "Organization",
  name: "GrowthEngine OS",
  description:
    "AI marketing firm running creative, qualifying funnels, and persistent AI follow-up that fills sales calendars.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap"
        />
      </head>
      <body>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        {children}
      </body>
    </html>
  );
}
