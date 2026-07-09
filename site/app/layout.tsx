import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "LCT — Geometry, Speed, and an Honest Negative Result",
    template: "%s · LCT interactive paper",
  },
  description:
    "An interactive 3D paper on determinant-one linear canonical transforms, NanoGPT controls, and the experiments that overturned an early positive result.",
  applicationName: "LCT interactive paper",
  authors: [{ name: "Alok Singh" }],
  keywords: [
    "linear canonical transform",
    "structured linear layers",
    "NanoGPT",
    "phase space",
    "machine learning evaluation",
  ],
  openGraph: {
    title: "The transform keeps its area. The benchmark keeps us honest.",
    description:
      "Touch the determinant-one geometry, then audit the fair-control NanoGPT evidence.",
    type: "article",
  },
  twitter: {
    card: "summary",
    title: "LCT — an interactive paper",
    description: "A 3D phase-space instrument and an honest negative result.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
