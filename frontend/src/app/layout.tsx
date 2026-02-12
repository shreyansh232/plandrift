import type { Metadata } from "next";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider } from "@/components/ui/sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Planfirst",
  description: "Plan better trips, faster.",
  openGraph: {
    title: "Planfirst",
    description: "Plan better trips, faster.",
    images: [
      {
        url: "/og.png",
        width: 1200,
        height: 630,
        alt: "Planfirst - AI That Plans Before You Do",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Planfirst",
    description: "Plan better trips, faster.",
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <TooltipProvider>
          <SidebarProvider defaultOpen={false}>
            {children}
          </SidebarProvider>
        </TooltipProvider>
      </body>
    </html>
  );
}
