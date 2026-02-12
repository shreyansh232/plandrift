"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ChatInterface } from "@/components/chat/ChatInterface";
import { ChatHistorySidebar } from "@/components/layout/ChatHistorySidebar";
import { Header } from "@/components/layout/Header";
import { SidebarInset } from "@/components/ui/sidebar";
import { useProfile } from "@/lib/useProfile";

export function TripPageClient() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [prompt, setPrompt] = useState<string | null>(null);
  const [initialTripId, setInitialTripId] = useState<string | null>(null);
  const { user, loading, signOut } = useProfile();

  useEffect(() => {
    if (loading) return;
    if (!user) {
      router.replace("/login");
    }
  }, [loading, user, router]);

  useEffect(() => {
    const initial = searchParams.get("prompt");
    const tripIdParam =
      searchParams.get("tripId") || searchParams.get("id");
    if (tripIdParam) {
      setInitialTripId(tripIdParam);
      setPrompt(null);
    } else {
      // No tripId - this is a new trip, reset both states
      setInitialTripId(null);
      setPrompt(initial?.trim() || "");
    }
  }, [router, searchParams]);

  // Only show loading state if prompt is null (still loading) and no tripId
  // Empty string prompt is valid (new trip)
  if (prompt === null && !initialTripId) return null;

  return (
    <div className="h-svh w-full flex overflow-hidden bg-background">
      <ChatHistorySidebar user={user} loading={loading} onSignOut={signOut} />
      <SidebarInset className="flex flex-col h-full overflow-hidden relative">
        <Header user={user} loading={loading} onSignOut={signOut} />
        <ChatInterface
          key={initialTripId || "new"} // Force remount when switching trips
          initialPrompt={prompt || ""}
          initialTripId={initialTripId}
          user={user}
          loading={loading}
          onSignOut={signOut}
        />
      </SidebarInset>
    </div>
  );
}
