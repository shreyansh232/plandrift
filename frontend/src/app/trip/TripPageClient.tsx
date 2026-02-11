"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ChatInterface } from "@/components/chat/ChatInterface";
import { useProfile } from "@/lib/useProfile";

export function TripPageClient() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [prompt, setPrompt] = useState<string | null>(null);
  const { user, loading, signOut } = useProfile();

  useEffect(() => {
    if (loading) return;
    if (!user) {
      router.replace("/login");
    }
  }, [loading, user, router]);

  useEffect(() => {
    const initial = searchParams.get("prompt");
    if (!initial || !initial.trim()) {
      router.replace("/");
      return;
    }
    setPrompt(initial.trim());
  }, [router, searchParams]);

  if (!prompt) return null;

  return (
    <div className="h-screen overflow-hidden bg-white">
      <ChatInterface
        initialPrompt={prompt}
        user={user}
        loading={loading}
        onSignOut={signOut}
      />
    </div>
  );
}
