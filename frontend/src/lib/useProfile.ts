"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { getProfile, isAuthenticated, logout } from "@/lib/api";
import type { AuthUser } from "@/lib/api";

/**
 * Profile hook that keeps the user in sync with the auth state.
 *
 * - Fetches profile on mount if a token exists.
 * - Re-fetches when the tab becomes visible again (handles token refresh).
 * - Listens for a custom "auth:tokens-updated" event to re-fetch after
 *   silent token refreshes elsewhere in the app.
 */
export function useProfile() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const fetchingRef = useRef(false);

  const fetchUser = useCallback(async () => {
    if (fetchingRef.current) return;
    if (!isAuthenticated()) {
      setUser(null);
      setLoading(false);
      return;
    }
    fetchingRef.current = true;
    try {
      const data = await getProfile();
      setUser(data);
    } catch {
      // If getProfile fails even after refresh, user is truly logged out
      setUser(null);
    } finally {
      setLoading(false);
      fetchingRef.current = false;
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  // Re-fetch on tab visibility change (user returns to tab after token expired)
  useEffect(() => {
    const handleVisibility = () => {
      if (document.visibilityState === "visible") {
        fetchUser();
      }
    };
    document.addEventListener("visibilitychange", handleVisibility);
    return () =>
      document.removeEventListener("visibilitychange", handleVisibility);
  }, [fetchUser]);

  // Re-fetch when tokens are refreshed anywhere in the app
  useEffect(() => {
    const handleTokenUpdate = () => {
      fetchUser();
    };
    window.addEventListener("auth:tokens-updated", handleTokenUpdate);
    return () =>
      window.removeEventListener("auth:tokens-updated", handleTokenUpdate);
  }, [fetchUser]);

  // Periodic heartbeat — re-validate every 10 minutes
  useEffect(() => {
    const interval = setInterval(() => {
      if (isAuthenticated()) {
        fetchUser();
      }
    }, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchUser]);

  const signOut = () => logout().finally(() => setUser(null));

  return { user, loading, signOut };
}
