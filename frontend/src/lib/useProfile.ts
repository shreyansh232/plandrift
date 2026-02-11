"use client";

import { useEffect, useState } from "react";
import { getProfile, isAuthenticated, logout } from "@/lib/api";
import type { AuthUser } from "@/lib/api";

export function useProfile() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    if (!isAuthenticated()) {
      setLoading(false);
      return;
    }
    setLoading(true);
    getProfile()
      .then((data) => {
        if (active) setUser(data);
      })
      .catch(() => {
        if (active) setUser(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const signOut = () => logout().finally(() => setUser(null));

  return { user, loading, signOut };
}
