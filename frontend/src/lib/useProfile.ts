"use client";

import { useEffect, useState } from "react";
import { getProfile, isAuthenticated, logout } from "@/lib/api";
import type { AuthUser } from "@/lib/api";

export function useProfile() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isAuthenticated()) return;
    setLoading(true);
    getProfile()
      .then((data) => setUser(data))
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  const signOut = () => logout().finally(() => setUser(null));

  return { user, loading, signOut };
}
