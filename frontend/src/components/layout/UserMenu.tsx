"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import type { AuthUser } from "@/lib/api";
import Link from "next/link";

type UserMenuProps = {
  user: AuthUser | null;
  loading?: boolean;
  onSignOut: () => void;
};

function getInitials(name?: string | null, email?: string | null): string {
  const source = name || email || "";
  if (!source) return "U";
  const parts = source.split(" ").filter(Boolean);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
}

export function UserMenu({ user, loading, onSignOut }: UserMenuProps) {
  const [open, setOpen] = useState(false);
  const [imageError, setImageError] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (event: MouseEvent) => {
      if (!wrapperRef.current) return;
      if (!wrapperRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  if (!user) {
    return (
      <Link href="/login">
        <Button
          variant="outline"
          className="border-black/20"
          disabled={loading}
        >
          Sign in
        </Button>
      </Link>
    );
  }

  const initials = getInitials(user.name, user.email);
  const showImage = !!user.picture_url && !imageError;

  return (
    <div className="relative" ref={wrapperRef}>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="size-9 rounded-full border border-black/10 overflow-hidden bg-white flex items-center justify-center text-xs font-semibold text-black"
      >
        {showImage ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={user.picture_url ?? ""}
            alt={user.name || user.email}
            className="h-full w-full object-cover"
            referrerPolicy="no-referrer"
            onError={() => setImageError(true)}
          />
        ) : (
          initials
        )}
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-40 rounded-xl border border-black/10 bg-white shadow-lg">
          <button
            type="button"
            onClick={() => {
              setOpen(false);
              onSignOut();
            }}
            className="w-full px-4 py-2 text-left text-sm text-black hover:bg-slate-50"
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
