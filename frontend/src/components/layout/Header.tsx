"use client";

import Link from "next/link";
import Image from "next/image";
import { UserMenu } from "./UserMenu";
import { useProfile } from "@/lib/useProfile";
import type { AuthUser } from "@/lib/api";

interface HeaderProps {
  showUserMenu?: boolean;
  children?: React.ReactNode;
  /** When provided, the Header uses these instead of creating its own useProfile hook. */
  user?: AuthUser | null;
  loading?: boolean;
  onSignOut?: () => void;
}

export function Header({
  showUserMenu = true,
  children,
  user: userProp,
  loading: loadingProp,
  onSignOut: onSignOutProp,
}: HeaderProps) {
  // Use the shared profile from props if provided; otherwise fall back to
  // an independent hook (e.g. when Header is used outside trip pages).
  const ownProfile = useProfile();
  const user = userProp !== undefined ? userProp : ownProfile.user;
  const loading = loadingProp !== undefined ? loadingProp : ownProfile.loading;
  const signOut = onSignOutProp ?? ownProfile.signOut;

  return (
    <header className="relative z-20 max-w-6xl mx-auto flex items-center justify-between px-6 py-6 w-full">
      <Link href="/" className="font-medium text-xl tracking-tight text-foreground flex items-center gap-2">
        <Image src="/favicon.ico" alt="" width={24} height={24} className="size-6" />
        Planfirst
      </Link>
      <nav className="flex items-center gap-6">
        {showUserMenu && (
          <UserMenu user={user} loading={loading} onSignOut={signOut} />
        )}
        {children}
      </nav>
    </header>
  );
}
