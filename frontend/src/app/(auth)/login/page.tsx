"use client";

import { motion } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import Image from "next/image";
import { getGoogleLoginUrl } from "@/lib/api";
import { Header } from "@/components/layout/Header";

export default function LoginPage() {
  const handleGoogleSignIn = () => {
    window.location.href = getGoogleLoginUrl();
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col w-full">
      <Header showUserMenu={false}>
        <Link
          href="/"
          className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1.5"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Home
        </Link>
      </Header>

      <main className="flex-1 flex flex-col lg:flex-row overflow-hidden w-full">
        {/* Left: Overlapping Photo Collage */}
        <div className="hidden lg:flex lg:w-1/2 relative items-center justify-center bg-secondary/40 overflow-hidden">
          <div className="relative w-[420px] h-[480px]">
            <motion.div
              initial={{ opacity: 0, rotate: -8 }}
              animate={{ opacity: 1, rotate: -6 }}
              transition={{ duration: 0.6, delay: 0 }}
              className="absolute top-0 left-0 w-[240px] h-[300px] rounded-2xl overflow-hidden shadow-xl border-4 border-background z-10"
            >
              <Image
                src="/login/collage-1.jpg"
                alt="Turquoise ocean aerial view"
                fill
                className="object-cover"
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, rotate: 5 }}
              animate={{ opacity: 1, rotate: 3 }}
              transition={{ duration: 0.6, delay: 0.12 }}
              className="absolute top-6 right-0 w-[220px] h-[280px] rounded-2xl overflow-hidden shadow-xl border-4 border-background z-20"
            >
              <Image
                src="/login/collage-2.jpg"
                alt="European golden hour street"
                fill
                className="object-cover"
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, rotate: 4 }}
              animate={{ opacity: 1, rotate: 2 }}
              transition={{ duration: 0.6, delay: 0.24 }}
              className="absolute bottom-0 left-8 w-[230px] h-[260px] rounded-2xl overflow-hidden shadow-xl border-4 border-background z-30"
            >
              <Image
                src="/login/collage-3.jpg"
                alt="Alpine mountain lake reflection"
                fill
                className="object-cover"
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, rotate: -3 }}
              animate={{ opacity: 1, rotate: -2 }}
              transition={{ duration: 0.6, delay: 0.36 }}
              className="absolute bottom-8 right-4 w-[200px] h-[270px] rounded-2xl overflow-hidden shadow-xl border-4 border-background z-40"
            >
              <Image
                src="/login/collage-4.jpg"
                alt="Cherry blossoms in Kyoto"
                fill
                className="object-cover"
              />
            </motion.div>
          </div>
        </div>

        {/* Right: Sign In */}
        <div className="flex-1 flex items-center justify-center px-8 bg-background">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="w-full max-w-sm"
          >
            <div className="text-center mb-10">
              <h1 className="text-3xl sm:text-4xl mb-3 font-semibold text-foreground">
                Welcome to Planfirst
              </h1>
              <p className="text-muted-foreground">
                Sign in to start planning your next journey.
              </p>
            </div>

            <button
              onClick={handleGoogleSignIn}
              className="w-full flex items-center justify-center gap-3 bg-primary text-primary-foreground rounded-full py-3.5 px-6 text-sm font-medium hover:opacity-90 transition-opacity cursor-pointer"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" />
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
              </svg>
              Continue with Google
            </button>
          </motion.div>
        </div>
      </main>
    </div>
  );
}