"use client";

import { useState, Suspense } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Compass, Shield, Zap, Globe } from "lucide-react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { ChatHistorySidebar } from "@/components/layout/ChatHistorySidebar";
import { Header } from "@/components/layout/Header";
import { SidebarInset } from "@/components/ui/sidebar";
import { useProfile } from "@/lib/useProfile";

const Home = () => {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");
  const { user, loading, signOut } = useProfile();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    const encoded = encodeURIComponent(prompt.trim());
    router.push(`/trip?prompt=${encoded}`);
  };

  const handleAutoStart = () => {
    if (!prompt.trim()) return;
    const encoded = encodeURIComponent(prompt.trim());
    router.push(`/trip?prompt=${encoded}`);
  };

  return (
    <div className="min-h-svh w-full flex bg-background">
      <Suspense fallback={<div className="w-16 bg-[#FAFAF8]" />}>
        <ChatHistorySidebar user={user} loading={loading} onSignOut={signOut} />
      </Suspense>
      <SidebarInset className="flex-1 flex flex-col">
        <Header />

        <main className="flex-1 max-w-6xl mx-auto px-6 w-full">
          {/* Hero */}
          <section className="pt-16 pb-24">
            <div className="grid lg:grid-cols-2 gap-16 items-center">
              {/* Left: Copy + Input */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="flex flex-col gap-8"
              >
                <div className="flex flex-col gap-5">
                  <span className="text-sm font-medium uppercase tracking-[0.3em] text-accent">
                    AI-Powered Travel
                  </span>
                  <h1 className="text-4xl sm:text-5xl lg:text-6xl leading-[1.1] text-foreground">
                    Your next journey,{" "}
                    <span className="italic">thoughtfully planned</span>
                  </h1>
                  <p className="text-lg text-muted-foreground max-w-lg leading-relaxed">
                    Describe your dream trip in a sentence. We&apos;ll craft a
                    complete itinerary — flights, stays, experiences — tailored
                    to your budget and style.
                  </p>
                </div>

                <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                  <div className="rounded-2xl border-2 border-border bg-card shadow-lg p-4">
                    <textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleAutoStart();
                        }
                      }}
                      placeholder="Plan a trip to Chicago from NYC for a week with a $1,500 budget..."
                      aria-label="Describe your trip"
                      className="w-full h-24 resize-none text-base bg-transparent text-foreground placeholder:text-muted-foreground/60 focus:outline-none font-body"
                    />
                    <div className="flex justify-end pt-2">
                      <button
                        type="submit"
                        className="inline-flex items-center gap-2 bg-primary text-primary-foreground px-6 py-2.5 rounded-full text-sm font-medium hover:opacity-90 transition-opacity cursor-pointer"
                      >
                        Start planning
                        <ArrowRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </form>

                <div className="flex items-center gap-8 text-sm text-muted-foreground">
                  <span className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                    Personalized
                  </span>
                  <span className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                    Budget-aware
                  </span>
                  <span className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                    Multilingual
                  </span>
                </div>
              </motion.div>

              {/* Right: Hero Image */}
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="relative hidden lg:block"
              >
                <div className="rounded-3xl overflow-hidden shadow-2xl relative w-full h-[520px]">
                  <Image
                    src="/hero/hero-travel.jpg"
                    alt="Stunning coastal village at golden hour"
                    fill
                    className="object-cover"
                    priority
                  />
                </div>
                <div className="absolute -bottom-6 -left-6 bg-card rounded-2xl shadow-lg border border-border p-4 max-w-[220px] z-10">
                  <p className="text-xs text-muted-foreground mb-1">Latest trip planned</p>
                  <p className="text-sm font-medium text-foreground">7 days in Cinque Terre</p>
                  <p className="text-xs text-accent mt-1">$2,400 · 2 travelers</p>
                </div>
              </motion.div>
            </div>
          </section>

          {/* How it works */}
          <section id="how" className="py-24 border-t border-border">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <span className="text-sm font-medium uppercase tracking-[0.3em] text-accent mb-4 block">
                How it works
              </span>
              <h2 className="text-3xl sm:text-4xl mb-4 text-foreground">
                Travel planning, reimagined
              </h2>
              <p className="text-muted-foreground text-lg max-w-xl mx-auto">
                From idea to itinerary — simple, fast, and personalized.
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                {
                  icon: Compass,
                  title: "Describe your trip",
                  desc: "Share where, when, and your budget.",
                },
                {
                  icon: Zap,
                  title: "We build your plan",
                  desc: "AI researches and creates your trip plan.",
                },
                {
                  icon: Shield,
                  title: "Refine and go",
                  desc: "Swap, add, or change anything instantly.",
                },
                {
                  icon: Globe,
                  title: "Chat in any language",
                  desc: "Talk naturally — we understand any language.",
                },
              ].map((item, i) => (
                <motion.div
                  key={item.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 1.2, delay: i * 0.3 }}
                  className="group bg-card border border-border rounded-2xl p-5 hover:shadow-lg hover:border-accent/20 transition-all duration-500"
                >
                  <div className="w-12 h-12 rounded-xl bg-accent/10 flex items-center justify-center mb-6 group-hover:bg-accent/20 transition-colors">
                    <item.icon className="w-6 h-6 text-accent" />
                  </div>
                  <h3 className="text-lg font-medium mb-3 text-foreground group-hover:text-accent transition-colors">{item.title}</h3>
                  <p className="text-muted-foreground text-sm leading-tighter">{item.desc}</p>
                </motion.div>
              ))}
            </div>
          </section>

          {/* Footer */}
          <footer className="py-12 border-t border-border flex items-center justify-between text-sm text-muted-foreground">
            <span className="font-display text-foreground">Plandrift</span>
            <span>© {new Date().getFullYear()} Plandrift. All rights reserved.</span>
          </footer>
        </main>
      </SidebarInset>
    </div>
  );
};

export default Home;
