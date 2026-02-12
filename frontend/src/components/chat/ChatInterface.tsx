"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import { ArrowUp, Sparkles, Loader2, Plane, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  startTripTokenStream,
  clarifyTripTokenStream,
  proceedTripTokenStream,
  confirmAssumptionsTokenStream,
  refineTripTokenStream,
  getTripMessages,
  getTrip,
  ApiError,
  clearTokens,
} from "@/lib/api";
import type { AuthUser, StreamEvent, StreamMeta } from "@/lib/api";
import { MarkdownRenderer } from "@/components/chat/MarkdownRenderer";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  phase?: string;
}

interface ChatInterfaceProps {
  initialPrompt?: string;
  initialTripId?: string | null;
  user?: AuthUser | null;
  loading?: boolean;
  onSignOut?: () => void;
}

type NextAction =
  | "text_input"
  | "proceed_confirm"
  | "proceed_continue"
  | "assumptions_confirm"
  | "modify_input"
  | "done";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ChatInterface({
  initialPrompt = "",
  initialTripId = null,
  user,
}: ChatInterfaceProps) {
  const router = useRouter();
  const idCounterRef = useRef(0);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("Thinking...");
  const [tripId, setTripId] = useState<string | null>(null);
  const [nextAction, setNextAction] = useState<NextAction>("text_input");
  const [hasHighRisk, setHasHighRisk] = useState(false);
  const [restored, setRestored] = useState(false);
  const [streamingHasDelta, setStreamingHasDelta] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const startedRef = useRef(false);
  const startInFlightRef = useRef(false);
  const actionInFlightRef = useRef(false);
  const initialMessageIdRef = useRef<string | null>(null);
  const pendingPromptRef = useRef(initialPrompt);
  const storageKey = useMemo(() => {
    if (tripId) return `planfirst_chat_trip_${tripId}`;
    if (initialTripId) return `planfirst_chat_trip_${initialTripId}`;
    return `planfirst_chat_prompt_${initialPrompt || "new"}`;
  }, [initialPrompt, initialTripId, tripId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    if (!isLoading && nextAction === "text_input") {
      inputRef.current?.focus();
    }
  }, [isLoading, nextAction]);

  useEffect(() => {
    pendingPromptRef.current = initialPrompt;
  }, [initialPrompt]);

  // Reset state when navigating to a new trip (no initialTripId)
  useEffect(() => {
    if (!initialTripId) {
      // Clear all state for new trip
      setMessages([]);
      setInput("");
      setTripId(null);
      setNextAction("text_input");
      setHasHighRisk(false);
      setRestored(true);
      startedRef.current = false;
      startInFlightRef.current = false;
      actionInFlightRef.current = false;
      initialMessageIdRef.current = null;
      idCounterRef.current = 0;
    }
  }, [initialTripId]);


  const addMessage = useCallback(
    (role: Message["role"], content: string, phase?: string) => {
      const cleaned =
        role === "assistant"
          ? content.replace(/\n{3,}/g, "\n\n")
          : content;
      const id = `${Date.now()}-${++idCounterRef.current}`;

      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === role && last.content === cleaned) {
          return prev;
        }
        return [
          ...prev,
          {
            id,
            role,
            content: cleaned,
            phase,
          },
        ];
      });
    },
    [],
  );

  const consumeStream = useCallback(
    async (
      stream: AsyncGenerator<StreamEvent, void, void>,
      nextActionFromMeta?: (meta: StreamMeta | null) => NextAction,
    ) => {
      const id = `${Date.now()}-${++idCounterRef.current}`;
      let meta: StreamMeta | null = null;
      let hasContent = false;
      let created = false;
      initialMessageIdRef.current = null;
      setStreamingHasDelta(false);

      const ensureMessage = () => {
        if (created) return;
        const createdId = initialMessageIdRef.current || id;
        initialMessageIdRef.current = createdId;
        setMessages((prev) => [
          ...prev,
          { id: createdId, role: "assistant", content: "", phase: undefined },
        ]);
        created = true;
      };

      for await (const event of stream) {
        if (event.type === "meta") {
          meta = event.data;
          if (event.data.trip_id) setTripId(event.data.trip_id);
          setHasHighRisk(event.data.has_high_risk);
        }
        if (event.type === "status") {
          setLoadingText(event.data);
        }
        // Handle delta events (chunked streaming)
        if (event.type === "delta") {
          const cleaned = event.data.replace(/\n{3,}/g, "\n\n");
          if (!created) {
            ensureMessage();
          }
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === (initialMessageIdRef.current || id)
                ? {
                    ...msg,
                    content: hasContent ? msg.content + cleaned : cleaned,
                  }
                : msg,
            ),
          );
          hasContent = true;
          setStreamingHasDelta(true);
        }
        // Handle token events (character-by-character streaming)
        if (event.type === "token") {
          if (!created) {
            ensureMessage();
          }
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === (initialMessageIdRef.current || id)
                ? { ...msg, content: msg.content + event.data }
                : msg,
            ),
          );
          hasContent = true;
          setStreamingHasDelta(true);
        }
      }

      if (nextActionFromMeta) {
        setNextAction(nextActionFromMeta(meta));
      }
      setStreamingHasDelta(false);
    },
    [],
  );

  const handleError = useCallback(
    (err: unknown) => {
      if (err instanceof ApiError && err.status === 401) {
        clearTokens();
        router.push("/login");
        return;
      }
      
      const message =
        err instanceof ApiError
          ? err.detail
          : "Something went wrong. Please try again.";
      addMessage("assistant", `Error: ${message}`);
    },
    [addMessage, router],
  );

  // ---- phase handlers ----

  const doStart = useCallback(
    async (overridePrompt?: string, force = false) => {
      if (startInFlightRef.current) return;
      if (!force && startedRef.current) return;
      startInFlightRef.current = true;
      startedRef.current = true;
      setIsLoading(true);
      setLoadingText("Starting your trip plan...");

      try {
        const prompt = (overridePrompt || initialPrompt).trim();
        if (!prompt) {
          throw new Error("Please enter a trip request to get started.");
        }
        pendingPromptRef.current = prompt;
        // Only add user message on the initial call (not force re-starts,
        // since handleSubmit already adds the user message before calling doStart).
        if (!force) {
          addMessage("user", prompt);
        }
        const stream = await startTripTokenStream(prompt);
        await consumeStream(stream, () => "text_input");
      } catch (err) {
        handleError(err);
      } finally {
        setIsLoading(false);
        startInFlightRef.current = false;
      }
  }, [initialPrompt, consumeStream, handleError, addMessage]);

  const doClarify = useCallback(
    async (answers: string) => {
      if (!tripId) return;
      if (actionInFlightRef.current) return;
      actionInFlightRef.current = true;
      setIsLoading(true);
      setLoadingText("Analyzing feasibility...");

      try {
        const stream = await clarifyTripTokenStream(tripId, answers);
        await consumeStream(stream, (meta) =>
          meta?.has_high_risk ? "proceed_confirm" : "proceed_continue",
        );
      } catch (err) {
        handleError(err);
        setNextAction("text_input");
      } finally {
        setIsLoading(false);
        actionInFlightRef.current = false;
      }
    },
    [tripId, consumeStream, handleError],
  );

  const doProceed = useCallback(
    async (proceed: boolean, force = false) => {
      if (!tripId) return;
      if (
        !force &&
        nextAction !== "proceed_continue" &&
        nextAction !== "proceed_confirm"
      ) {
        return;
      }
      if (actionInFlightRef.current) return;
      actionInFlightRef.current = true;
      setIsLoading(true);
      setLoadingText(
        proceed ? "Generating assumptions..." : "Processing...",
      );

      try {
        const stream = await proceedTripTokenStream(tripId, proceed);
        await consumeStream(stream, (meta) =>
          meta?.phase === "planning"
            ? "done"
            : meta?.phase === "assumptions"
            ? "assumptions_confirm"
            : "text_input",
        );
      } catch (err) {
        handleError(err);
      } finally {
        setIsLoading(false);
        actionInFlightRef.current = false;
      }
    },
    [tripId, nextAction, consumeStream, handleError],
  );

  const doConfirmAssumptions = useCallback(
    async (confirmed: boolean, modifications?: string) => {
      if (!tripId) return;
      if (actionInFlightRef.current) return;
      actionInFlightRef.current = true;
      setIsLoading(true);
      setLoadingText(
        "Researching prices and building your itinerary...",
      );

      try {
        const stream = await confirmAssumptionsTokenStream(
          tripId,
          confirmed,
          modifications,
        );
        await consumeStream(stream, () => "done");
      } catch (err) {
        handleError(err);
        setNextAction("assumptions_confirm");
      } finally {
        setIsLoading(false);
        actionInFlightRef.current = false;
      }
    },
    [tripId, consumeStream, handleError],
  );

  const doRefine = useCallback(
    async (refinementType: string) => {
      if (!tripId) return;
      if (actionInFlightRef.current) return;
      actionInFlightRef.current = true;
      setIsLoading(true);
      setLoadingText("Refining your plan...");

      try {
        const stream = await refineTripTokenStream(tripId, refinementType);
        await consumeStream(stream, () => "done");
      } catch (err) {
        handleError(err);
        setNextAction("done");
      } finally {
        setIsLoading(false);
        actionInFlightRef.current = false;
      }
    },
    [tripId, consumeStream, handleError],
  );

  // ---- load on mount ----

  useEffect(() => {
    if (initialTripId) return;
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) {
        setRestored(true);
        return;
      }
      const stored = JSON.parse(raw) as {
        messages?: Message[];
        tripId?: string | null;
        nextAction?: NextAction;
        hasHighRisk?: boolean;
      };
      if (stored.messages?.length) {
        const seen = new Set<string>();
        const deduped = stored.messages.map((msg, idx) => {
          if (!seen.has(msg.id)) {
            seen.add(msg.id);
            return msg;
          }
          const newId = `${msg.id}-${idx}`;
          seen.add(newId);
          return { ...msg, id: newId };
        });
        setMessages(deduped);
        setTripId(stored.tripId || null);
        setNextAction(stored.nextAction || "text_input");
        setHasHighRisk(!!stored.hasHighRisk);
        startedRef.current = true;
      }
      setRestored(true);
    } catch {
      setRestored(true);
    }
  }, [initialTripId, storageKey]);

  useEffect(() => {
    if (!initialTripId) return;
    let mounted = true;
    (async () => {
      try {
        const [history, tripDetail] = await Promise.all([
          getTripMessages(initialTripId),
          getTrip(initialTripId),
        ]);
        if (!mounted) return;
        setTripId(initialTripId);
        
        const mapped = history.map((msg, idx) => ({
          id: msg.id || `${msg.created_at}-${idx}`,
          role: msg.role as Message["role"],
          content: msg.content,
          phase: msg.phase ?? undefined,
        }));
        
        if (mapped.length > 0) {
          setMessages(mapped);
        } else {
          const storageKey = `planfirst_chat_trip_${initialTripId}`;
          const stored = localStorage.getItem(storageKey);
          if (stored) {
            try {
              const parsed = JSON.parse(stored) as {
                messages?: Message[];
              };
              if (parsed.messages && parsed.messages.length > 0) {
                setMessages(parsed.messages);
              }
            } catch {
              // Ignore parse errors
            }
          }
        }
        
        const phase = tripDetail.latest_version?.phase;
        const risk = tripDetail.latest_version
          ?.risk_assessment_json as Record<string, unknown> | null;
        if (phase === "feasibility") {
          const overall = risk?.overall_feasible;
          const highRisk = overall === false;
          setHasHighRisk(highRisk);
          setNextAction(highRisk ? "proceed_confirm" : "proceed_continue");
        } else if (phase === "assumptions") {
          setNextAction("assumptions_confirm");
        } else if (phase === "planning" || phase === "refinement") {
          setNextAction("done");
        } else {
          setNextAction("text_input");
        }
        startedRef.current = true;
        setRestored(true);
      } catch (err) {
        if (!mounted) return;
        
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          router.push("/login");
          return;
        }
        
        const storageKey = `planfirst_chat_trip_${initialTripId}`;
        const stored = localStorage.getItem(storageKey);
        if (stored) {
          try {
            const parsed = JSON.parse(stored) as {
              messages?: Message[];
            };
            if (parsed.messages && parsed.messages.length > 0) {
              setMessages(parsed.messages);
            }
          } catch {
            // Ignore parse errors
          }
        }
        setRestored(true);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [initialTripId]);

  useEffect(() => {
    if (!restored) return;
    if (startedRef.current) return;
    if (!initialPrompt.trim()) return;
    doStart();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [restored]);

  useEffect(() => {
    const payload = JSON.stringify({
      messages,
      tripId,
      nextAction,
      hasHighRisk,
    });
    localStorage.setItem(storageKey, payload);
  }, [messages, tripId, nextAction, hasHighRisk, storageKey]);

  // ---- form submission ----

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;

    addMessage("user", text);
    setInput("");

    switch (nextAction) {
      case "text_input":
        if (!tripId) {
          const basePrompt = pendingPromptRef.current || initialPrompt;
          const lastAssistant = [...messages]
            .reverse()
            .find((msg) => msg.role === "assistant");
          const lastText = lastAssistant?.content.toLowerCase() || "";
          const expectsOrigin =
            lastText.includes("traveling from") ||
            lastText.includes("travelling from") ||
            lastText.includes("where you're traveling from");
          const expectsDestination =
            lastText.includes("where you want to go") ||
            lastText.includes("destination");

          const fromToMatch = text.match(/from\s+(.+?)\s+to\s+(.+)/i);
          const toMatch = text.match(/(.+?)\s+to\s+(.+)/i);

          let origin = fromToMatch?.[1]?.trim();
          let destination = fromToMatch?.[2]?.trim();
          if (!origin || !destination) {
            origin = toMatch?.[1]?.trim();
            destination = toMatch?.[2]?.trim();
          }

          if (expectsOrigin && !origin) origin = text;
          if (expectsDestination && !destination) destination = text;

          const mergedParts = [basePrompt];
          if (origin) mergedParts.push(`Origin: ${origin}`);
          if (destination) mergedParts.push(`Destination: ${destination}`);
          if (!origin && !destination) {
            mergedParts.push(`Additional details: ${text}`);
          }

          const mergedPrompt = mergedParts.join("\n").trim();
          pendingPromptRef.current = mergedPrompt;
          await doStart(mergedPrompt, true);
        } else {
          await doClarify(text);
        }
        break;
      case "modify_input":
        await doConfirmAssumptions(false, text);
        break;
      case "done":
        await doRefine(text);
        break;
      default:
        break;
    }
  };

  // ---- render helpers ----

  const displayMessages = useMemo(() => {
    return messages.filter((msg, idx) => {
      if (idx === 0) return true;
      const prev = messages[idx - 1];
      return !(
        msg.role === prev.role &&
        msg.content.trim() === prev.content.trim()
      );
    });
  }, [messages]);

  return (
    <div className="flex flex-col h-full overflow-hidden bg-[#FAFAF8]">
      {/* Messages Area */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-6 py-8 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
        <div className="max-w-3xl mx-auto space-y-6">
          {displayMessages.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="w-16 h-16 rounded-2xl bg-accent/10 flex items-center justify-center mb-6">
                <Sparkles className="w-8 h-8 text-accent" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">
                Start Planning Your Trip
              </h3>
              <p className="text-muted-foreground max-w-sm">
                Describe your dream destination and we&apos;ll help you plan the perfect itinerary.
              </p>
            </div>
          )}

          {displayMessages.map((message, index) => (
            <div
              key={`${message.id}-${index}`}
              className={`flex ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {message.role === "assistant" && !message.content ? null : (
                <div className={`flex items-start gap-3 max-w-[90%] ${
                  message.role === "user" ? "flex-row-reverse" : ""
                }`}>
                  {/* Avatar */}
                  {message.role === "user" ? (
                    <div className="mt-0.5 shrink-0 w-8 h-8 rounded-full overflow-hidden bg-accent border-2 border-accent">
                      {user?.picture_url ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={user.picture_url}
                          alt={user.name || user.email}
                          className="w-full h-full object-cover"
                          referrerPolicy="no-referrer"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-white text-xs font-semibold">
                          {(user?.name?.[0] || user?.email?.[0] || "U").toUpperCase()}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="mt-0.5 shrink-0 w-8 h-8 rounded-full flex items-center justify-center overflow-hidden">
                      <img
                        src="/favicon.ico"
                        alt="Planfirst"
                        width={40}
                        height={40}
                        className="w-10 h-10 object-contain"
                      />
                    </div>
                  )}

                  {/* Message Bubble */}
                  <div className={`
                    rounded-2xl px-5 py-4 shadow-sm
                    ${message.role === "user"
                      ? "bg-accent text-white rounded-br-md"
                      : "bg-white border border-border/20 text-foreground rounded-bl-md"
                    }
                  `}>
                    {message.role === "assistant" ? (
                      <MarkdownRenderer content={message.content} />
                    ) : (
                      <p className="text-[15px] leading-relaxed">{message.content}</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Action Cards */}
          {!isLoading && nextAction === "proceed_continue" && (
            <div className="flex justify-start pl-11">
              <div className="bg-white rounded-2xl border border-border/30 shadow-sm p-5 max-w-md">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-xl bg-green-50 flex items-center justify-center">
                    <Plane className="w-5 h-5 text-green-600" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-foreground">Ready to move forward?</h4>
                    <p className="text-sm text-muted-foreground">Your trip looks feasible!</p>
                  </div>
                </div>
                <Button
                  onClick={() => {
                    addMessage("user", "Looks good, continue!");
                    setNextAction("text_input");
                    doProceed(true, true);
                  }}
                  className="w-full bg-accent hover:bg-accent/90 text-white rounded-xl"
                >
                  Continue to Planning
                </Button>
              </div>
            </div>
          )}

          {/* Loading Indicator */}
          {isLoading && !streamingHasDelta && (
            <div className="flex justify-start pl-11">
              <div className="flex items-center gap-3 bg-white rounded-2xl border border-border/30 shadow-sm px-5 py-4">
                <div className="w-8 h-8 rounded-full bg-accent/10 flex items-center justify-center">
                  <Loader2 className="w-4 h-4 text-accent animate-spin" />
                </div>
                <p className="text-sm text-muted-foreground">{loadingText}</p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-border/20 px-4 sm:px-6 py-5">
        <div className="max-w-3xl mx-auto">
          {isLoading ? (
            <div className="relative">
              <div className="relative flex items-center">
                <div className="w-full h-14 pl-6 pr-14 rounded-full bg-muted/50 flex items-center">
                  <span className="text-muted-foreground text-sm">Waiting for response...</span>
                </div>
                <div className="absolute right-2 w-10 h-10 rounded-full bg-muted flex items-center justify-center">
                  <ArrowUp className="w-5 h-5 text-muted-foreground" />
                </div>
              </div>
            </div>
          ) : nextAction === "text_input" || nextAction === "modify_input" ? (
            <form onSubmit={handleSubmit} className="relative">
              <div className="relative flex items-center">
                <Input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={
                    nextAction === "modify_input"
                      ? "Describe what you'd like to change..."
                      : "Type your message..."
                  }
                  className="w-full h-14 pl-6 pr-14 rounded-full bg-muted/30 border-transparent shadow-sm focus:bg-white focus:ring-2 focus:ring-accent/20 transition-all text-[15px]"
                />
                <button
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  className="absolute right-2 w-10 h-10 rounded-full bg-accent text-white flex items-center justify-center hover:bg-accent/90 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                >
                  <ArrowUp className="w-5 h-5" />
                </button>
              </div>
              {nextAction === "modify_input" && (
                <div className="flex justify-center mt-3">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setNextAction("assumptions_confirm")}
                    className="h-10 px-4 rounded-full"
                  >
                    Cancel
                  </Button>
                </div>
              )}
            </form>
          ) : nextAction === "proceed_confirm" ? (
            <div className="flex gap-3 justify-center">
              <div className="bg-amber-50 rounded-xl p-4 flex items-center gap-4">
                <div className="w-10 h-10 rounded-xl bg-amber-100 flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-amber-600" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-amber-900">This trip has some risks</p>
                  <p className="text-xs text-amber-700">Review before proceeding</p>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => {
                      addMessage("user", "Let me reconsider.");
                      doProceed(false);
                    }}
                    className="rounded-xl"
                  >
                    Reconsider
                  </Button>
                  <Button
                    onClick={() => {
                      addMessage("user", "Let's proceed anyway.");
                      doProceed(true);
                    }}
                    className="rounded-xl bg-accent hover:bg-accent/90"
                  >
                    Proceed
                  </Button>
                </div>
              </div>
            </div>
          ) : nextAction === "assumptions_confirm" ? (
            <div className="flex gap-3 justify-center">
              <div className="bg-white rounded-2xl border border-border/30 shadow-sm p-5 max-w-lg w-full">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-xl bg-accent/10 flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-accent" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-foreground">Review Assumptions</h4>
                    <p className="text-sm text-muted-foreground">Everything look correct?</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <Button
                    onClick={() => {
                      addMessage("user", "Looks good — go ahead and plan!");
                      doConfirmAssumptions(true);
                    }}
                    className="flex-1 rounded-xl bg-accent hover:bg-accent/90 text-white"
                  >
                    Continue
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setNextAction("modify_input")}
                    className="flex-1 rounded-xl hover:bg-accent hover:text-white hover:border-accent transition-all"
                  >
                    Make Changes
                  </Button>
                </div>
              </div>
            </div>
          ) : nextAction === "done" ? (
            <div className="space-y-4">
              <form onSubmit={handleSubmit} className="relative">
                <div className="relative flex items-center">
                  <Input
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Request changes like 'add more hiking' or 'make it cheaper'..."
                    className="w-full h-14 pl-6 pr-14 rounded-full bg-muted/30 border-transparent shadow-sm focus:bg-white focus:ring-2 focus:ring-accent/20 transition-all text-[15px]"
                  />
                  <button
                    type="submit"
                    disabled={!input.trim() || isLoading}
                    className="absolute right-2 w-10 h-10 rounded-full bg-accent text-white flex items-center justify-center hover:bg-accent/90 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                  >
                    <ArrowUp className="w-5 h-5" />
                  </button>
                </div>
              </form>
              <p className="text-xs text-muted-foreground text-center">
                Your trip plan is ready! Continue refining or start a new plan.
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}


