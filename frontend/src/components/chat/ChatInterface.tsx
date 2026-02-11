"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { UserMenu } from "@/components/layout/UserMenu";
import {
  startTripStream,
  clarifyTripStream,
  proceedTripStream,
  confirmAssumptionsStream,
  refineTripStream,
  ApiError,
} from "@/lib/api";
import type { AuthUser, StreamEvent, StreamMeta } from "@/lib/api";

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
  initialPrompt: string;
  user?: AuthUser | null;
  loading?: boolean;
  onSignOut?: () => void;
}

/**
 * What kind of input the UI is waiting for from the user.
 *
 *  text_input        – free-text (clarification answers / refinement)
 *  proceed_confirm   – proceed / reconsider buttons (high-risk feasibility)
 *  proceed_continue  – "continue" button (low-risk feasibility)
 *  assumptions_confirm – confirm / modify buttons
 *  modify_input      – free-text for assumption modifications
 *  done              – conversation complete, offer refinement
 */
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
  initialPrompt,
  user,
  loading,
  onSignOut,
}: ChatInterfaceProps) {
  const idCounterRef = useRef(0);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("Thinking...");
  const [tripId, setTripId] = useState<string | null>(null);
  const [nextAction, setNextAction] = useState<NextAction>("text_input");
  const [hasHighRisk, setHasHighRisk] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [restored, setRestored] = useState(false);
  const [streamingHasDelta, setStreamingHasDelta] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const startedRef = useRef(false);
  const startInFlightRef = useRef(false);
  const actionInFlightRef = useRef(false);
  const initialMessageIdRef = useRef<string | null>(null);
  const pendingPromptRef = useRef(initialPrompt);
  const storageKey = `plandrift_chat_${initialPrompt}`;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Focus input when it becomes available
  useEffect(() => {
    if (!isLoading && nextAction === "text_input") {
      inputRef.current?.focus();
    }
  }, [isLoading, nextAction]);

  useEffect(() => {
    pendingPromptRef.current = initialPrompt;
  }, [initialPrompt]);

  // ---- helpers ----

  const renderBoldInline = (text: string) => {
    const parts = text.split("**");
    return parts.map((part, idx) =>
      idx % 2 === 1 ? (
        <strong key={idx} className="font-semibold text-black">
          {part}
        </strong>
      ) : (
        <span key={idx}>{part}</span>
      ),
    );
  };

  const renderInline = (text: string) => {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    const parts = text.split(urlRegex);
    return parts.map((part, idx) => {
      if (part.startsWith("http")) {
        return (
          <a
            key={`link-${idx}`}
            href={part}
            target="_blank"
            rel="noreferrer"
            className="text-slate-900 underline underline-offset-4"
          >
            {part}
          </a>
        );
      }
      return <span key={`text-${idx}`}>{renderBoldInline(part)}</span>;
    });
  };

  const renderFormatted = (text: string) => {
    const lines = text.split("\n");
    return (
      <div className="space-y-2">
        {lines.map((line, idx) => {
          const trimmed = line.trim();
          if (!trimmed) return <div key={idx} className="h-2" />;

          if (
            trimmed.startsWith("• ") ||
            trimmed.startsWith("- ") ||
            trimmed.startsWith("→ ")
          ) {
            const bullet = trimmed.startsWith("→ ") ? "→" : "•";
            const content = trimmed.replace(/^•\s|^-\s|^→\s/, "");
            return (
              <div key={idx} className="flex gap-2">
                <span className="text-slate-400">{bullet}</span>
                <span>{renderInline(content)}</span>
              </div>
            );
          }

          return (
            <div key={idx} className="text-sm leading-relaxed">
              {renderInline(trimmed)}
            </div>
          );
        })}
      </div>
    );
  };

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
      let hasDelta = false;
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
                    content: hasDelta ? msg.content + cleaned : cleaned,
                  }
                : msg,
            ),
          );
          hasDelta = true;
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
      const message =
        err instanceof ApiError
          ? err.detail
          : "Something went wrong. Please try again.";
      setError(message);
      addMessage("assistant", `Error: ${message}`);
    },
    [addMessage],
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
      setError(null);

      try {
        const prompt = (overridePrompt || initialPrompt).trim();
        if (!prompt) {
          throw new Error("Please enter a trip request to get started.");
        }
        pendingPromptRef.current = prompt;
        if (messages.length === 0) {
          addMessage("user", prompt);
        }
        const stream = await startTripStream(prompt);
        await consumeStream(stream, () => "text_input");
      } catch (err) {
        handleError(err);
      } finally {
        setIsLoading(false);
        startInFlightRef.current = false;
      }
  }, [initialPrompt, consumeStream, handleError, addMessage, messages.length]);

  const doClarify = useCallback(
    async (answers: string) => {
      if (!tripId) return;
      if (actionInFlightRef.current) return;
      actionInFlightRef.current = true;
      setIsLoading(true);
      setLoadingText("Analyzing feasibility...");
      setError(null);

      try {
        const stream = await clarifyTripStream(tripId, answers);
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
    [tripId, nextAction, consumeStream, handleError],
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
      setError(null);

      try {
        const stream = await proceedTripStream(tripId, proceed);
        await consumeStream(stream, (meta) =>
          meta?.phase === "assumptions"
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
        "Researching prices and building your itinerary — this may take a minute...",
      );
      setError(null);

      try {
        const stream = await confirmAssumptionsStream(
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
      setError(null);

      try {
        const stream = await refineTripStream(tripId, refinementType);
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

  // ---- start on mount ----

  useEffect(() => {
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
  }, [storageKey]);

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
    if (!tripId && messages.length) {
      const userText = messages
        .filter((msg) => msg.role === "user")
        .map((msg) => msg.content)
        .join("\n");
      if (userText.trim()) {
        pendingPromptRef.current = userText.trim();
      }
    }
  }, [messages, tripId, nextAction, hasHighRisk, storageKey]);

  useEffect(() => {
    return () => {
      localStorage.removeItem(storageKey);
    };
  }, [storageKey]);

  // ---- form submission (routes to correct phase handler) ----

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;

    addMessage("user", text);
    setInput("");

    switch (nextAction) {
      case "text_input":
        // During clarification or general text input
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
        // User is providing modifications to assumptions
        await doConfirmAssumptions(false, text);
        break;
      case "done":
        // User is requesting a refinement
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
    <div className="flex flex-col h-screen overflow-hidden bg-white">
      {/* Header */}
      <div className="border-b border-black/10 bg-white/80 backdrop-blur">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <a href="/" className="text-lg font-semibold tracking-tight">
            Plandrift
          </a>
          {onSignOut && (
            <UserMenu user={user || null} loading={loading} onSignOut={onSignOut} />
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-8">
        <div className="max-w-3xl mx-auto space-y-6">
          {displayMessages.map((message, index) => (
            <div
              key={`${message.id}-${index}`}
              className={`flex ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {message.role === "assistant" && !message.content ? null : (
              <div className="flex items-start gap-3 max-w-[92%]">
                {message.role !== "user" && (
                  <div className="mt-1 size-8 shrink-0 rounded-full border border-black/10 bg-white flex items-center justify-center text-xs font-semibold text-black">
                    PD
                  </div>
                )}
                <Card
                  className={`p-4 shadow-sm ${
                    message.role === "user"
                      ? "bg-black text-white border border-black"
                      : "bg-white text-black border border-black/10"
                  }`}
                >
                  <div className="whitespace-pre-wrap text-sm leading-relaxed">
                    {message.role === "assistant"
                      ? renderFormatted(message.content)
                      : message.content}
                  </div>
                </Card>
              </div>
              )}
            </div>
          ))}

          {!isLoading && nextAction === "proceed_continue" && (
            <div className="flex justify-start">
              <div className="flex items-start gap-3 max-w-[92%]">
                <div className="mt-1 size-8 shrink-0 rounded-full border border-black/10 bg-white flex items-center justify-center text-xs font-semibold text-black">
                  PD
                </div>
                <Card className="p-4 shadow-sm bg-white text-black border border-black/10">
                  <div className="text-sm text-slate-600">
                    Ready to move forward?
                  </div>
                  <div className="mt-3">
                  <Button
                    onClick={() => {
                      addMessage("user", "Looks good, continue!");
                      setNextAction("text_input");
                      doProceed(true, true);
                    }}
                    disabled={isLoading}
                    className="rounded-full bg-black text-white hover:bg-black/90"
                  >
                    Continue to Planning
                  </Button>
                  </div>
                </Card>
              </div>
            </div>
          )}

          {/* Loading indicator */}
          {isLoading && !streamingHasDelta && (
            <div className="flex justify-start">
              <div className="flex items-start gap-3 max-w-[92%]">
                <div className="mt-1 size-8 shrink-0 rounded-full border border-black/10 bg-white flex items-center justify-center text-xs font-semibold text-black">
                  PD
                </div>
                <Card className="bg-white border border-black/10 p-4 shadow-sm">
                  <p className="text-xs text-slate-600">{loadingText}</p>
                </Card>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Error banner */}
      {/* Action bar — adapts to current phase */}
      <div className="bg-white border-t border-black/10 px-6 py-4">
        <div className="max-w-3xl mx-auto">
          {isLoading ? (
            // Disabled input while loading
            <div className="flex gap-2 items-center">
              <Input
                disabled
                placeholder="Waiting for response..."
                className="flex-1 bg-white border-black/10 text-black opacity-50 rounded-full"
              />
              <Button disabled>Send</Button>
            </div>
          ) : nextAction === "text_input" || nextAction === "modify_input" ? (
            // Free-text input (clarification, modifications, refinements)
            <form onSubmit={handleSubmit} className="flex gap-2">
              <Input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  nextAction === "modify_input"
                    ? "Describe what you'd like to change..."
                    : "Type your response..."
                }
                className="flex-1 bg-white border-black/10 text-black rounded-full"
              />
              <Button
                type="submit"
                disabled={!input.trim()}
                className="rounded-full bg-black text-white hover:bg-black/90"
              >
                Send
              </Button>
              {nextAction === "modify_input" && (
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setNextAction("assumptions_confirm")}
                >
                  Cancel
                </Button>
              )}
            </form>
          ) : nextAction === "proceed_confirm" ? (
            // High-risk feasibility — proceed or reconsider
            <div className="flex gap-3 justify-center">
              <Button
                onClick={() => {
                  addMessage("user", "Let's proceed anyway.");
                  doProceed(true);
                }}
                disabled={isLoading}
              >
                Proceed Anyway
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  addMessage("user", "Let me reconsider.");
                  doProceed(false);
                }}
                disabled={isLoading}
              >
                Reconsider
              </Button>
            </div>
          ) : nextAction === "assumptions_confirm" ? (
            // Confirm or modify assumptions
            <div className="flex gap-3 justify-center">
              <Button
                onClick={() => {
                  addMessage("user", "Looks good — go ahead and plan!");
                  doConfirmAssumptions(true);
                }}
                disabled={isLoading}
              >
                Looks Good, Plan It!
              </Button>
              <Button
                variant="outline"
                onClick={() => setNextAction("modify_input")}
                disabled={isLoading}
              >
                I Want Changes
              </Button>
            </div>
          ) : nextAction === "done" ? (
            // Plan ready — refinement or finish
            <form onSubmit={handleSubmit} className="space-y-2">
              <div className="flex gap-2">
                <Input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Request a change, e.g. 'make it cheaper' or 'add more hiking'..."
                  className="flex-1 bg-white border-black/10 text-black rounded-full"
                />
                <Button
                  type="submit"
                  disabled={!input.trim()}
                  className="rounded-full bg-black text-white hover:bg-black/90"
                >
                  Refine
                </Button>
              </div>
              <p className="text-xs text-slate-600 text-center">
                Happy with your plan? You can continue refining anytime.
              </p>
            </form>
          ) : null}
        </div>
      </div>
    </div>
  );
}
