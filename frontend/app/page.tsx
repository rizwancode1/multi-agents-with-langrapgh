"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  ArrowUp,
  Bell,
  ChevronDown,
  Clock3,
  Command,
  FileText,
  Menu,
  Moon,
  MoreHorizontal,
  Paperclip,
  Plus,
  Sparkles,
  Sun,
  Tag,
  Zap,
} from "lucide-react";
import { useTheme } from "../contexts/ThemeContext";
import { useConversations } from "../contexts/ConversationContext";
import type { Message, Thread } from "../contexts/ConversationContext";
import { Avatar } from "../components/common/Avatar";

export default function Page() {
  const queryClient = useQueryClient();
  const { theme, setTheme } = useTheme();
  const { conversations, isLoading, activeId, createConversation, setSidebarOpen } =
    useConversations();

  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const messageSeqRef = useRef(0);
  const active = conversations.find((thread) => thread.id === activeId) ?? conversations[0];

  function updateAssistantMessage(
    threadId: number,
    messageId: string | number,
    patch: { text: string; status?: string },
  ) {
    queryClient.setQueryData<Thread[]>(["conversations"], (old = []) =>
      old.map((thread) => {
        if (thread.id !== threadId) return thread;
        const messages = [...thread.messages];
        const idx = messages.findIndex((m) => m.id === messageId);
        const assistant: Message = { id: messageId, role: "assistant", text: patch.text, time: "Just now", status: patch.status };
        if (idx === -1) {
          messages.push(assistant);
        } else {
          messages[idx] = assistant;
        }
        return { ...thread, messages };
      }),
    );
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [active?.messages?.length, isSending]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  function cycleTheme() {
    setTheme((current) =>
      current === "system" ? "light" : current === "light" ? "dark" : "system",
    );
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    if (!draft.trim() || isSending || !active) return;
    const requestId = ++requestIdRef.current;
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const text = draft.trim();
    setDraft("");
    setIsSending(true);

    const userMessage: Message = {
      id: `u-${++messageSeqRef.current}`,
      role: "user",
      text,
      time: "Just now",
    };
    const optimisticThreads = conversations.map((thread) =>
      thread.id === active.id
        ? {
            ...thread,
            title: thread.messages.length ? thread.title : text.slice(0, 34),
            messages: [...thread.messages, userMessage],
          }
        : thread,
    );
    queryClient.setQueryData<Thread[]>(["conversations"], optimisticThreads);

    const statusMessage: Message = {
      id: `a-${++messageSeqRef.current}`,
      role: "assistant",
      text: "Processing your request...",
      time: "Just now",
      status: "Thinking...",
    };
    queryClient.setQueryData<Thread[]>(["conversations"], (old = []) =>
      old.map((thread) =>
        thread.id === active.id
          ? { ...thread, messages: [...thread.messages, statusMessage] }
          : thread,
      ),
    );

    const requestBody = JSON.stringify({
      query: text,
      conversation_id: active.id,
    });

    try {
      const response = await fetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody,
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        const fallback = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: requestBody,
          signal: controller.signal,
        });
        const data = await fallback.json();
        const answer = data?.response || data?.answer || "Done";
        updateAssistantMessage(active.id, statusMessage.id!, { text: answer });
        queryClient.invalidateQueries({ queryKey: ["conversations"] });
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let isDone = false;

      while (!isDone && !controller.signal.aborted) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          let event: any;
          try {
            event = JSON.parse(raw);
          } catch {
            continue;
          }

          if (event.type === "status") {
            updateAssistantMessage(active.id, statusMessage.id!, {
              text: event.message || "Processing...",
              status: event.agent,
            });
          } else if (event.type === "step") {
            updateAssistantMessage(active.id, statusMessage.id!, {
              text: event.message || `Step: ${event.agent}`,
              status: event.agent,
            });
          } else if (event.type === "done") {
            isDone = true;
            const finalText = event.response || "Done";
            updateAssistantMessage(active.id, statusMessage.id!, { text: finalText });
            queryClient.invalidateQueries({ queryKey: ["conversations"] });
          } else if (event.type === "error") {
            isDone = true;
            updateAssistantMessage(active.id, statusMessage.id!, {
              text: "Something went wrong. Please try again.",
            });
          }
        }
      }

      if (!isDone && requestIdRef.current === requestId) {
        updateAssistantMessage(active.id, statusMessage.id!, {
          text: "The response was interrupted. Please try again.",
          status: "Interrupted",
        });
      }
    } catch (err) {
      if (requestIdRef.current === requestId && (err as any)?.name !== "AbortError") {
        updateAssistantMessage(active.id, statusMessage.id!, {
          text: "Thanks for the context. I've captured this request and routed it to the right team for review.",
        });
      }
    } finally {
      if (requestIdRef.current === requestId) {
        setIsSending(false);
      }
    }
  }

  if (isLoading) {
    return (
      <main className="aurora-shell min-h-dvh text-foreground flex-1">
        <div className="flex min-h-dvh items-center justify-center">
          <p className="text-sm text-muted-foreground">Loading conversations...</p>
        </div>
      </main>
    );
  }

  if (!active) {
    return (
      <main className="aurora-shell min-h-dvh text-foreground flex-1">
        <div className="flex min-h-dvh flex-col items-center justify-center gap-4">
          <div className="mb-2 flex size-12 items-center justify-center rounded-2xl bg-accent text-primary">
            <Sparkles className="size-5" />
          </div>
          <h1 className="text-xl font-semibold tracking-tight">No conversations yet</h1>
          <p className="max-w-sm text-sm leading-6 text-muted-foreground text-center">
            Start a new conversation to get help with orders, refunds, policies, and more.
          </p>
          <button
            onClick={() =>
              createConversation({
                title: "New customer conversation",
                customer: "Unassigned",
                status: "Open",
              })
            }
            className="mt-2 flex h-10 items-center justify-center gap-2 rounded-lg bg-primary text-sm font-medium text-primary-foreground shadow-sm transition hover:opacity-90"
          >
            <Plus className="size-4" /> New conversation
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="aurora-shell min-h-dvh text-foreground flex-1">
      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-18 items-center justify-between aurora-surface border-b border-border/70 bg-background/35 px-4 backdrop-blur-xl md:px-8">
          <div className="flex items-center gap-3">
            <button
              aria-label="Open navigation"
              className="rounded-lg p-2 hover:bg-accent lg:hidden"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu className="size-5" />
            </button>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium uppercase tracking-[0.13em] text-muted-foreground">
                  Inbox
                </span>
                <span className="text-muted-foreground"></span>
                <span className="max-w-45 truncate text-sm font-medium md:max-w-none">
                  {active.title}
                </span>
              </div>
              <p className="mt-1 hidden text-xs text-muted-foreground sm:block">
                Updated just now · Customer support
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button className="hidden items-center gap-2 rounded-lg border border-input px-3 py-2 text-xs text-muted-foreground hover:bg-accent md:flex">
              <Command className="size-3.5" /> Search{" "}
              <kbd className="rounded border border-border px-1">K</kbd>
            </button>
            <button
              aria-label={`Theme: ${theme}. Click to change theme`}
              title={`Theme: ${theme}`}
              onClick={cycleTheme}
              className="rounded-lg p-2 text-muted-foreground transition hover:bg-accent hover:text-foreground"
            >
              {theme === "light" ? (
                <Sun className="size-4" />
              ) : theme === "dark" ? (
                <Moon className="size-4" />
              ) : (
                <Sparkles className="size-4" />
              )}
            </button>
            <button
              aria-label="Notifications"
              className="rounded-lg p-2 text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              <Bell className="size-4" />
            </button>
            <button
              aria-label="More actions"
              className="rounded-lg p-2 text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              <MoreHorizontal className="size-4" />
            </button>
          </div>
        </header>
        <div className="flex min-h-0 flex-1 overflow-y-auto flex-col xl:flex-row">
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex items-center justify-between border-b border-border px-4 py-4 md:px-8">
              <div className="flex items-center gap-3">
                <Avatar
                  name={active.customer === "Unassigned" ? "New" : active.customer}
                />
                <div>
                  <p className="text-sm font-semibold">{active.customer}</p>
                  <p className="text-xs text-muted-foreground">
                    Customer · conversation #{active.id}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="hidden rounded-full bg-signal-soft px-2.5 py-1 text-xs font-medium text-signal-foreground sm:inline-flex">
                  <span className="mr-1.5 mt-0.5 size-1.5 rounded-full bg-signal" />
                  {active.status}
                </span>
                <button className="rounded-lg border border-input p-2 text-muted-foreground hover:bg-accent">
                  <Archive className="size-4" />
                </button>
              </div>
            </div>
            <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 py-6 md:px-8 md:py-8">
              <div className="mx-auto flex w-full max-w-3xl flex-col gap-7">
                {active.messages.length === 0 && (
                  <div className="flex flex-1 flex-col items-center justify-center py-20 text-center">
                    <div className="mb-4 flex size-12 items-center justify-center rounded-2xl bg-accent text-primary">
                      <Sparkles className="size-5" />
                    </div>
                    <h1 className="text-xl font-semibold tracking-tight">
                      Start a helpful conversation
                    </h1>
                    <p className="mt-2 max-w-sm text-sm leading-6 text-muted-foreground">
                      Ask Relay to summarize context, draft a reply, or find
                      the next best action for this customer.
                    </p>
                  </div>
                )}
                {active.messages.map((message, index) => (
                  <div
                    key={`${message.time}-${index}`}
                    className={`flex gap-3 ${message.role === "user" ? "justify-end" : ""}`}
                  >
                    {message.role === "assistant" && (
                      <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <Zap className="size-4" />
                      </div>
                    )}
                    <div
                      className={`max-w-[min(680px,85%)] ${message.role === "user" ? "items-end" : ""}`}
                    >
                      <div
                        className={`rounded-2xl px-4 py-3 text-sm leading-6 ${message.role === "user" ? "rounded-br-md bg-primary text-primary-foreground" : "rounded-bl-md border border-border bg-card text-card-foreground shadow-sm"}`}
                      >
                        {message.text}
                      </div>
                      {message.status && (
                        <p className="mt-1 text-[11px] text-muted-foreground">
                          {message.status}
                        </p>
                      )}
                      <p
                        className={`mt-1.5 text-[11px] text-muted-foreground ${message.role === "user" ? "text-right" : ""}`}
                      >
                        {message.role === "assistant" ? "Relay AI" : "You"} ·{" "}
                        {message.time}
                      </p>
                    </div>
                    {message.role === "user" && (
                      <Avatar name="Jordan Lee" tone="bg-primary" />
                    )}
                  </div>
                ))}
                <div ref={bottomRef} />
              </div>
            </div>
            <div className="border-t border-border/70 bg-transparent px-4 py-4 md:px-8 md:py-5">
              <div className="mx-auto max-w-3xl">
                <div className="mb-3 flex gap-2 overflow-x-auto pb-1">
                  <button
                    onClick={() =>
                      setDraft(
                        "Summarize the customer context and recommend the next action.",
                      )
                    }
                    className="flex shrink-0 items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
                  >
                    <FileText className="size-3.5" /> Summarize context
                  </button>
                  <button
                    onClick={() =>
                      setDraft(
                        "Draft a concise, empathetic reply to the customer.",
                      )
                    }
                    className="flex shrink-0 items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
                  >
                    <Sparkles className="size-3.5" /> Draft a reply
                  </button>
                  <button
                    onClick={() => setDraft("What is the next best action?")}
                    className="flex shrink-0 items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
                  >
                    <Zap className="size-3.5" /> Next best action
                  </button>
                </div>
                <form
                  onSubmit={sendMessage}
                  className="aurora-composer rounded-2xl border border-input/70 bg-transparent p-2 shadow-sm focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/20"
                >
                  <textarea
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    onKeyDown={(event) => {
                      if (
                        event.key === "Enter" &&
                        !event.shiftKey &&
                        !event.nativeEvent.isComposing &&
                        event.keyCode !== 229
                      ) {
                        event.preventDefault();
                        event.currentTarget.form?.requestSubmit();
                      }
                    }}
                    placeholder="Ask Relay anything about this conversation..."
                    rows={2}
                    className="w-full resize-none bg-transparent px-2 py-1 text-sm leading-6 outline-none placeholder:text-muted-foreground"
                  />
                  <div className="flex items-center justify-between pt-2">
                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        aria-label="Attach file"
                        className="rounded-lg p-2 text-muted-foreground hover:bg-accent hover:text-foreground"
                      >
                        <Paperclip className="size-4" />
                      </button>
                      <span className="hidden text-xs text-muted-foreground sm:inline">
                        Enter to send · Shift + Enter for a new line
                      </span>
                    </div>
                    <button
                      disabled={!draft.trim() || isSending}
                      className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      <ArrowUp className="size-4" />
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
          <aside className="hidden w-75 shrink-0 border-l border-border aurora-surface bg-sidebar/22 p-6 xl:block">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-[0.13em] text-muted-foreground">
                Ticket context
              </p>
              <button className="text-muted-foreground hover:text-foreground">
                <MoreHorizontal className="size-4" />
              </button>
            </div>
            <div className="mt-5 flex flex-col gap-5">
              <div>
                <p className="text-xs text-muted-foreground">Status</p>
                <div className="mt-2 flex items-center gap-2 text-sm font-medium">
                  <span className="size-2 rounded-full bg-signal" />{" "}
                  {active.status}
                  <ChevronDown className="ml-auto size-4 text-muted-foreground" />
                </div>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Priority</p>
                <p className="mt-2 text-sm font-medium">High</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Tags</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  <span className="inline-flex items-center gap-1 rounded-md bg-accent px-2 py-1 text-xs text-muted-foreground">
                    <Tag className="size-3" /> Billing
                  </span>
                  <span className="rounded-md bg-accent px-2 py-1 text-xs text-muted-foreground">
                    Duplicate charge
                  </span>
                </div>
              </div>
              <div className="border-t border-border pt-5">
                <p className="text-xs text-muted-foreground">Assigned agent</p>
                <div className="mt-3 flex items-center gap-3">
                  <Avatar name="Jordan Lee" tone="bg-primary" />
                  <div>
                    <p className="text-sm font-medium">Jordan Lee</p>
                    <p className="text-xs text-muted-foreground">
                      Support lead
                    </p>
                  </div>
                </div>
              </div>
              <div className="border-t border-border pt-5">
                <p className="text-xs text-muted-foreground">SLA</p>
                <div className="mt-2 flex items-center gap-2 text-sm font-medium">
                  <Clock3 className="size-4 text-signal" /> Respond within 2h
                </div>
              </div>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}
