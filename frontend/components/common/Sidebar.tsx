"use client";

import { useMemo, useState } from "react";
import {
  Check,
  Headphones,
  Inbox,
  MoreHorizontal,
  Plus,
  Search,
  Star,
  Users,
  X,
} from "lucide-react";
import { Avatar } from "@/components/common/Avatar";
import { useConversations } from "@/contexts/ConversationContext";

export default function Sidebar() {
  const {
    conversations,
    activeId,
    setActiveId,
    createConversation,
    sidebarOpen,
    setSidebarOpen,
  } = useConversations();
  const [search, setSearch] = useState("");

  const filteredThreads = useMemo(
    () =>
      conversations.filter((thread) =>
        `${thread.title} ${thread.customer}`
          .toLowerCase()
          .includes(search.toLowerCase()),
      ),
    [conversations, search],
  );

  const active =
    conversations.find((thread) => thread.id === activeId) ?? conversations[0];

  async function newConversation() {
    await createConversation({
      title: "New customer conversation",
      customer: "Unassigned",
      status: "Open",
    });
  }

  return (
    <>
      {sidebarOpen && (
        <button
          aria-label="Close navigation"
          className="fixed inset-0 z-20 bg-foreground/20 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        className={`fixed inset-y-0 z-30 flex w-76 flex-col border-r border-border bg-sidebar/24 aurora-surface transition-[left] lg:relative lg:left-0 ${sidebarOpen ? "left-0" : "-left-full"}`}
      >
        <div className="flex h-18 items-center justify-between border-b border-border px-5">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <Headphones className="size-4" />
            </div>
            <div>
              <p className="text-sm font-semibold tracking-tight">Relay Desk</p>
              <p className="text-xs text-muted-foreground">Support workspace</p>
            </div>
          </div>
          <button
            className="rounded-lg p-2 text-muted-foreground hover:bg-accent hover:text-foreground lg:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="size-4" />
          </button>
        </div>
        <div className="flex flex-col gap-5 p-4">
          <button
            onClick={newConversation}
            className="flex h-10 items-center justify-center gap-2 rounded-lg bg-primary text-sm font-medium text-primary-foreground shadow-sm transition hover:opacity-90"
          >
            <Plus className="size-4" /> New conversation{" "}
            <kbd className="ml-1 hidden rounded border border-primary-foreground/20 px-1.5 py-0.5 text-[10px] opacity-70 sm:inline">
              ⌘ K
            </kbd>
          </button>
          <nav className="flex flex-col gap-1" aria-label="Workspace navigation">
            {[
              ["Inbox", Inbox],
              ["Assigned to me", Users],
              ["Starred", Star],
              ["Resolved", Check],
            ].map(([label, Icon]) => (
              <button
                key={label as string}
                className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm ${label === "Inbox" ? "bg-accent font-medium text-foreground" : "text-muted-foreground hover:bg-accent hover:text-foreground"}`}
              >
                <Icon className="size-4" />
                {label as string}
                <span className="ml-auto text-xs text-muted-foreground">
                  {label === "Inbox" ? "12" : ""}
                </span>
              </button>
            ))}
          </nav>
        </div>
        <div className="flex min-h-0 flex-1 flex-col border-t border-border pt-4">
          <div className="flex items-center justify-between px-5 pb-3">
            <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              Recent conversations
            </p>
            <button className="text-muted-foreground hover:text-foreground">
              <MoreHorizontal className="size-4" />
            </button>
          </div>
          <label className="mx-4 mb-3 flex h-9 items-center gap-2 rounded-lg border border-input bg-background px-3 text-muted-foreground">
            <Search className="size-4" />
            <span className="sr-only">Search conversations</span>
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search conversations"
              className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            />
          </label>
          <div className="flex min-h-0 flex-col gap-1 overflow-y-auto px-3">
            {filteredThreads.map((thread) => (
              <button
                key={thread.id}
                onClick={() => {
                  setActiveId(thread.id);
                  setSidebarOpen(false);
                }}
                className={`flex items-start gap-3 rounded-xl p-3 text-left transition ${thread.id === active?.id ? "bg-primary text-primary-foreground shadow-sm" : "hover:bg-accent"}`}
              >
                <Avatar
                  name={thread.customer}
                  tone={
                    thread.id === active?.id
                      ? "bg-signal-soft text-primary"
                      : "bg-muted text-muted-foreground"
                  }
                />
                <span className="min-w-0 flex-1">
                  <span className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-medium">
                      {thread.title}
                    </span>
                    <span
                      className={`size-1.5 shrink-0 rounded-full ${thread.status === "Resolved" ? "bg-muted-foreground" : "bg-signal"}`}
                    />
                  </span>
                  <span
                    className={`mt-1 block truncate text-xs ${thread.id === active?.id ? "text-primary-foreground/65" : "text-muted-foreground"}`}
                  >
                    {thread.customer} · {thread.status}
                  </span>
                </span>
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-3 border-t border-border p-4">
          <Avatar name="Jordan Lee" tone="bg-primary" />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">Jordan Lee</p>
            <p className="text-xs text-muted-foreground">Support lead</p>
          </div>
          <button className="rounded-lg p-2 text-muted-foreground hover:bg-accent hover:text-foreground">
            <MoreHorizontal className="size-4" />
          </button>
        </div>
      </aside>
    </>
  );
}
