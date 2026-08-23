"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export type Message = {
  id?: string | number;
  role: "user" | "assistant";
  text: string;
  time: string;
  status?: string;
};

export type Thread = {
  id: number;
  title: string;
  customer: string;
  status: string;
  messages: Message[];
};

interface ConversationContextValue {
  conversations: Thread[];
  isLoading: boolean;
  activeId: number | null;
  setActiveId: (id: number | null) => void;
  createConversation: (data: {
    title: string;
    customer?: string;
    status?: string;
  }) => Promise<Thread>;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

const ConversationContext = createContext<ConversationContextValue | undefined>(
  undefined,
);

type BackendMessage = {
  id: number;
  role: "user" | "assistant";
  text: string;
  time: string;
  status?: string | null;
};

type BackendConversation = {
  id: number;
  title: string;
  customer: string;
  status: string;
  thread_id?: string | null;
  messages: BackendMessage[];
};

function conversationIdFromUrl(): number | null {
  if (typeof window === "undefined") return null;
  const param = new URLSearchParams(window.location.search).get("conversation");
  const id = param ? Number(param) : NaN;
  return Number.isInteger(id) && id > 0 ? id : null;
}

function updateUrlConversationId(conversationId: number | null) {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (conversationId) {
    url.searchParams.set("conversation", String(conversationId));
  } else {
    url.searchParams.delete("conversation");
  }
  window.history.replaceState({}, "", url.toString());
}

export function ConversationProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [activeId, setActiveIdState] = useState<number | null>(() =>
    conversationIdFromUrl(),
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const setActiveId = (id: number | null) => {
    setActiveIdState(id);
    updateUrlConversationId(id);
  };

  useEffect(() => {
    const onPopState = () => setActiveIdState(conversationIdFromUrl());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const { data: conversations = [], isLoading } = useQuery({
    queryKey: ["conversations"],
    queryFn: async (): Promise<Thread[]> => {
      const res = await fetch("/api/conversations");
      if (!res.ok) throw new Error("Failed to fetch conversations");
      const data: BackendConversation[] = await res.json();
      return data.map((c) => ({
        id: c.id,
        title: c.title,
        customer: c.customer,
        status: c.status,
        messages: c.messages.map((m) => ({
          id: m.id,
          role: m.role,
          text: m.text,
          time: m.time,
          status: m.status ?? undefined,
        })),
      }));
    },
  });

  const createMutation = useMutation({
    mutationFn: async (data: {
      title: string;
      customer?: string;
      status?: string;
    }): Promise<Thread> => {
      const res = await fetch("/api/conversations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to create conversation");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["conversations"] }),
  });

  const createConversation = async (data: {
    title: string;
    customer?: string;
    status?: string;
  }): Promise<Thread> => {
    const created = await createMutation.mutateAsync(data);
    setActiveId(created.id);
    setSidebarOpen(false);
    return created;
  };

  // Derive the effective active conversation instead of calling setState in an
  // effect: when the active one disappears (or nothing is selected yet) the
  // first conversation wins, without a cascading re-render.
  const effectiveActiveId =
    activeId !== null && conversations.some((thread) => thread.id === activeId)
      ? activeId
      : conversations[0]?.id ?? null;

  return (
    <ConversationContext.Provider
      value={{
        conversations,
        isLoading,
        activeId: effectiveActiveId,
        setActiveId,
        createConversation,
        sidebarOpen,
        setSidebarOpen,
      }}
    >
      {children}
    </ConversationContext.Provider>
  );
}

export function useConversations() {
  const ctx = useContext(ConversationContext);
  if (!ctx) {
    throw new Error("useConversations must be used within ConversationProvider");
  }
  return ctx;
}
