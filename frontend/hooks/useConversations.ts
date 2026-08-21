"use client"

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"

const API_URL = "/api/conversations"

export interface Message {
  id: number
  role: "user" | "assistant"
  text: string
  status?: string
  time: string
}

export interface Conversation {
  id: number
  title: string
  customer: string
  status: string
  messages: Message[]
  created_at: string
  updated_at: string
}

export function useConversations() {
  return useQuery({
    queryKey: ["conversations"],
    queryFn: async (): Promise<Conversation[]> => {
      const res = await fetch(API_URL)
      if (!res.ok) throw new Error("Failed to fetch conversations")
      return res.json()
    },
  })
}

export function useConversation(id: number) {
  return useQuery({
    queryKey: ["conversations", id],
    queryFn: async (): Promise<Conversation> => {
      const res = await fetch(`${API_URL}/${id}`)
      if (!res.ok) throw new Error("Failed to fetch conversation")
      return res.json()
    },
    enabled: !!id,
  })
}

export function useCreateConversation() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (data: { title: string; customer?: string; status?: string }) => {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      })
      if (!res.ok) throw new Error("Failed to create conversation")
      return res.json()
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  })
}

export function useAddMessage(conversationId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (data: { role: string; text: string; status?: string }) => {
      const res = await fetch(`${API_URL}/${conversationId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      })
      if (!res.ok) throw new Error("Failed to add message")
      return res.json()
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["conversations", conversationId] })
      qc.invalidateQueries({ queryKey: ["conversations"] })
    },
  })
}
