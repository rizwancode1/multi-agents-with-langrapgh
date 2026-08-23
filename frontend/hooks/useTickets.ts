"use client"

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

const API_URL = "/api/tickets"

export type TicketStatus = "open" | "in_progress" | "resolved" | "closed" | "cancelled"
export type TicketPriority = "low" | "medium" | "high" | "urgent"

export interface Ticket {
  ticket_id: string
  customer_name: string
  customer_email: string
  subject: string
  description: string
  status: TicketStatus
  priority: TicketPriority
  order_id: string | null
  created_at: string | null
  updated_at: string | null
}

export interface TicketListResponse {
  tickets: Ticket[]
  total: number
  limit: number
  offset: number
}

export interface TicketStats {
  total: number
  by_status: Partial<Record<TicketStatus, number>>
  by_priority: Partial<Record<TicketPriority, number>>
}

export interface TicketFilters {
  status?: TicketStatus
  priority?: TicketPriority
  email?: string
  order_id?: string
  q?: string
  limit?: number
  offset?: number
}

function buildQuery(filters: TicketFilters = {}): string {
  const params = new URLSearchParams()
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== "") params.set(key, String(value))
  })
  const qs = params.toString()
  return qs ? `?${qs}` : ""
}

export function useTickets(filters: TicketFilters = {}, refetchIntervalMs = 15000) {
  return useQuery({
    queryKey: ["tickets", filters],
    queryFn: async (): Promise<TicketListResponse> => {
      const res = await fetch(`${API_URL}${buildQuery(filters)}`)
      if (!res.ok) throw new Error("Failed to fetch tickets")
      return res.json()
    },
    refetchInterval: refetchIntervalMs,
  })
}

export function useTicketStats(refetchIntervalMs = 15000) {
  return useQuery({
    queryKey: ["tickets", "stats"],
    queryFn: async (): Promise<TicketStats> => {
      const res = await fetch(`${API_URL}/stats`)
      if (!res.ok) throw new Error("Failed to fetch ticket stats")
      return res.json()
    },
    refetchInterval: refetchIntervalMs,
  })
}

export function useUpdateTicket() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({
      ticketId,
      ...patch
    }: {
      ticketId: string
      status?: TicketStatus
      priority?: TicketPriority
    }) => {
      const res = await fetch(`${API_URL}/${encodeURIComponent(ticketId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      })
      if (!res.ok) throw new Error("Failed to update ticket")
      return res.json()
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tickets"] })
    },
  })
}
