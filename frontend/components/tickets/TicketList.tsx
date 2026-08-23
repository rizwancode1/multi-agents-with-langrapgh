"use client"

import { LifeBuoy, RefreshCcw } from "lucide-react"

import {
  useTickets,
  type Ticket,
  type TicketFilters,
} from "@/hooks/useTickets"
import { PriorityBadge, StatusBadge } from "./TicketBadge"

function relativeTime(iso: string | null): string {
  if (!iso) return ""
  const diffMs = Date.now() - new Date(iso).getTime()
  const minutes = Math.floor(diffMs / 60000)
  if (minutes < 1) return "Just now"
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

export function TicketSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="flex flex-col gap-3">
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="h-16 animate-pulse rounded-xl border border-border bg-accent/40"
        />
      ))}
    </div>
  )
}

export function TicketList({
  filters = {},
  limit = 8,
  compact = false,
}: {
  filters?: TicketFilters
  limit?: number
  compact?: boolean
}) {
  const { data, isLoading, isError, refetch, isRefetching } = useTickets(
    { ...filters, limit },
    15000,
  )

  if (isLoading) {
    return <TicketSkeleton rows={compact ? 3 : 5} />
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center gap-2 rounded-xl border border-border p-4 text-center">
        <p className="text-xs text-muted-foreground">
          Couldn&apos;t load tickets.
        </p>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-1.5 rounded-lg border border-input px-2.5 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <RefreshCcw className="size-3" /> Retry
        </button>
      </div>
    )
  }

  const tickets: Ticket[] = data?.tickets ?? []

  if (tickets.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border p-6 text-center">
        <LifeBuoy className="size-5 text-muted-foreground" />
        <p className="text-xs leading-5 text-muted-foreground">
          No tickets yet — ask Relay to create one and it will show up here
          live.
        </p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2" aria-busy={isRefetching}>
      {tickets.map((ticket) => (
        <article
          key={ticket.ticket_id}
          className="rounded-xl border border-border bg-card p-3 shadow-sm transition hover:border-ring/40"
        >
          <div className="flex items-start justify-between gap-2">
            <p
              className={`font-medium ${compact ? "line-clamp-1 text-xs" : "text-sm"}`}
              title={ticket.subject}
            >
              {ticket.subject}
            </p>
            <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
              {ticket.ticket_id}
            </span>
          </div>
          {!compact && (
            <p className="mt-1 line-clamp-2 text-xs leading-5 text-muted-foreground">
              {ticket.description}
            </p>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <StatusBadge status={ticket.status} />
            <PriorityBadge priority={ticket.priority} />
            <span className="ml-auto text-[10px] text-muted-foreground">
              {ticket.customer_name} · {relativeTime(ticket.created_at)}
            </span>
          </div>
        </article>
      ))}
    </div>
  )
}
