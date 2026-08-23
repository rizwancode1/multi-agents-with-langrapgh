"use client"

import type { TicketPriority, TicketStatus } from "@/hooks/useTickets"

const STATUS_STYLES: Record<TicketStatus, string> = {
  open: "bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/20",
  in_progress:
    "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
  resolved:
    "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
  closed: "bg-slate-500/10 text-slate-600 dark:text-slate-400 border-slate-500/20",
  cancelled: "bg-slate-500/10 text-slate-500 border-slate-500/20",
}

const STATUS_LABELS: Record<TicketStatus, string> = {
  open: "Open",
  in_progress: "In progress",
  resolved: "Resolved",
  closed: "Closed",
  cancelled: "Cancelled",
}

const PRIORITY_STYLES: Record<TicketPriority, string> = {
  low: "bg-slate-500/10 text-slate-600 dark:text-slate-400 border-slate-500/20",
  medium: "bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/20",
  high: "bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/20",
  urgent: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
}

export function StatusBadge({ status }: { status: TicketStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium ${STATUS_STYLES[status] ?? STATUS_STYLES.closed}`}
    >
      <span className="size-1.5 rounded-full bg-current" />
      {STATUS_LABELS[status] ?? status}
    </span>
  )
}

export function PriorityBadge({ priority }: { priority: TicketPriority }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium capitalize ${PRIORITY_STYLES[priority] ?? PRIORITY_STYLES.low}`}
    >
      {priority}
    </span>
  )
}
