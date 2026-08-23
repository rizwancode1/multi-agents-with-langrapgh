"use client"

import { useTicketStats } from "@/hooks/useTickets"

export function TicketStatsStrip() {
  const { data, isLoading } = useTicketStats(15000)

  if (isLoading || !data) {
    return (
      <div className="grid grid-cols-4 gap-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="h-14 animate-pulse rounded-xl border border-border bg-accent/40"
          />
        ))}
      </div>
    )
  }

  const open = data.by_status["open"] ?? 0
  const inProgress = data.by_status["in_progress"] ?? 0
  const resolved = data.by_status["resolved"] ?? 0
  const urgent = data.by_priority["urgent"] ?? 0

  const cards = [
    { label: "Open", value: open, accent: "text-sky-600 dark:text-sky-400" },
    {
      label: "In progress",
      value: inProgress,
      accent: "text-amber-600 dark:text-amber-400",
    },
    {
      label: "Resolved",
      value: resolved,
      accent: "text-emerald-600 dark:text-emerald-400",
    },
    { label: "Urgent", value: urgent, accent: "text-red-600 dark:text-red-400" },
  ]

  return (
    <div className="grid grid-cols-4 gap-2">
      {cards.map((card) => (
        <div
          key={card.label}
          className="rounded-xl border border-border bg-card px-2 py-2.5 text-center shadow-sm"
        >
          <p className={`text-lg font-semibold leading-none ${card.accent}`}>
            {card.value}
          </p>
          <p className="mt-1 text-[10px] text-muted-foreground">{card.label}</p>
        </div>
      ))}
    </div>
  )
}
