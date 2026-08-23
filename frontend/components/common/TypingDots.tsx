"use client"

export function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1 py-1.5" role="status" aria-label="Relay AI is responding">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="size-1.5 animate-bounce rounded-full bg-muted-foreground/70"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </span>
  )
}
