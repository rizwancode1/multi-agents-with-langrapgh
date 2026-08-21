"use client";

export function Avatar({ name, tone = "bg-signal" }: { name: string; tone?: string }) {
  return (
    <span
      className={`flex size-8 shrink-0 items-center justify-center rounded-full text-xs font-semibold text-primary-foreground ${tone}`}
    >
      {name
        .split(" ")
        .map((part) => part[0])
        .join("")
        .slice(0, 2)}
    </span>
  );
}
