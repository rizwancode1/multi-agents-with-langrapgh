"use client"

import { createContext, useContext, useEffect, useState } from "react"

type Theme = "light" | "dark" | "system"

interface ThemeContextValue {
  theme: Theme
  setTheme: (theme: Theme | ((prev: Theme) => Theme)) => void
  resolvedTheme: "light" | "dark"
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined)

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>("system")
  const [resolvedTheme, setResolvedTheme] = useState<"light" | "dark">("light")

  useEffect(() => {
    const saved = window.localStorage.getItem("relay-theme") as Theme | null
    if (saved) setTheme(saved)
  }, [])

  useEffect(() => {
    const root = document.documentElement
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches
    const isDark = theme === "dark" || (theme === "system" && prefersDark)
    setResolvedTheme(isDark ? "dark" : "light")
    root.classList.toggle("dark", isDark)
    root.classList.toggle("light", !isDark)
    root.dataset.theme = theme
    window.localStorage.setItem("relay-theme", theme)
  }, [theme])

  return (
    <ThemeContext.Provider value={{ theme, setTheme, resolvedTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider")
  return ctx
}
