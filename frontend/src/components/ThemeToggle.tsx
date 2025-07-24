"use client"

import * as React from "react"
import { SunIcon, MoonIcon, MonitorIcon } from "@phosphor-icons/react"
import { useTheme } from "next-themes"

import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)

  // Only render after client-side hydration
  React.useEffect(() => {
    setMounted(true)
  }, [])

  const themes = [
    { value: "system", label: "System", icon: MonitorIcon },
    { value: "light", label: "Light", icon: SunIcon },
    { value: "dark", label: "Dark", icon: MoonIcon },
  ]

  // Show loading state during hydration
  if (!mounted) {
    return (
      <div className="space-y-2">
        <div className="flex gap-0.5 bg-foreground/6 p-0.5 rounded-full">
          {themes.map(({ value, label, icon: Icon }) => (
            <div key={value} className="flex items-center">
              <div className="flex items-center p-2 rounded-full text-sm text-muted-foreground">
                <Icon size={16} />
                <span className="sr-only">{label}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <RadioGroup
        value={theme}
        onValueChange={(value) => setTheme(value)}
        className="flex gap-0.5 bg-foreground/6 p-0.5 rounded-full"
      >
        {themes.map(({ value, label, icon: Icon }) => (
          <div key={value} className="flex items-center">
            <RadioGroupItem value={value} id={value} className="peer sr-only" />
            <label htmlFor={value} className="flex items-center p-2 rounded-full cursor-pointer text-sm text-foreground/60 peer-data-[state=checked]:bg-foreground/10 peer-data-[state=checked]:text-foreground">
              <Icon size={16} weight="bold" />
              <span className="sr-only">{label}</span>
            </label>
          </div>
        ))}
      </RadioGroup>
    </div>
  )
}
