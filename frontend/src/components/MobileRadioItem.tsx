"use client"

import * as React from "react"
import { CircleIcon } from "@phosphor-icons/react"
import { cn } from "@/lib/utils"

interface MobileRadioItemProps {
  children: React.ReactNode
  selected?: boolean
  onSelect?: () => void
  className?: string
  disabled?: boolean
}

export function MobileRadioItem({
  children,
  selected = false,
  onSelect,
  className,
  disabled = false,
  ...props
}: MobileRadioItemProps) {
  return (
    <button
      onClick={onSelect}
      disabled={disabled}
      className={cn(
        "flex items-center justify-between w-full rounded-md px-3 py-2 text-sm transition-colors",
        "hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        "disabled:pointer-events-none disabled:opacity-50",
        selected && "bg-muted",
        className
      )}
      {...props}
    >
      <span className="text-left">{children}</span>
      {selected && (
        <CircleIcon
          size={16}
          weight="fill"
          className="text-primary-foreground shrink-0"
        />
      )}
    </button>
  )
}
