// components/DropdownFilterSlider.tsx
"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface DropdownFilterSliderProps {
  icon: React.ComponentType<any>
  title: string
  label: string
  value: [number, number]
  min: number
  max: number
  step?: number
  formatter: (value: number) => string
  onValueChange: (range: [number, number]) => void
  onReset: () => void
  isFiltered: boolean
}

export function DropdownFilterSlider({
  icon: Icon,
  title,
  label,
  value,
  min,
  max,
  step = 0.1,
  formatter,
  onValueChange,
  onReset,
  isFiltered,
}: DropdownFilterSliderProps) {
  const [localRange, setLocalRange] = useState<[number, number]>(value)

  useEffect(() => {
    setLocalRange(value)
  }, [value])

  const handleSliderChange = (values: number[]) => {
    const newRange: [number, number] = [values[0], values[1]]
    setLocalRange(newRange)
  }

  const handleSliderCommit = (values: number[]) => {
    const newRange: [number, number] = [values[0], values[1]]
    onValueChange(newRange)
  }

  const tag = isFiltered ? (
    `${formatter(value[0])} - ${formatter(value[1])}`
  ) : undefined

  return (
    <>
      {/* Desktop Version */}
      <div className="hidden sm:block">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="transparent" size="sm" className="flex items-center gap-2 text-foreground/80">
              <Icon size={16} />
              {title}
              {tag && (
                <span className="tag-primary">{tag}</span>
              )}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="w-80" align="end">
            <DropdownMenuLabel className="text-sm">{label}</DropdownMenuLabel>
            <DropdownMenuSeparator />

            <div className="p-2 space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-primary-foreground">{formatter(localRange[0])}</span>
                <span className="text-primary-foreground">{formatter(localRange[1])}</span>
              </div>

              <div className="px-2">
                <Slider
                  value={localRange}
                  min={min}
                  max={max}
                  step={step}
                  onValueChange={handleSliderChange}
                  onValueCommit={handleSliderCommit}
                  className="w-full"
                />
              </div>

              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{formatter(min)}</span>
                <span>{formatter(max)}</span>
              </div>
            </div>

            <DropdownMenuSeparator />
            <div className="dropdown-footer">
              <Button
                variant="secondary"
                size="xs"
                onClick={onReset}
                disabled={!isFiltered}
              >
                Reset
              </Button>
            </div>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Mobile Version */}
      <div className="block sm:hidden space-y-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="p-2 space-y-3">
          <div className="flex justify-between text-xs">
            <span className="text-primary-foreground">{formatter(localRange[0])}</span>
            <span className="text-primary-foreground">{formatter(localRange[1])}</span>
          </div>
          <div className="px-2">
            <Slider
              value={localRange}
              min={min}
              max={max}
              step={step}
              onValueChange={handleSliderChange}
              onValueCommit={handleSliderCommit}
              className="w-full"
            />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{formatter(min)}</span>
            <span>{formatter(max)}</span>
          </div>
        </div>
      </div>
    </>
  )
}
