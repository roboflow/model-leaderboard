"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { MobileRadioItem } from "@/components/MobileRadioItem"

interface DropdownFilterRadioProps {
  icon: React.ComponentType<any>
  title: string
  label: string
  availableItems: string[]
  selectedItem: string
  onItemChange: (item: string) => void
  tag?: string
}

export function DropdownFilterRadio({
  icon: Icon,
  title,
  label,
  availableItems,
  selectedItem,
  onItemChange,
  tag,
}: DropdownFilterRadioProps) {
  const displayTag = tag || selectedItem

  return (
    <>
      {/* Desktop Version */}
      <div className="hidden sm:block">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="transparent" size="sm" className="flex items-center gap-2 text-foreground/80">
              <Icon size={16} />
              {title}
              <span className="tag-primary">
                {displayTag}
              </span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="w-56" align="end">
            <DropdownMenuLabel>{label}</DropdownMenuLabel>
            <DropdownMenuSeparator />

            <div className="p-2">
              <RadioGroup
                value={selectedItem}
                onValueChange={onItemChange}
                className="gap-2"
              >
                {availableItems.map((item) => (
                  <div key={item} className="flex items-center space-x-2">
                    <RadioGroupItem
                      value={item}
                      id={item}
                      className="shrink-0"
                    />
                    <label
                      htmlFor={item}
                      className="flex items-center cursor-pointer flex-1 text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      <span>{item}</span>
                    </label>
                  </div>
                ))}
              </RadioGroup>
            </div>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Mobile Version */}
      <div className="block sm:hidden space-y-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="block sm:hidden space-y-1">
        {availableItems.map((item) => (
          <MobileRadioItem
            key={item}
            selected={selectedItem === item}
            onSelect={() => onItemChange(item)}
          >
            {item}
          </MobileRadioItem>
        ))}
      </div>
      </div>
    </>
  )
}
