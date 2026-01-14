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

interface GroupedItem {
  groupName: string
  items: { key: string; label: string }[]
}

interface DropdownFilterRadioProps {
  icon: React.ComponentType<any>
  title: string
  label: string
  availableItems?: string[]
  groupedItems?: GroupedItem[]
  selectedItem: string
  onItemChange: (item: string) => void
  tag?: string
}

export function DropdownFilterRadio({
  icon: Icon,
  title,
  label,
  availableItems,
  groupedItems,
  selectedItem,
  onItemChange,
  tag,
}: DropdownFilterRadioProps) {
  const displayTag = tag || selectedItem

  // Helper function to render radio items
  const renderRadioItems = () => {
    if (groupedItems) {
      return groupedItems.map((group, groupIndex) => (
        <div key={group.groupName}>
          {groupIndex > 0 && <DropdownMenuSeparator className="my-2" />}
          <div className="px-2 py-1">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
              {group.groupName}
            </div>
            <div className="space-y-2">
              {group.items.map((item) => (
                <div key={item.key} className="flex items-center space-x-2">
                  <RadioGroupItem
                    value={item.key}
                    id={item.key}
                    className="shrink-0"
                  />
                  <label
                    htmlFor={item.key}
                    className="flex items-center cursor-pointer flex-1 text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    <span>{item.label}</span>
                  </label>
                </div>
              ))}
            </div>
          </div>
        </div>
      ))
    } else if (availableItems) {
      return availableItems.map((item) => (
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
      ))
    }
    return null
  }

  // Helper function to render mobile items
  const renderMobileItems = () => {
    if (groupedItems) {
      return groupedItems.map((group) => (
        <div key={group.groupName} className="space-y-1">
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-2 py-1">
            {group.groupName}
          </div>
          {group.items.map((item) => (
            <MobileRadioItem
              key={item.key}
              selected={selectedItem === item.key}
              onSelect={() => onItemChange(item.key)}
            >
              {item.label}
            </MobileRadioItem>
          ))}
        </div>
      ))
    } else if (availableItems) {
      return availableItems.map((item) => (
        <MobileRadioItem
          key={item}
          selected={selectedItem === item}
          onSelect={() => onItemChange(item)}
        >
          {item}
        </MobileRadioItem>
      ))
    }
    return null
  }

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
                {renderRadioItems()}
              </RadioGroup>
            </div>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Mobile Version */}
      <div className="block sm:hidden space-y-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="block sm:hidden space-y-2">
          {renderMobileItems()}
        </div>
      </div>
    </>
  )
}
