"use client"

import { useRef } from "react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu"
import { ScrollArea } from "@/components/ui/scroll-area"
import { MobileCheckboxItem } from "@/components/MobileCheckboxItem"

interface DropdownFilterCheckboxProps {
  icon: React.ComponentType<any>
  title: string
  label: string
  availableItems: string[]
  selectedItems: Set<string>
  onItemToggle: (item: string) => void
  onClearAll: () => void
  onSelectAll?: () => void
  maxDisplayItems?: number
  className?: string
}

export function DropdownFilterCheckbox({
  icon: Icon,
  title,
  label,
  availableItems,
  selectedItems,
  onItemToggle,
  onClearAll,
  onSelectAll,
  maxDisplayItems = 2,
  className,
}: DropdownFilterCheckboxProps) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const totalCount = availableItems.length
  const activeCount = selectedItems.size
  const isFiltered = activeCount > 0 && activeCount < totalCount

  return (
    <>
      {/* Desktop Version */}
      <div className="hidden sm:block">
        <DropdownMenu>
          <DropdownMenuTrigger
            ref={triggerRef}
            className="focus-visible:outline-none"
            onMouseDown={(e) => e.preventDefault()}
            asChild
          >
            <Button variant="transparent" size="sm" className={`flex items-center gap-2 text-foreground/80 ${className}`}>
              <Icon size={16} />
              {title}
              {isFiltered && (
                <div className="ml-1 flex gap-1 flex-wrap">
                  {activeCount <= maxDisplayItems ? (
                    // Show individual item names for small selections
                    Array.from(selectedItems).slice(0, maxDisplayItems).map((item) => (
                      <span key={item} className="tag-primary">
                        {item}
                      </span>
                    ))
                  ) : (
                    // Show count for large selections
                    <span className="tag-primary">
                      {activeCount}
                    </span>
                  )}
                </div>
              )}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
              className="w-56"
            align="end"
            onMouseDown={(e) => {
              e.preventDefault();
              triggerRef.current?.click();
            }}
          >
            <DropdownMenuLabel>{label}</DropdownMenuLabel>
            <DropdownMenuSeparator />

            <ScrollArea className="h-48">
              <div className="space-y-1">
                {availableItems.map((item) => (
                  <DropdownMenuCheckboxItem
                    key={item}
                    checked={selectedItems.has(item)}
                    onCheckedChange={() => onItemToggle(item)}
                    onSelect={(e) => {
                      // Prevent menu from closing on click
                      e.preventDefault()
                    }}
                    className="[&_svg]:text-primary-foreground"
                  >
                    {item}
                  </DropdownMenuCheckboxItem>
                ))}
              </div>
            </ScrollArea>

            {/* Action Buttons */}
            <DropdownMenuSeparator />
            <div className="dropdown-footer">
              {onSelectAll && (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={onSelectAll}
                  disabled={activeCount === totalCount}
                  className="mr-2"
                >
                  All
                </Button>
              )}
              <Button
                variant="secondary"
                size="xs"
                onClick={onClearAll}
                disabled={activeCount === 0}
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
        <div className=" space-y-1">
          {availableItems.map((item) => (
            <MobileCheckboxItem
              key={item}
              selected={selectedItems.has(item)}
              onSelect={() => onItemToggle(item)}
            >
              {item}
            </MobileCheckboxItem>
          ))}
        </div>
      </div>
    </>
  )
}
