"use client"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ColumnsPlusRightIcon } from "@phosphor-icons/react"
import { MobileCheckboxItem } from "@/components/MobileCheckboxItem"

interface ColumnDefinition {
  key: string
  label: string
  group?: string
}

interface ColumnToggleProps {
  columns: ColumnDefinition[]
  visibleColumns: Set<string>
  onToggleColumn: (columnKey: string) => void
  onShowAll: () => void
  onHideAll: () => void
  onResetToDefaults: () => void
}

export function ColumnToggle({
  columns,
  visibleColumns,
  onToggleColumn,
  onShowAll,
  onHideAll,
  onResetToDefaults
}: ColumnToggleProps) {
  // Group columns by category
  const groupedColumns = columns.reduce((acc, column) => {
    const group = column.group || 'Other'
    if (!acc[group]) acc[group] = []
    acc[group].push(column)
    return acc
  }, {} as Record<string, ColumnDefinition[]>)

  const visibleCount = visibleColumns.size
  const totalCount = columns.length

  return (
    <>
      <div className="hidden sm:block">
        <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="transparent" size="sm" className="flex items-center gap-2 text-foreground/80">
            <ColumnsPlusRightIcon size={16} />
            Columns <span className="text-xs font-normal leading-tight text-muted-foreground">({visibleCount}/{totalCount})</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-64 max-h-96 overflow-y-auto">
          <DropdownMenuLabel>Toggle Columns</DropdownMenuLabel>

          {/* Grouped columns */}
          {Object.entries(groupedColumns).map(([groupName, groupColumns]) => (
            <div key={groupName}>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="text-xs font-medium text-muted-foreground px-2 py-2">
                {groupName}
              </DropdownMenuLabel>
              {groupColumns.map((column) => (
                <DropdownMenuCheckboxItem
                  key={column.key}
                  checked={visibleColumns.has(column.key)}
                  onCheckedChange={() => onToggleColumn(column.key)}
                  onSelect={(e) => {
                    // prevent menu from closing on click
                    e.preventDefault()
                  }}
                  className="[&_svg]:text-primary-foreground"
                >
                  {column.label}
                </DropdownMenuCheckboxItem>
              ))}
            </div>
          ))}

          {/* Reset Button */}
          <DropdownMenuSeparator />
          <div className="dropdown-footer">
            <Button
              variant="secondary"
              size="xs"
              onClick={onShowAll}
            >
              Show All
            </Button>
            <Button
              variant="secondary"
              size="xs"
              onClick={onHideAll}
            >
              Hide All
            </Button>

            <Button
              variant="secondary"
              size="xs"
              onClick={onResetToDefaults}
            >
              Reset
            </Button>
          </div>
        </DropdownMenuContent>
      </DropdownMenu>
      </div>

      {/* Mobile Version */}
      <div className="block sm:hidden space-y-2">
        {/* <div className="text-sm font-medium">Columns</div> */}
        <div className="space-y-1">
          {columns.map((column) => (
            <MobileCheckboxItem
            key={column.key}
            selected={visibleColumns.has(column.key)}
            onSelect={() => onToggleColumn(column.key)}
          >
            {column.label}
          </MobileCheckboxItem>
          ))}
          </div>
      </div>  
    </>
  )
}
