"use client"

import { Button } from "@/components/ui/button"
import { TableHead } from "@/components/ui/table"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { ArrowDownIcon, ArrowUpIcon, ArrowsDownUpIcon, InfoIcon } from "@phosphor-icons/react"

type SortDirection = "asc" | "desc" | null

interface SortableTableHeaderProps {
  columnKey: string
  label: string
  width?: string
  sortable?: boolean
  tooltip?: string
  currentSortColumn?: string
  currentSortDirection?: SortDirection
  onSort?: (columnKey: string) => void
  className?: string
}

export function SortableTableHeader({
  columnKey,
  label,
  width,
  sortable = true,
  tooltip,
  currentSortColumn,
  currentSortDirection,
  onSort,
  className
}: SortableTableHeaderProps) {
  const getSortIcon = () => {
    if (currentSortColumn !== columnKey) return <ArrowsDownUpIcon size={16} />
    return currentSortDirection === "asc" ?
      <ArrowUpIcon size={16} /> :
      <ArrowDownIcon size={16} />
  }

  const handleSort = () => {
    if (sortable && onSort) {
      onSort(columnKey)
    }
  }

  const headerContent = sortable ? (
    <Button
      variant="ghost"
      onClick={handleSort}
      className={`h-auto p-0 text-sm hover:bg-transparent justify-start gap-0.5 font-normal rounded-none ${currentSortColumn === columnKey ? "text-primary-foreground hover:text-primary-foreground" : ""}`}
    >
      {label}
      {tooltip && <InfoIcon size={14} />}
      {getSortIcon()}
    </Button>
  ) : (
    <span className="inline-flex items-center gap-0.5 h-auto p-0 text-sm hover:bg-transparent justify-start font-normal whitespace-nowrap">{label} {tooltip && <InfoIcon size={14} />}</span>
  )

  return (
    <TableHead className={`${width} ${className}`}>

      {tooltip ? (
        <TooltipProvider>
          <Tooltip delayDuration={700}>
            <TooltipTrigger asChild>
              {headerContent}
            </TooltipTrigger>
            <TooltipContent forceMount className="bg-violet-300 fill-violet-300 text-black">
                <p>{tooltip}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ) : (
        headerContent
      )}
    </TableHead>
  )
}
