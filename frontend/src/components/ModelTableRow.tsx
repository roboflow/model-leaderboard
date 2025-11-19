"use client"

import { TableRow, TableCell } from "@/components/ui/table"
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip"
import { SiArxiv } from "@icons-pack/react-simple-icons"
import { SlidersHorizontalIcon } from "@phosphor-icons/react"
import Link from "next/link"
import { formatters } from "@/lib/formatters"

// Utility function to parse arXiv ID from URL
const parseArxivId = (url: string): string => {
  const match = url.match(/arxiv\.org\/abs\/(.+)/);
  return match ? match[1] : "arXiv";
}

interface ModelResult {
  metadata: {
    model: string
    license: string
    architecture: string
    github_url: string
    paper_url: string
    param_count: number
    pretrain_datasets: string[]
    run_parameters: {
      // YOLO-style parameters
      imgsz?: number
      iou?: number
      max_det?: number
      conf?: number
      verbose?: boolean

      // DETR-style parameters
      resolution?: number
      num_queries?: number
      num_select?: number
      threshold?: number
    }
  }
  map50_95: number
  map50: number
  map75: number
  small_objects: {
    map50_95: number
    map50: number
    map75: number
  }
  medium_objects: {
    map50_95: number
    map50: number
    map75: number
  }
  large_objects: {
    map50_95: number
    map50: number
    map75: number
  }
  f1_50: number
  f1_75: number
  f1_small_objects: {
    f1_50: number
    f1_75: number
  }
  f1_medium_objects: {
    f1_50: number
    f1_75: number
  }
  f1_large_objects: {
    f1_50: number
    f1_75: number
  }
}

interface Column {
  key: string
  label: string
  width: string
  sortable?: boolean
  group: string
  defaultVisible: boolean
  tooltip?: string
  formatter?: 'decimal' | 'percentage' | 'parameters' // Add formatter property
}

interface ModelTableRowProps {
  result: ModelResult
  columns: Column[]
  sortColumn?: string
  columnRange?: { min: number, max: number, column: string } | null
  selectedBenchmark?: string // Optional: for PCS tables to determine formatting
}

function getNestedValue(obj: any, path: string) {
  return path.split('.').reduce((current, key) => current?.[key], obj)
}

// Helper function to format values based on column formatter
const getFormattedValue = (value: any, column?: Column, shouldShowAsterisk: boolean = false, selectedBenchmark?: string) => {
  if (value === null || value === undefined) return '—'
  if (typeof value !== 'number') return value
  
  let formattedValue: string
  
  // Check if column has a specific formatter
  if (column?.formatter) {
    switch (column.formatter) {
      case 'decimal':
        // Use 2 decimal places for sa_co_bio metrics (pmf), otherwise 1
        const decimalPlaces = (selectedBenchmark?.includes('sa_co_bio') && column.key === 'results.pmf') ? 2 : 1
        formattedValue = formatters.decimal(value, decimalPlaces)
        break
      case 'percentage': 
        formattedValue = formatters.percentage(value)
        break
      case 'parameters':
        formattedValue = formatters.parameters(value)
        break
      default:
        formattedValue = formatters.percentage(value) // Default for backward compatibility
    }
  } else {
    // Default behavior for existing tables (object detection)
    formattedValue = formatters.percentage(value)
  }
  
  // Add asterisk if explicitly requested
  if (shouldShowAsterisk) {
    formattedValue += '*'
  }
  
  return formattedValue
}

export function ModelTableRow({ result, columns, sortColumn, columnRange, selectedBenchmark }: ModelTableRowProps) {
  // Check if this is a PCS table by looking for PCS-specific columns
  const isPCSTable = columns.some(col => 
    col.key === 'results.cgf' || 
    col.key === 'results.pmf' ||
    col.key === 'results.miou' || 
    col.key === 'results.gold' || 
    col.key === 'results.ap_coco_o'
  )

  const renderCellContent = (columnKey: string) => {
    let baseContent
    
    // Find the column definition to get formatter info
    const column = columns.find(col => col.key === columnKey)

    switch (columnKey) {
      case "metadata.model":
        const modelContent = result.metadata.github_url ? (
          <Link
            href={result.metadata.github_url}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium link-primary transition-colors duration-200 inline-flex items-center gap-2 group"
          >
            <span>{result.metadata.model}</span>
          </Link>
        ) : (
          <span className="font-medium">{result.metadata.model}</span>
        )

        // Access run_parameters from metadata
        const runParams = result.metadata.run_parameters

        // Create mapping of parameter keys to descriptive names
        const parameterLabels = {
          imgsz: 'Image size',
          iou: 'IoU threshold',
          max_det: 'Maximum detections',
          conf: 'Confidence threshold',
          verbose: 'Verbose output',
          resolution: 'Input resolution',
          num_queries: 'Number of queries',
          num_select: 'Number of selections',
          threshold: 'Detection threshold'
        }

        // Build tooltip content, only including defined parameters
        let tooltipContent = ''
        if (runParams) {
          const paramEntries = Object.entries(runParams)
            .filter(([key, value]) => value !== undefined && value !== null)
            .map(([key, value]) => `${parameterLabels[key as keyof typeof parameterLabels]}: ${value}`)

          tooltipContent = paramEntries.join('\n')
        }

        baseContent = (
          <div className="flex items-center gap-2">
            {modelContent}
            {tooltipContent && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <SlidersHorizontalIcon size={14} className="text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="bg-violet-300 fill-violet-300 text-black max-w-xs">
                  <div className="whitespace-pre-line text-xs">{tooltipContent}</div>
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        )
        break

      case "metadata.param_count":
        baseContent = formatters.parameters(result.metadata.param_count)
        break

      case "paper":
        baseContent = result.metadata.paper_url ? (
          <Link
            href={result.metadata.paper_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-semibold transition-colors text-primary-foreground bg-foreground/6 hover:bg-primary"
          >
            <SiArxiv size={10} className="text-primary-foreground" />
            <span>Arxiv</span>
          </Link>
        ) : (
          <span className="text-muted-foreground text-xs">—</span>
        )
        break

      case "metadata.license":
        baseContent = (
          <span className="text-xs bg-foreground/6 px-2 py-1 rounded">
            {result.metadata.license}
          </span>
        )
        break

      default:
        const value = getNestedValue(result, columnKey)
        
        // Check if this specific value should have an asterisk
        let shouldShowAsterisk = false
        if (isPCSTable && column?.key.startsWith('results.')) {
          // Extract the metric name from the column key (e.g., 'results.cgf' -> 'cgf')
          const metricName = column.key.replace('results.', '')
          const asteriskKey = `${metricName}_asterisk`
          
          // Check if asterisk flag exists and is true, otherwise default to true for non-null values
          const asteriskFlag = getNestedValue(result, `results.${asteriskKey}`)
          shouldShowAsterisk = asteriskFlag !== undefined ? asteriskFlag : (value !== null && value !== undefined)
        }
        
        baseContent = getFormattedValue(value, column, shouldShowAsterisk, selectedBenchmark)
    }

    // Add horizontal bar background if this is the sorted column and it's numeric
    if (columnRange && columnKey === columnRange.column) {
      const value = getNestedValue(result, columnKey)
      if (typeof value === 'number' && !isNaN(value) && isFinite(value)) {
        // Calculate percentage for bar width
        const range = columnRange.max - columnRange.min
        const percentage = range === 0 
          ? 100 // If all values are the same, show full bar
          : ((value - columnRange.min) / range) * 100

        // Ensure minimum visibility and handle edge cases
        const barWidth = Math.max(Math.min(percentage, 100), 2)

        return (
          <div className="relative">
            {/* Background bar */}
            <div
              className="absolute inset-0 bg-gradient-primary border border-gradient-primary rounded transition-all duration-200"
              style={{ width: `${barWidth}%` }}
            />
            {/* Content on top */}
            <div className="relative z-10 px-2 py-1 text-primary-foreground">
              {baseContent}
            </div>
          </div>
        )
      }
    }

    return baseContent
  }

  return (
    <TableRow key={result.metadata.model} className="hover:bg-foreground/6">
      {columns.map((column) => (
        <TableCell key={column.key} className="px-4">
          {renderCellContent(column.key)}
        </TableCell>
      ))}
    </TableRow>
  )
}