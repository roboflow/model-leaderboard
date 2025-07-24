"use client"

// UI Components
import { Table, TableBody, TableHeader, TableCell, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import Link from "next/link"

// Custom Hooks
import { useRangeFilter } from "@/hooks/useRangeFilter"
import { useUniqueValues } from "@/hooks/useUniqueValues"
import { useSetFilter } from "@/hooks/useSetFilter"
import { useColumnManager } from "@/hooks/useColumnManager"

// Filter Components
import { SearchInput } from "@/components/SearchInput"
import { ColumnToggle } from "@/components/ColumnToggle"
import { DropdownFilterSlider } from "@/components/DropdownFilterSlider"
import { DropdownFilterRadio } from "@/components/DropdownFilterRadio"
import { DropdownFilterCheckbox } from "@/components/DropdownFilterCheckbox"
import { MobileControls } from "@/components/MobileControls"

// Table Components
import { SortableTableHeader } from "@/components/SortableTableHeader"
import { ModelTableRow } from "@/components/ModelTableRow"
import { SkeletonTable } from "@/components/SkeletonTable"

// Data & Utils
import aggregateResults from "@/data/aggregate_results.json"
import { formatters } from "@/lib/formatters"
import { CircuitryIcon, FileTextIcon, DatabaseIcon, CpuIcon, GaugeIcon, ArrowSquareOutIcon, HeartIcon } from "@phosphor-icons/react"

import { useState, useMemo, useEffect } from "react"

type SortDirection = "asc" | "desc" | null

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

function getNestedValue(obj: any, path: string) {
  return path.split('.').reduce((current, key) => current?.[key], obj)
}

function getSearchableText(result: ModelResult): string {
  const searchableFields = [
    result.metadata.model,
    result.metadata.license,
    formatters.parameters(result.metadata.param_count),
    formatters.percentage(result.map50_95),
    formatters.percentage(result.map50),
    formatters.percentage(result.map75),
    formatters.decimal(result.map50_95),
    formatters.decimal(result.map50),
    formatters.decimal(result.map75),
    formatters.percentage(result.f1_50),
    formatters.percentage(result.f1_75),
    formatters.percentage(result.small_objects.map50_95),
    formatters.percentage(result.small_objects.map50),
    formatters.percentage(result.small_objects.map75),
    formatters.percentage(result.medium_objects.map50_95),
    formatters.percentage(result.medium_objects.map50),
    formatters.percentage(result.medium_objects.map75),
    formatters.percentage(result.large_objects.map50_95),
    formatters.percentage(result.large_objects.map50),
    formatters.percentage(result.large_objects.map75),
    formatters.percentage(result.f1_small_objects.f1_50),
    formatters.percentage(result.f1_small_objects.f1_75),
    formatters.percentage(result.f1_medium_objects.f1_50),
    formatters.percentage(result.f1_medium_objects.f1_75),
    formatters.percentage(result.f1_large_objects.f1_50),
    formatters.percentage(result.f1_large_objects.f1_75),
    formatters.decimal(result.map50_95),
    formatters.decimal(result.map50),
    formatters.decimal(result.map75),
    formatters.decimal(result.f1_50),
    formatters.decimal(result.f1_75),
    result.metadata.param_count.toString(),
  ]

  return searchableFields.join(' ').toLowerCase()
}





export default function Home() {
  // ============================================================================
  // STATE
  // ============================================================================
  const [search, setSearch] = useState("")
  const [sortColumn, setSortColumn] = useState<string>("map50_95")
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc")
  const [isLoading, setIsLoading] = useState(true)
  const [selectedDataset, setSelectedDataset] = useState("COCO 2017")

  // ============================================================================
  // DATA EXTRACTION & FILTERS
  // ============================================================================
  // Extract unique values for filter dropdowns
  const availableLicenses = useUniqueValues(aggregateResults, (result) => result.metadata.license)
  const availableArchitectures = useUniqueValues(aggregateResults, (result) => result.metadata.architecture)
  const availablePretrainDatasets = useUniqueValues(aggregateResults, (result) => result.metadata.pretrain_datasets)
  const availableDatasets = useMemo(() => ["COCO 2017"], [])

  // Initialize filter hooks
  const licenseFilter = useSetFilter(availableLicenses)
  const architectureFilter = useSetFilter(availableArchitectures)
  const pretrainDatasetFilter = useSetFilter(availablePretrainDatasets)

  // Calculate parameter range for slider
  const { minParams, maxParams } = useMemo(() => {
    if (aggregateResults.length === 0) return { minParams: 0, maxParams: 100 }
    const paramCounts = aggregateResults.map(result => result.metadata.param_count / 1_000_000)
    return { minParams: Math.min(...paramCounts), maxParams: Math.max(...paramCounts) }
  }, [])

  const parameterFilter = useRangeFilter(minParams, maxParams)
  const columnManager = useColumnManager()

  // ============================================================================
  // HANDLERS
  // ============================================================================
  const handleDatasetChange = (dataset: string) => setSelectedDataset(dataset)
  
  const handleSort = (key: string) => {
    if (sortColumn === key) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortColumn(key)
      setSortDirection("desc")
    }
  }

  // ============================================================================
  // EFFECTS
  // ============================================================================
  // Loading simulation
  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 100)
    return () => clearTimeout(timer)
  }, [])

  // ============================================================================
  // COMPUTED VALUES
  // ============================================================================
  // Apply all filters and sorting
  const filteredAndSortedResults = useMemo(() => {
    let filtered = aggregateResults

    // Search filter
    if (search.trim()) {
      const searchLower = search.toLowerCase().trim()
      filtered = filtered.filter(result => getSearchableText(result).includes(searchLower))
    }

    // License filter
    if (licenseFilter.selectedItems.size > 0) {
      filtered = filtered.filter(result => licenseFilter.selectedItems.has(result.metadata.license))
    }

    // Architecture filter
    if (architectureFilter.selectedItems.size > 0) {
      filtered = filtered.filter(result => architectureFilter.selectedItems.has(result.metadata.architecture))
    }

    // Pretrain datasets filter
    if (pretrainDatasetFilter.selectedItems.size > 0) {
      filtered = filtered.filter(result => {
        return result.metadata.pretrain_datasets?.some(dataset =>
          pretrainDatasetFilter.selectedItems.has(dataset)
        )
      })
    }

    // Parameter range filter
    const [minParamsMillion, maxParamsMillion] = parameterFilter.value
    filtered = filtered.filter(result => {
      const paramCountMillion = result.metadata.param_count / 1_000_000
      return paramCountMillion >= minParamsMillion && paramCountMillion <= maxParamsMillion
    })

    // Apply sorting
    if (!sortColumn || !sortDirection) return filtered
    return [...filtered].sort((a, b) => {
      const aValue = getNestedValue(a, sortColumn)
      const bValue = getNestedValue(b, sortColumn)
      if (aValue < bValue) return sortDirection === "asc" ? -1 : 1
      if (aValue > bValue) return sortDirection === "asc" ? 1 : -1
      return 0
    })
  }, [search, licenseFilter.selectedItems, architectureFilter.selectedItems, pretrainDatasetFilter.selectedItems, parameterFilter.value, sortColumn, sortDirection])

  // Calculate range for bar indicators
  const columnRange = useMemo(() => {
    if (!sortColumn || filteredAndSortedResults.length === 0) return null
    const nonNumericColumns = ['metadata.model', 'metadata.license', 'paper']
    if (nonNumericColumns.includes(sortColumn)) return null
    
    const values = filteredAndSortedResults.map(result => getNestedValue(result, sortColumn))
    const numericValues = values.filter(v => typeof v === 'number')
    if (numericValues.length === 0) return null
    
    return { min: Math.min(...numericValues), max: Math.max(...numericValues), column: sortColumn }
  }, [sortColumn, filteredAndSortedResults])

  // ============================================================================
  // RENDER
  // ============================================================================
  return (
    <>
      {/* Hero Section */}
      <section className="pt-6 sm:pt-12">
        <div className="container-base mx-auto">
          <div className="flex justify-between">
            <div className="space-y-2">
              <Badge variant="primary">
                <Link href="https://github.com/roboflow/supervision" target="_blank" className="flex items-center gap-1">
                  Powered by SuperVision <ArrowSquareOutIcon size={14} weight="thin" />
                </Link>
              </Badge>
              <h1 className="text-3xl sm:text-4xl"><span className="text-primary-foreground">Computer Vision</span><br />Model Leaderboard</h1>
              <p className="prose prose-sm max-w-lg">
                Compare computer vision models benchmarked on the COCO 2017 dataset using standardized mAP and F1 metrics. Filter by architecture, size, or license to find the best model for your use case.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Main Leaderboard */}
      <section className="pb-12">
        <div className="container-base mx-auto">
          <div className="relative">
            {/* Search & Filters */}
            <div className="flex flex-row gap-2 py-4 sm:flex-wrap">
              <SearchInput
                value={search}
                onChange={setSearch}
                placeholder="Search models..."
                className="flex-1"
              />

              {/* Desktop Filters */}
              <div className="hidden sm:flex gap-2 flex-wrap">
                <DropdownFilterCheckbox
                  icon={CircuitryIcon}
                  title="Architecture"
                  label="Filter by Architecture"
                  availableItems={availableArchitectures}
                  selectedItems={architectureFilter.selectedItems}
                  onItemToggle={architectureFilter.toggleItem}
                  onClearAll={architectureFilter.clearAll}
                  onSelectAll={architectureFilter.selectAll}
                />

                <DropdownFilterSlider
                  icon={CpuIcon}
                  title="Parameters"
                  label="Filter by Parameter Count"
                  formatter={formatters.parametersFromMillion}
                  step={0.1}
                  {...parameterFilter}
                />

                <DropdownFilterCheckbox
                  icon={DatabaseIcon}
                  title="Pretrained on"
                  label="Filter by Pretrained Datasets"
                  availableItems={availablePretrainDatasets}
                  selectedItems={pretrainDatasetFilter.selectedItems}
                  onItemToggle={pretrainDatasetFilter.toggleItem}
                  onClearAll={pretrainDatasetFilter.clearAll}
                  onSelectAll={pretrainDatasetFilter.selectAll}
                />

                <DropdownFilterCheckbox
                  icon={FileTextIcon}
                  title="License"
                  label="Filter by License"
                  availableItems={availableLicenses}
                  selectedItems={licenseFilter.selectedItems}
                  onItemToggle={licenseFilter.toggleItem}
                  onClearAll={licenseFilter.clearAll}
                  onSelectAll={licenseFilter.selectAll}
                />

                <DropdownFilterRadio
                  icon={GaugeIcon}
                  title="Benchmark"
                  label="Select Dataset"
                  availableItems={availableDatasets}
                  selectedItem={selectedDataset}
                  onItemChange={handleDatasetChange}
                />

                <Separator orientation="vertical" className="max-h-9" />

                <ColumnToggle
                  columns={columnManager.allColumns.map(col => ({ key: col.key, label: col.label, group: col.group }))}
                  visibleColumns={columnManager.visibleColumns}
                  onToggleColumn={columnManager.toggleColumn}
                  onShowAll={columnManager.showAllColumns}
                  onHideAll={columnManager.hideAllColumns}
                  onResetToDefaults={columnManager.resetToDefaults}
                />
              </div>

              {/* Mobile Controls */}
              <MobileControls
                // Pass the complete filter objects
                licenseFilter={licenseFilter}
                architectureFilter={architectureFilter}
                pretrainDatasetFilter={pretrainDatasetFilter}
                parameterFilter={parameterFilter}
                
                // Pass the available data
                availableLicenses={availableLicenses}
                availableArchitectures={availableArchitectures}
                availablePretrainDatasets={availablePretrainDatasets}
                
                // Dataset selection (simple)
                availableDatasets={availableDatasets}
                selectedDataset={selectedDataset}
                onDatasetChange={handleDatasetChange}
                
                // Column management (already consolidated)
                columnManager={columnManager}
              />
            </div>

            {/* Results Table */}
            <ScrollArea className="h-[625px] w-full max-w-[1504px] overflow-x-auto rounded-md border whitespace-nowrap">
              <Table className="min-w-max table-auto text-foreground/60">
                <TableHeader>
                  <TableRow className="sticky top-0 z-20 bg-background/10 backdrop-blur-lg px-0">
                    {columnManager.filteredColumns.map((column) => (
                      <SortableTableHeader
                        key={column.key}
                        columnKey={column.key}
                        label={column.label}
                        width={column.width}
                        sortable={column.sortable}
                        tooltip={column.tooltip}
                        currentSortColumn={sortColumn}
                        currentSortDirection={sortDirection}
                        onSort={handleSort}
                        className="px-4 text-foreground/60"
                      />
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {isLoading ? (
                    <SkeletonTable columns={columnManager.filteredColumns} />
                  ) : filteredAndSortedResults.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={columnManager.filteredColumns.length} className="text-center py-8 text-muted-foreground">
                        {search ? (
                          <div>
                            <p>No models found matching "{search}"</p>
                            <Button
                              variant="link"
                              onClick={() => setSearch("")}
                              className="mt-2 text-primary-foreground"
                            >
                              Clear search
                            </Button>
                          </div>
                        ) : (
                          "No models to display"
                        )}
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredAndSortedResults.map((result) => (
                      <ModelTableRow
                        key={result.metadata.model}
                        result={result}
                        columns={columnManager.filteredColumns}
                        sortColumn={sortColumn}
                        columnRange={columnRange}
                      />
                    ))
                  )}
                </TableBody>
              </Table>
              <ScrollBar orientation="horizontal" />
              </ScrollArea>

              {/* Results Summary */}
              <div className="w-full mt-4 flex flex-wrap items-start gap-2 justify-between">
                <div className="space-y-2">
                  <div className="text-sm text-muted-foreground">
                    {isLoading ? (
                      <div className="h-4 bg-muted/50 animate-pulse rounded w-40"></div>
                    ) : (
                      <>
                        Displaying <span className="font-medium text-foreground">{filteredAndSortedResults.length}</span> out of{" "}
                        <span className="font-medium text-foreground">{aggregateResults.length}</span> models
                      </>
                    )}
                  </div>

                  {/* Active Filters Indicator */}
                  {!isLoading && (search || licenseFilter.selectedItems.size > 0 || architectureFilter.selectedItems.size > 0 || pretrainDatasetFilter.selectedItems.size > 0 || parameterFilter.isFiltered) && (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span>Filtered by:</span>
                      {search && <span className="bg-muted px-2 py-0.5 rounded">Search</span>}
                      {licenseFilter.selectedItems.size > 0 && <span className="bg-muted px-2 py-0.5 rounded">License</span>}
                      {architectureFilter.selectedItems.size > 0 && <span className="bg-muted px-2 py-0.5 rounded">Architecture</span>}
                      {pretrainDatasetFilter.selectedItems.size > 0 && <span className="bg-muted px-2 py-0.5 rounded">Pretrained Datasets</span>}
                      {parameterFilter.isFiltered && <span className="bg-muted px-2 py-0.5 rounded">Parameters</span>}
                    </div>
                  )}
                </div>
                <div className="prose prose-sm">Made with <HeartIcon size={14} weight="fill" className="inline-block text-primary-foreground" /> by <Link href="https://roboflow.com" target="_blank" className="link-primary">Roboflow</Link></div>
              </div>
          </div>


        </div>
      </section>
    </>
  )
}
