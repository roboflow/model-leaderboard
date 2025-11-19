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
import { usePCSColumnManager } from "@/hooks/usePCSColumnManager"

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

// Components
import { HeroSection } from "@/components/HeroSection"
import { TaskToggle } from "@/components/TaskToggle"
import { TaskCard } from "@/components/TaskCard"


// Data & Utils
import pcsResults from "@/data/pcs_sample_results.json"
import { formatters } from "@/lib/formatters"
import { CircuitryIcon, FileTextIcon, DatabaseIcon, CpuIcon, GaugeIcon, ArrowSquareOutIcon, HeartIcon } from "@phosphor-icons/react"

import { useState, useMemo, useEffect } from "react"

type SortDirection = "asc" | "desc" | null

interface PCSModelResult {
  metadata: {
    model: string
    license: string
    architecture: string
    github_url: string
    paper_url: string
    param_count: number | null
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
  results: {
    cgf?: number | null
    pmf?: number | null
    ap?: number | null
    ap_coco_o?: number | null
    gold?: number | null
    silver?: number | null
    bronze?: number | null
    bio?: number | null
    miou?: number | null
    // Optional asterisk control for each metric
    cgf_asterisk?: boolean
    pmf_asterisk?: boolean
    ap_asterisk?: boolean
    ap_coco_o_asterisk?: boolean
    gold_asterisk?: boolean
    silver_asterisk?: boolean
    bronze_asterisk?: boolean
    bio_asterisk?: boolean
    miou_asterisk?: boolean
  }
}

interface BenchmarkData {
  metadata: {
    task: string
    benchmark: string
    description: string
    columns: string[]
  }
  models: PCSModelResult[]
}

interface PCSData {
  instance_segmentation: {
    lvis: BenchmarkData
    sa_co_gold: BenchmarkData
    sa_co_silver: BenchmarkData
    sa_co_bronze: BenchmarkData
    sa_co_bio: BenchmarkData
  }
  box_detection: {
    lvis: BenchmarkData
    coco: BenchmarkData
    sa_co_gold: BenchmarkData
    sa_co_silver: BenchmarkData
    sa_co_bronze: BenchmarkData
    sa_co_bio: BenchmarkData
  }
  semantic_segmentation: {
    ade_847: BenchmarkData
    pc_59: BenchmarkData
    cityscapes: BenchmarkData
  }
}

function getNestedValue(obj: any, path: string) {
  return path.split('.').reduce((current, key) => current?.[key], obj)
}

function getSearchableText(result: PCSModelResult): string {
  const searchableFields = [
    result.metadata.model,
    result.metadata.license,
    result.metadata.architecture,
    result.metadata.param_count ? formatters.parameters(result.metadata.param_count) : '',
    result.metadata.param_count?.toString() || '',
    // Add all result values that exist
    result.results.cgf?.toString() || '',
    result.results.pmf?.toString() || '',
    result.results.ap?.toString() || '',
    result.results.ap_coco_o?.toString() || '',
    result.results.gold?.toString() || '',
    result.results.silver?.toString() || '',
    result.results.bronze?.toString() || '',
    result.results.bio?.toString() || '',
    result.results.miou?.toString() || '',
  ].filter(Boolean) // Remove empty strings

  return searchableFields.join(' ').toLowerCase()
}




export default function PCSClient() {
  // ============================================================================
  // STATE
  // ============================================================================
  const [search, setSearch] = useState("")
  const [sortColumn, setSortColumn] = useState<string>("results.cgf") // Default to CGF for SA-Co variants (will be updated based on benchmark)
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc")
  const [isLoading, setIsLoading] = useState(true)
  const [selectedBenchmark, setSelectedBenchmark] = useState("instance_segmentation.sa_co_gold")

  // Cast the imported data to our PCS type
  const pcsData = pcsResults as unknown as PCSData
  
  // Debug logging to help identify issues
  // useEffect(() => {
  //   console.log('PCS Data loaded:', {
  //     hasInstanceSegmentation: !!pcsData?.instance_segmentation,
  //     hasBoxDetection: !!pcsData?.box_detection,
  //     hasSemanticSegmentation: !!pcsData?.semantic_segmentation,
  //     selectedBenchmark
  //   })
  // }, [pcsData, selectedBenchmark])

  // ============================================================================
  // BENCHMARK GROUPS & CURRENT DATA
  // ============================================================================
  const benchmarkGroups = useMemo(() => [
    {
      groupName: "Instance Segmentation",
      benchmarks: [
        { key: "instance_segmentation.sa_co_gold", label: "SA-Co/Gold" },
        { key: "instance_segmentation.sa_co_silver", label: "SA-Co/Silver" },
        { key: "instance_segmentation.sa_co_bronze", label: "SA-Co/Bronze" },
        { key: "instance_segmentation.sa_co_bio", label: "SA-Co/Bio" },
        { key: "instance_segmentation.lvis", label: "LVIS" }
      ]
    },
    {
      groupName: "Box Detection",
      benchmarks: [
        { key: "box_detection.sa_co_gold", label: "SA-Co/Gold" },
        { key: "box_detection.sa_co_silver", label: "SA-Co/Silver" },
        { key: "box_detection.sa_co_bronze", label: "SA-Co/Bronze" },
        { key: "box_detection.sa_co_bio", label: "SA-Co/Bio" },
        { key: "box_detection.lvis", label: "LVIS" },
        { key: "box_detection.coco", label: "COCO" },
      ]
    },
    {
      groupName: "Semantic Segmentation",
      benchmarks: [
        { key: "semantic_segmentation.ade_847", label: "ADE-847" },
        { key: "semantic_segmentation.pc_59", label: "PC-59" },
        { key: "semantic_segmentation.cityscapes", label: "Cityscapes" }
      ]
    }
  ], [])

  // Get current benchmark data
  const currentBenchmarkData = useMemo(() => {
    const [task, benchmark] = selectedBenchmark.split('.')
    if (task === 'instance_segmentation' && pcsData.instance_segmentation) {
      return pcsData.instance_segmentation[benchmark as keyof typeof pcsData.instance_segmentation]
    } else if (task === 'box_detection' && pcsData.box_detection) {
      return pcsData.box_detection[benchmark as keyof typeof pcsData.box_detection]
    } else if (task === 'semantic_segmentation' && pcsData.semantic_segmentation) {
      return pcsData.semantic_segmentation[benchmark as keyof typeof pcsData.semantic_segmentation]
    }
    return null
  }, [selectedBenchmark, pcsData])

  const currentModels = currentBenchmarkData?.models || []

  // ============================================================================
  // DATA EXTRACTION & FILTERS
  // ============================================================================
  // Extract unique values for filter dropdowns from current models
  const availableLicenses = useUniqueValues(currentModels, (result: PCSModelResult) => result.metadata.license)
  const availableArchitectures: string[] = [] // Commented out: useUniqueValues(currentModels, (result: PCSModelResult) => result.metadata.architecture)
  const availablePretrainDatasets: string[] = [] // Commented out: useUniqueValues(currentModels, (result: PCSModelResult) => result.metadata.pretrain_datasets)

  // Initialize filter hooks
  const licenseFilter = useSetFilter(availableLicenses)
  const architectureFilter = useSetFilter(availableArchitectures) // Empty array - filter disabled
  const pretrainDatasetFilter = useSetFilter(availablePretrainDatasets) // Empty array - filter disabled

  // Calculate parameter range for slider - only for models with non-null param_count
  const { minParams, maxParams } = useMemo(() => {
    if (currentModels.length === 0) return { minParams: 0, maxParams: 100 }
    const paramCounts = currentModels
      .filter((result: PCSModelResult) => result.metadata.param_count !== null)
      .map((result: PCSModelResult) => result.metadata.param_count! / 1_000_000)
    if (paramCounts.length === 0) return { minParams: 0, maxParams: 100 }
    return { minParams: Math.min(...paramCounts), maxParams: Math.max(...paramCounts) }
  }, [currentModels])

  const parameterFilter = useRangeFilter(minParams, maxParams)
  const columnManager = usePCSColumnManager(selectedBenchmark)

  // ============================================================================
  // HANDLERS
  // ============================================================================
  const handleBenchmarkChange = (benchmark: string) => {
    setSelectedBenchmark(benchmark)
    // Set appropriate default sort column based on benchmark
    if (benchmark.includes('lvis')) {
      setSortColumn("results.ap")
    } else if (benchmark.includes('coco')) {
      setSortColumn("results.ap")
    } else if (benchmark.includes('sa_co_bio')) {
      setSortColumn("results.pmf")
    } else if (benchmark.includes('sa_co')) {
      setSortColumn("results.cgf")
    } else if (benchmark.includes('ade_847') || benchmark.includes('pc_59') || benchmark.includes('cityscapes')) {
      setSortColumn("results.miou")
    } else {
      setSortColumn("results.gold") // fallback
    }
    setSortDirection("desc")
  }
  
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
    let filtered = currentModels

    // Search filter
    if (search.trim()) {
      const searchLower = search.toLowerCase().trim()
      filtered = filtered.filter((result: PCSModelResult) => getSearchableText(result).includes(searchLower))
    }

    // License filter
    if (licenseFilter.selectedItems.size > 0) {
      filtered = filtered.filter((result: PCSModelResult) => licenseFilter.selectedItems.has(result.metadata.license))
    }

    // Architecture filter - commented out
    // if (architectureFilter.selectedItems.size > 0) {
    //   filtered = filtered.filter((result: PCSModelResult) => architectureFilter.selectedItems.has(result.metadata.architecture))
    // }

    // Pretrain datasets filter - commented out
    // if (pretrainDatasetFilter.selectedItems.size > 0) {
    //   filtered = filtered.filter((result: PCSModelResult) => {
    //     return result.metadata.pretrain_datasets?.some((dataset: string) =>
    //       pretrainDatasetFilter.selectedItems.has(dataset)
    //     )
    //   })
    // }

    // Parameter range filter - commented out since all param_count values are null
    // const [minParamsMillion, maxParamsMillion] = parameterFilter.value
    // filtered = filtered.filter((result: PCSModelResult) => {
    //   // If param_count is null, include the model (don't filter it out)
    //   if (result.metadata.param_count === null) return true
    //   
    //   // If param_count exists, apply the range filter
    //   const paramCountMillion = result.metadata.param_count / 1_000_000
    //   return paramCountMillion >= minParamsMillion && paramCountMillion <= maxParamsMillion
    // })

    // Apply sorting
    if (!sortColumn || !sortDirection) return filtered
    return [...filtered].sort((a, b) => {
      const aValue = getNestedValue(a, sortColumn)
      const bValue = getNestedValue(b, sortColumn)
      if (aValue < bValue) return sortDirection === "asc" ? -1 : 1
      if (aValue > bValue) return sortDirection === "asc" ? 1 : -1
      return 0
    })
  }, [currentModels, search, licenseFilter.selectedItems, /* architectureFilter.selectedItems, pretrainDatasetFilter.selectedItems, parameterFilter.value, */ sortColumn, sortDirection])

  // Calculate range for bar indicators with improved null handling
  const columnRange = useMemo(() => {
    if (!sortColumn || filteredAndSortedResults.length === 0) return null
    const nonNumericColumns = ['metadata.model', 'metadata.license', 'paper']
    if (nonNumericColumns.includes(sortColumn)) return null
    
    const values = filteredAndSortedResults.map((result: PCSModelResult) => getNestedValue(result, sortColumn))
    // Filter out null, undefined, NaN, and non-numeric values
    const numericValues = values.filter((v: any) => typeof v === 'number' && !isNaN(v) && isFinite(v))
    if (numericValues.length === 0) return null
    
    const min = Math.min(...numericValues)
    const max = Math.max(...numericValues)
    
    // Only return range if we have meaningful data
    if (min === max && numericValues.length === 1) return null
    
    return { min, max, column: sortColumn }
  }, [sortColumn, filteredAndSortedResults])

  // ============================================================================
  // RENDER
  // ============================================================================
  return (
    <>
      {/* Hero Section */}

      <TaskToggle 
        tasks={[
          { href: "/", label: "Object Detection" },
          { href: "/pcs", label: "Promptable Concept Segmentation" }
        ]}
      />

      {/* <section className="pt-6 sm:pt-12">
      <div className="container-base">
        <div className="flex">
          <div className="space-y-2">
            <Badge variant="primary">
              <Link href="https://github.com/roboflow/supervision" target="_blank" className="flex gap-1">
              Powered by supervision <ArrowSquareOutIcon size={14} weight="thin" />
              </Link>
            </Badge>
            <h1 className="text-3xl sm:text-4xl">
              <span className="text-primary-foreground">Promptable Concept Segmentation</span>
              Model Leaderboard
            </h1>
            <p className="prose prose-sm max-w-lg">
            Identify the location of objects in an image with pixel-level precision using text prompts, point prompts, or specific regions of an image as reference. Model performance is measured by how well the model accurately identifies real-world objects.
            </p>
          </div>
        </div>
      </div>
    </section> */}
      <HeroSection 
        title="Promptable Concept Segmentation"
        subtitle="Model Leaderboard"
        description="Identify the location of all instances of a visual concept in an image with pixel-level precision (or predict nothing if the concept does not exist) using text prompts, specific regions of an image as reference, or both. Model performance is measured by how well the model accurately identifies real-world objects in complex scenes."
        videoUrl="/model-leaderboard-fork/raccoon-tails-pcs.mp4"
        videoPosterUrl="/model-leaderboard-fork/raccoon-tails.avif"
      />

     

      {/* <TaskCard 
        title="Promptable Concept Segmentation"
        description="Identify the location of objects in an image with pixel-level precision using text prompts, point prompts, or specific regions of an image as reference. Model performance is measured by how well the model accurately identifies real-world objects."
        videoUrl="/model-leaderboard-fork/video-leaderboard-pcs.mp4"
        videoPosterUrl="/model-leaderboard-fork/video-leaderboard-pcs.avif"
      /> */}

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
                  label="Select Benchmark"
                  groupedItems={benchmarkGroups.map(group => ({
                    groupName: group.groupName,
                    items: group.benchmarks.map(b => ({
                      key: b.key,
                      label: b.label
                    }))
                  }))}
                  selectedItem={selectedBenchmark}
                  tag={(() => {
                    const group = benchmarkGroups.find(g => 
                      g.benchmarks.some(b => b.key === selectedBenchmark)
                    )
                    const benchmarkLabel = group?.benchmarks.find(b => b.key === selectedBenchmark)?.label
                    return group && benchmarkLabel ? `${group.groupName}: ${benchmarkLabel}` : selectedBenchmark
                  })()}
                  onItemChange={handleBenchmarkChange}
                />

                <Separator orientation="vertical" className="max-h-9" />

                <ColumnToggle
                  columns={columnManager.allColumns
                    .filter(col => col.key !== 'metadata.param_count') // Hide parameter column from toggle
                    .map(col => ({ key: col.key, label: col.label, group: col.group }))}
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
                architectureFilter={architectureFilter} // Empty filter - disabled
                pretrainDatasetFilter={pretrainDatasetFilter} // Empty filter - disabled
                parameterFilter={parameterFilter} // Keep for interface compatibility, but slider is commented out
                
                // Pass the available data
                availableLicenses={availableLicenses}
                availableArchitectures={availableArchitectures} // Empty array - disabled
                availablePretrainDatasets={availablePretrainDatasets} // Empty array - disabled
                
                // Benchmark selection (using grouped data)
                availableBenchmarks={benchmarkGroups.map(group => ({
                  groupName: group.groupName,
                  items: group.benchmarks.map(b => ({
                    key: b.key,
                    label: b.label
                  }))
                }))}
                selectedBenchmark={selectedBenchmark}
                onBenchmarkChange={handleBenchmarkChange}
                
                // Column management (already consolidated)
                columnManager={columnManager}
              />
            </div>

            {/* Results Table */}
            <ScrollArea className="max-h-[625px] w-full max-w-[1504px] overflow-x-auto rounded-md border whitespace-nowrap">
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
                    filteredAndSortedResults.map((result: PCSModelResult) => (
                      <ModelTableRow
                        key={result.metadata.model}
                        result={result as any} // Type assertion for compatibility
                        columns={columnManager.filteredColumns as any} // Type assertion for compatibility
                        sortColumn={sortColumn}
                        columnRange={columnRange}
                        selectedBenchmark={selectedBenchmark}
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
                        <span className="font-medium text-foreground">{currentModels.length}</span> models
                      </>
                    )}
                  </div>

                  {/* Active Filters Indicator */}
                  {!isLoading && (search || licenseFilter.selectedItems.size > 0 /* || architectureFilter.selectedItems.size > 0 || pretrainDatasetFilter.selectedItems.size > 0 || parameterFilter.isFiltered */) && (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span>Filtered by:</span>
                      {search && <span className="bg-muted px-2 py-0.5 rounded">Search</span>}
                      {licenseFilter.selectedItems.size > 0 && <span className="bg-muted px-2 py-0.5 rounded">License</span>}
                      {/* {architectureFilter.selectedItems.size > 0 && <span className="bg-muted px-2 py-0.5 rounded">Architecture</span>} */}
                      {/* {pretrainDatasetFilter.selectedItems.size > 0 && <span className="bg-muted px-2 py-0.5 rounded">Pretrained Datasets</span>} */}
                      {/* {parameterFilter.isFiltered && <span className="bg-muted px-2 py-0.5 rounded">Parameters</span>} */}
                    </div>
                  )}

                  <div className="text-xs text-muted-foreground">
                    <div>*Author reported, not yet independently validated</div>
                    <div>† partially trained on LVIS.</div>
                  </div>
                </div>
                <div className="prose prose-sm">Made with <HeartIcon size={14} weight="fill" className="inline-block text-primary-foreground" /> by <Link href="https://roboflow.com" target="_blank" className="link-primary">Roboflow</Link></div>
              </div>
          </div>

        </div>
      </section>
    </>
  )
}
