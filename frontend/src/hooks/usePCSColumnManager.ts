import { useState, useMemo, useEffect } from 'react'
import { 
  pcsColumns, 
  getColumnsForBenchmark, 
  getDefaultVisibleColumnsForBenchmark, 
  getAllColumnKeysForBenchmark,
  type PCSColumn 
} from '@/lib/pcs-columns'

/**
 * Hook for managing PCS table column visibility with benchmark-specific columns
 */
export function usePCSColumnManager(selectedBenchmark: string) {
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(new Set())

  // Get columns available for current benchmark
  const availableColumns = useMemo(() => 
    getColumnsForBenchmark(selectedBenchmark),
    [selectedBenchmark]
  )

  // Get filtered columns based on visibility and benchmark
  const filteredColumns = useMemo(() => 
    availableColumns.filter(col => visibleColumns.has(col.key)),
    [availableColumns, visibleColumns]
  )

  // Reset visible columns when benchmark changes
  useEffect(() => {
    const defaultColumns = getDefaultVisibleColumnsForBenchmark(selectedBenchmark)
    setVisibleColumns(defaultColumns)
  }, [selectedBenchmark])

  // Column manipulation handlers
  const toggleColumn = (columnKey: string) => {
    const newVisibleColumns = new Set(visibleColumns)
    if (newVisibleColumns.has(columnKey)) {
      newVisibleColumns.delete(columnKey)
    } else {
      newVisibleColumns.add(columnKey)
    }
    setVisibleColumns(newVisibleColumns)
  }

  const showAllColumns = () => {
    const allKeys = getAllColumnKeysForBenchmark(selectedBenchmark)
    setVisibleColumns(new Set(allKeys))
  }

  const hideAllColumns = () => {
    setVisibleColumns(new Set(['metadata.model'])) // Keep at least model column
  }

  const resetToDefaults = () => {
    const defaultColumns = getDefaultVisibleColumnsForBenchmark(selectedBenchmark)
    setVisibleColumns(defaultColumns)
  }

  return {
    // State
    visibleColumns,
    allColumns: availableColumns,
    filteredColumns,
    
    // Handlers
    toggleColumn,
    showAllColumns,
    hideAllColumns,
    resetToDefaults
  }
}
