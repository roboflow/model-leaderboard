import { useState, useMemo } from 'react'
import { allColumns, getDefaultVisibleColumns, getAllColumnKeys } from '@/lib/columns'

/**
 * Hook for managing table column visibility and related functionality
 */
export function useColumnManager() {
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(getDefaultVisibleColumns())

  // Get filtered columns based on visibility
  const filteredColumns = useMemo(() => 
    allColumns.filter(col => visibleColumns.has(col.key)),
    [visibleColumns]
  )

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
    setVisibleColumns(new Set(getAllColumnKeys()))
  }

  const hideAllColumns = () => {
    setVisibleColumns(new Set(['metadata.model'])) // Keep at least model column
  }

  const resetToDefaults = () => {
    setVisibleColumns(getDefaultVisibleColumns())
  }

  return {
    // State
    visibleColumns,
    allColumns,
    filteredColumns,
    
    // Handlers
    toggleColumn,
    showAllColumns,
    hideAllColumns,
    resetToDefaults
  }
} 