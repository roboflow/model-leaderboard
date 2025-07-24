import { useState } from 'react'

/**
 * Generic hook for managing Set-based filters
 * @param availableItems - Array of all available items to filter
 * @returns Object with selected set, handlers, and computed properties
 */
export function useSetFilter(availableItems: string[]) {
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set())

  const toggleItem = (item: string) => {
    const newSelected = new Set(selectedItems)
    if (newSelected.has(item)) {
      newSelected.delete(item)
    } else {
      newSelected.add(item)
    }
    setSelectedItems(newSelected)
  }

  const clearAll = () => {
    setSelectedItems(new Set())
  }

  const selectAll = () => {
    setSelectedItems(new Set(availableItems))
  }

  const hasSelections = selectedItems.size > 0
  const isAllSelected = selectedItems.size === availableItems.length

  return {
    selectedItems,
    toggleItem,
    clearAll,
    selectAll,
    hasSelections,
    isAllSelected
  }
} 