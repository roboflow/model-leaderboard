import { useState, useMemo } from "react"

export function useFilter<T extends string>(
  allItems: T[],
  initialSelected: Set<T> = new Set()
) {
  const [selectedItems, setSelectedItems] = useState<Set<T>>(initialSelected)

  const handlers = useMemo(() => ({
    onItemToggle: (item: T) => {
      const newSelected = new Set(selectedItems)
      if (newSelected.has(item)) {
        newSelected.delete(item)
      } else {
        newSelected.add(item)
      }
      setSelectedItems(newSelected)
    },
    onClearAll: () => setSelectedItems(new Set()),
    onSelectAll: () => setSelectedItems(new Set(allItems)),
  }), [selectedItems, allItems])

  return {
    selectedItems,
    availableItems: allItems,
    ...handlers,
  }
}
