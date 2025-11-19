// hooks/useRangeFilter.ts
import { useState, useEffect } from "react"

export function useRangeFilter(min: number, max: number) {
  const [value, setValue] = useState<[number, number]>([min, max])
  const [hasUserChanged, setHasUserChanged] = useState(false)

  // Update when min/max props change
  useEffect(() => {
    setValue([min, max])
    setHasUserChanged(false)
  }, [min, max])

  const onValueChange = (newRange: [number, number]) => {
    setHasUserChanged(true)
    setValue(newRange)
  }

  const onReset = () => {
    setHasUserChanged(false)
    setValue([min, max])
  }

  const isFiltered = hasUserChanged && (value[0] !== min || value[1] !== max)

  return {
    value,
    onValueChange,
    onReset,
    isFiltered,
    min,
    max,
  }
}
