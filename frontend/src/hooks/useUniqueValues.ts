import { useMemo } from 'react'

type ExtractorFunction<T, R> = (item: T) => R | R[] | null | undefined

/**
 * Generic hook to extract unique values from an array of data
 * @param data - Array of data to extract from
 * @param extractor - Function that extracts value(s) from each item
 * @returns Sorted array of unique values
 */
export function useUniqueValues<T, R extends string>(
  data: T[],
  extractor: ExtractorFunction<T, R>
): R[] {
  return useMemo(() => {
    const uniqueValues = new Set<R>()
    
    data.forEach(item => {
      const extracted = extractor(item)
      
      if (extracted) {
        if (Array.isArray(extracted)) {
          // Handle array values (like pretrain_datasets)
          extracted.forEach(value => {
            if (value) uniqueValues.add(value)
          })
        } else {
          // Handle single values (like license, architecture)
          uniqueValues.add(extracted)
        }
      }
    })
    
    return Array.from(uniqueValues).sort()
  }, [data, extractor])
} 