export const formatters = {
  /**
   * Format parameter count in millions (e.g., 25.3M)
   */
  parameters: (count: number): string => `${(count / 1_000_000).toFixed(1)}M`,

  /**
   * Format parameter count when already in millions (e.g., 25.3M)
   */
  parametersFromMillion: (count: number): string => `${count.toFixed(1)}M`,

  /**
   * Format decimal as percentage (e.g., 0.753 → 75.3%)
   */
  percentage: (value: number): string => `${(value * 100).toFixed(1)}%`,

  /**
   * Format number to fixed decimal places (default 3)
   */
  decimal: (value: number, places: number = 3): string => value.toFixed(places),

  /**
   * Format file size in appropriate units
   */
  fileSize: (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)}${units[unitIndex]}`;
  },
} as const;
