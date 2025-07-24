export interface Column {
  key: string
  label: string
  width: string
  sortable?: boolean
  group: 'Basic' | 'Core Metrics' | 'F1 Metrics' | 'Size-Specific' | 'Metadata'
  defaultVisible: boolean
  tooltip?: string
}

export const allColumns: Column[] = [
  {
    key: "metadata.model",
    label: "Model",
    width: "w-50",
    group: 'Basic',
    defaultVisible: true,
    tooltip: "Name of the computer vision model"
  },
  {
    key: "metadata.param_count",
    label: "Parameters (M)",
    width: "w-48",
    group: 'Basic',
    defaultVisible: true,
    tooltip: "Total number of trainable parameters in millions"
  },
  {
    key: "map50_95",
    label: "mAP 50:95",
    width: "w-40",
    group: 'Core Metrics',
    defaultVisible: true,
    tooltip: "Mean Average Precision at IoU thresholds from 0.5 to 0.95 (primary COCO metric)"
  },
  {
    key: "map50",
    label: "mAP 50",
    width: "w-40",
    group: 'Core Metrics',
    defaultVisible: true,
    tooltip: "Mean Average Precision at IoU threshold of 0.5"
  },
  {
    key: "map75",
    label: "mAP 75",
    width: "w-40",
    group: 'Core Metrics',
    defaultVisible: false,
    tooltip: "Mean Average Precision at IoU threshold of 0.75 (stricter localization)"
  },
  {
    key: "small_objects.map50_95",
    label: "mAP 50:95 (Small)",
    width: "w-40",
    group: 'Size-Specific',
    defaultVisible: false,
    tooltip: "mAP 50:95 for small objects"
  },
  {
    key: "medium_objects.map50_95",
    label: "mAP 50:95 (Medium)",
    width: "w-40",
    group: 'Size-Specific',
    defaultVisible: false,
    tooltip: "mAP 50:95 for medium objects"
  },
  {
    key: "large_objects.map50_95",
    label: "mAP 50:95 (Large)",
    width: "w-40",
    group: 'Size-Specific',
    defaultVisible: false,
    tooltip: "mAP 50:95 for large objects"
  },
  {
    key: "f1_50",
    label: "F1 50",
    width: "w-40",
    group: 'F1 Metrics',
    defaultVisible: false,
    tooltip: "F1 score at IoU threshold of 0.5"
  },
  {
    key: "f1_75",
    label: "F1 75",
    width: "w-40",
    group: 'F1 Metrics',
    defaultVisible: false,
    tooltip: "F1 score at IoU threshold of 0.75"
  },
  {
    key: "f1_small_objects.f1_50",
    label: "F1 50 (Small)",
    width: "w-40",
    group: 'Size-Specific',
    defaultVisible: false,
    tooltip: "F1 score at IoU 0.5 for small objects"
  },
  {
    key: "f1_small_objects.f1_75",
    label: "F1 75 (Small)",
    width: "w-40",
    group: 'Size-Specific',
    defaultVisible: false,
    tooltip: "F1 score at IoU 0.75 for small objects"
  },
  {
    key: "f1_medium_objects.f1_50",
    label: "F1 50 (Medium)",
    width: "w-40",
    group: 'Size-Specific',
    defaultVisible: false,
    tooltip: "F1 score at IoU 0.5 for medium objects"
  },
  {
    key: "f1_medium_objects.f1_75",
    label: "F1 75 (Medium)",
    width: "w-28",
    group: 'Size-Specific',
    defaultVisible: false,
    tooltip: "F1 score at IoU 0.75 for medium objects"
  },
  {
    key: "f1_large_objects.f1_50",
    label: "F1 50 (Large)",
    width: "w-24",
    group: 'Size-Specific',
    defaultVisible: false,
    tooltip: "F1 score at IoU 0.5 for large objects"
  },
  {
    key: "f1_large_objects.f1_75",
    label: "F1 75 (Large)",
    width: "w-24",
    group: 'Size-Specific',
    defaultVisible: false,
    tooltip: "F1 score at IoU 0.75 for large objects"
  },
  {
    key: "paper",
    label: "Paper",
    width: "w-10",
    sortable: false,
    group: 'Metadata',
    defaultVisible: true,
    tooltip: "Link to research paper"
  },
  {
    key: "metadata.license",
    label: "License",
    width: "w-10",
    group: 'Metadata',
    defaultVisible: true,
    tooltip: "Software license (e.g., MIT, Apache-2.0, AGPL-3.0)"
  },
]

/**
 * Get default visible columns based on defaultVisible property
 */
export const getDefaultVisibleColumns = (): Set<string> => {
  return new Set(allColumns.filter(col => col.defaultVisible).map(col => col.key))
}

/**
 * Get all column keys
 */
export const getAllColumnKeys = (): string[] => {
  return allColumns.map(col => col.key)
} 