"use client";

import * as React from "react";
import {
  FadersIcon,
  FileTextIcon,
  CircuitryIcon,
  DatabaseIcon,
  FunnelIcon,
  ColumnsPlusRightIcon
} from "@phosphor-icons/react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/useMobile";
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import { Separator } from "@/components/ui/separator";
import { DropdownFilterSlider } from "@/components/DropdownFilterSlider";
import { ColumnToggle } from "@/components/ColumnToggle";
import { DropdownFilterCheckbox } from "@/components/DropdownFilterCheckbox";
import { DropdownFilterRadio } from "@/components/DropdownFilterRadio";

import { CpuIcon, GaugeIcon } from "@phosphor-icons/react";
import { useRangeFilter } from "@/hooks/useRangeFilter";
import { formatters } from "@/lib/formatters";
import { useSetFilter } from "@/hooks/useSetFilter";
import { useColumnManager } from "@/hooks/useColumnManager";

interface MobileControlsProps {
  // Filter objects
  licenseFilter: ReturnType<typeof useSetFilter>;
  architectureFilter: ReturnType<typeof useSetFilter>;
  pretrainDatasetFilter: ReturnType<typeof useSetFilter>;
  parameterFilter: ReturnType<typeof useRangeFilter>;

  // Available data
  availableLicenses: string[];
  availableArchitectures: string[];
  availablePretrainDatasets: string[];

  // Dataset selection
  availableDatasets: string[];
  selectedDataset: string;
  onDatasetChange: (dataset: string) => void;

  // Column management
  columnManager: ReturnType<typeof useColumnManager>;
}

export function MobileControls({
  licenseFilter,
  architectureFilter,
  pretrainDatasetFilter,
  parameterFilter,
  availableLicenses,
  availableArchitectures,
  availablePretrainDatasets,
  availableDatasets,
  selectedDataset,
  onDatasetChange,
  columnManager,
}: MobileControlsProps) {
  const isMobile = useIsMobile();

  if (!isMobile) return null;

  return (
    <div className="sm:hidden">
      <Drawer>
        <DrawerTrigger asChild>
          <Button
            variant="transparent"
            size="sm"
            className="inline-flex items-center gap-2 sm:hidden"
          >
            <FadersIcon size={16} />
          </Button>
        </DrawerTrigger>
        <DrawerContent className="max-h-[90vh] sm:hidden">
          <DrawerHeader>
            <DrawerTitle>Filters & Column Settings</DrawerTitle>
            <DrawerDescription>
              Customize your view with filters and column visibility options.
            </DrawerDescription>
          </DrawerHeader>

          <div className="overflow-y-auto px-4 pb-4">
            <div className="space-y-6">
              {/* Filters Section */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <FunnelIcon size={16} />
                  <h3 className="font-semibold">Filters</h3>
                </div>
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

                <Separator />

                <DropdownFilterSlider
                  icon={CpuIcon}
                  title="Parameters"
                  label="Filter by Parameter Count"
                  formatter={formatters.parametersFromMillion}
                  step={0.1}
                  {...parameterFilter}
                />
                <Separator />

                <DropdownFilterCheckbox
                  icon={DatabaseIcon}
                  title="Pretrained Datasets"
                  label="Filter by Pretrained Datasets"
                  availableItems={availablePretrainDatasets}
                  selectedItems={pretrainDatasetFilter.selectedItems}
                  onItemToggle={pretrainDatasetFilter.toggleItem}
                  onClearAll={pretrainDatasetFilter.clearAll}
                  onSelectAll={pretrainDatasetFilter.selectAll}
                />
                <Separator />

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

                <Separator />

                <DropdownFilterRadio
                  icon={GaugeIcon}
                  title="Benchmark"
                  label="Select Dataset"
                  availableItems={availableDatasets}
                  selectedItem={selectedDataset}
                  onItemChange={onDatasetChange}
                />
              </div>

              <Separator />

              {/* Column Toggle Section */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <ColumnsPlusRightIcon size={16} />
                  <h3 className="font-semibold">Columns</h3>
                </div>

                <ColumnToggle
                  columns={columnManager.allColumns.map((col) => ({
                    key: col.key,
                    label: col.label,
                    group: col.group,
                  }))}
                  visibleColumns={columnManager.visibleColumns}
                  onToggleColumn={columnManager.toggleColumn}
                  onShowAll={columnManager.showAllColumns}
                  onHideAll={columnManager.hideAllColumns}
                  onResetToDefaults={columnManager.resetToDefaults}
                />
              </div>

              {/* Close Button */}
              <div className="pt-4">
                <DrawerClose asChild>
                  <Button variant="outline" className="w-full">
                    Done
                  </Button>
                </DrawerClose>
              </div>
            </div>
          </div>
        </DrawerContent>
      </Drawer>
    </div>
  );
}
