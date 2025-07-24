import { TableCell, TableRow } from "./ui/table";
import { Column } from "@/lib/columns";


export function SkeletonTable({ columns }: { columns: Column[] }) {
  return (
    <>
      {Array.from({ length: 13 }).map((_, i) => (
        <TableRow key={i}>
          {columns.map((column) => (
            <TableCell key={column.key} className={`${column.width} px-4 h-[45px]`}>
              <div className="h-4 bg-muted/50 animate-pulse rounded w-16"></div>
            </TableCell>
          ))}
        </TableRow>
      ))}
    </>
  )
}