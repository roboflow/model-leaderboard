"use client"

import { Button } from "@/components/ui/button"
import Link from "next/link"
import { usePathname } from "next/navigation"

interface TaskToggleProps {
  tasks: {
    href: string
    label: string
  }[]
}

export function TaskToggle({ tasks }: TaskToggleProps) {
  const pathname = usePathname()

  return (
    <div className="flex justify-center p-1 bg-muted w-full max-w-sm mx-auto rounded-full">
      {tasks.map((task) => (
        <Button 
          key={task.href}
          asChild 
          className={`w-full rounded-full ${
            pathname === task.href || (task.href !== "/" && pathname.startsWith(task.href))
              ? "bg-primary text-primary-foreground" 
              : "bg-transparent text-muted-foreground hover:bg-foreground/5"
          }`}
        >
          <Link href={task.href}>
            {task.label}
          </Link>
        </Button>
      ))}
    </div>
  )
}
