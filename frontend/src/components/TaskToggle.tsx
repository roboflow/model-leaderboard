"use client"

import { Button } from "@/components/ui/button"
import { Section } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"

interface TaskToggleProps {
  tasks: {
    href: string
    label: string
  }[]
}

export function TaskToggle({ tasks }: TaskToggleProps) {
  const pathname = usePathname()

  return (
    // <div className="mt-6 sm:mt-12 flex justify-center p-1 bg-muted dark:bg-muted/80 w-full max-w-[280px] md:max-w-sm mx-auto rounded-full">
    //   {tasks.map((task) => (
    //     <Button 
    //       key={task.href}
    //       asChild 
    //       className={`w-full rounded-full ${
    //         pathname === task.href || (task.href !== "/" && pathname.startsWith(task.href))
    //           ? "bg-violet-500/80 text-violet-50 hover:bg-violet-500/80" 
    //           : "bg-transparent text-muted-foreground hover:bg-foreground/5"
    //       }`}
    //     >
    //       <Link href={task.href}>
    //         {task.label}
    //       </Link>
    //     </Button>
    //   ))}
    // </div>
    <section className="">
      <div className="container-base">
        <div className="w-full overflow-x-auto border-b border-border [&::-webkit-scrollbar]:h-1 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-muted-foreground/20 [&::-webkit-scrollbar-thumb]:rounded-full hover:[&::-webkit-scrollbar-thumb]:bg-muted-foreground/40">
          <div className="flex gap-3 min-w-max">
            {tasks.map((task) => (
              <Button 
                key={task.href}
                asChild 
                className={cn(`border-b rounded-none bg-transparent whitespace-nowrap ${
                  pathname === task.href || (task.href !== "/" && pathname.startsWith(task.href))
                    ? "border-primary-foreground hover:bg-transparent" 
                    : "bg-transparent border-transparent text-muted-foreground hover:text-foreground hover:bg-transparent"
                }`)}
              >
                <Link href={task.href}>
                  {task.label}
                </Link>
              </Button>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
