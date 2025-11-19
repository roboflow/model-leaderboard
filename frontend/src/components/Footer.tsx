import { ThemeToggle } from "@/components/ThemeToggle"

export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="py-4">
      <div className="container-base">
        <div className="flex justify-between">
          <p className="text-xs text-muted-foreground">© {currentYear} Roboflow, Inc. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}
