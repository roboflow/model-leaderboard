"use client"
import { useState } from "react"
import { RoboflowLogo } from "./RoboflowLogo"
import { ThemeToggle } from "./ThemeToggle"
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from "@/components/ui/navigation-menu"
import Link from "next/link"
import { SiGithub } from "@icons-pack/react-simple-icons"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { ListIcon } from "@phosphor-icons/react"
import { Separator } from "@/components/ui/separator"
import { usePathname } from "next/navigation";

export function Header() {
  const [isOpen, setIsOpen] = useState(false)
  const pathname = usePathname();

  return (
    <header className="py-4">
      <div className="container-base mx-auto flex justify-between items-center">
        <Link href="https://roboflow.com" className="flex items-center" aria-label="Visit Roboflow website">
          <RoboflowLogo
            className="text-violet-700 dark:text-white h-6 w-auto"
          />
        </Link>

        <div className="hidden sm:flex items-center gap-2 ">
          <NavigationMenu>
            <NavigationMenuList>
              {[
                { href: "/", label: "Leaderboard" },
                { href: "/methodology/", label: "Methodology" },
                { href: "/faq/", label: "FAQ" },
              ].map(({ href, label }) => (
                <NavigationMenuItem key={href}>
                  <NavigationMenuLink 
                  data-active={pathname === href}
                  className="bg-transparent focus-visible:bg-transparent hover:bg-transparent hover:text-primary-foreground data-[active=true]:bg-transparent data-[active=true]:focus:bg-transparent data-[active=true]:hover:bg-transparent data-[active=true]:text-primary-foreground focus:bg-transparent" asChild>
                    <Link href={href}>{label}</Link>
                  </NavigationMenuLink>
                </NavigationMenuItem>
              ))}
               <NavigationMenuItem >
                <NavigationMenuLink className="hover:bg-transparent hover:text-primary-foreground focus:bg-transparent" asChild>
                  <Link href="https://github.com/roboflow/model-leaderboard" target="_blank">
                    <SiGithub size={16} className="text-foreground" />
                  </Link>
                </NavigationMenuLink>
              </NavigationMenuItem>
            </NavigationMenuList>
          </NavigationMenu>

          <ThemeToggle />
        </div>
        <div className="sm:hidden">
          <Popover open={isOpen} onOpenChange={setIsOpen}>
            <PopoverTrigger><ListIcon size={24} /></PopoverTrigger>
            <PopoverContent className="w-screen md:hidden rounded-none border-none bg-background/10 backdrop-blur-lg text-foreground h-screen px-4 sm:px-6 lg:px-8 py-8">
              <div className="flex flex-col gap-0.25">
                <Link className="flex py-4 px-1.5 nav-menu-link" href="/" onClick={() => setIsOpen(false)}>Leaderboard</Link>
                <Separator />
                <Link className="flex py-4 px-1.5 nav-menu-link" href="/methodology" onClick={() => setIsOpen(false)}>Methodology</Link>
                <Separator />
                <Link className="flex py-4 px-1.5 nav-menu-link" href="/faq" onClick={() => setIsOpen(false)}>FAQ</Link>
                <Separator />
                <div className="flex justify-between items-center py-4 px-1.5">
                  <div className="text-sm text-muted-foreground">Appearance</div>
                  <ThemeToggle />
                </div>
                <Separator />
                <div className="flex justify-center items-center py-4 px-1.5">
                  <Link className="py-2 px-1.5" href="https://github.com/roboflow/model-leaderboard" target="_blank" onClick={() => setIsOpen(false)}>
                    <SiGithub size={24} className="text-foreground/60" />
                  </Link>
                </div>
              </div>
            </PopoverContent>
          </Popover>
        </div>
      </div>
    </header>
  )
}
