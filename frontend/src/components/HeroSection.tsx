import { Badge } from "@/components/ui/badge"
import Link from "next/link"
import { ArrowSquareOutIcon } from "@phosphor-icons/react"

interface HeroSectionProps {
  title: string
  subtitle?: string
  description: string
  badgeText?: string
  badgeHref?: string
  videoUrl?: string
  videoPosterUrl?: string
}

export function HeroSection({ 
  title, 
  subtitle, 
  description, 
  badgeText = "Powered by supervision",
  badgeHref = "https://github.com/roboflow/supervision",
  videoUrl,
  videoPosterUrl
}: HeroSectionProps) {
  return (
    // <section className="pt-6 sm:pt-12">
    //   <div className="container-base mx-auto">
    //     <div className="flex justify-center">
    //       <div className="space-y-2 text-center">
    //         <Badge variant="primary">
    //           <Link href={badgeHref} target="_blank" className="flex items-center gap-1">
    //             {badgeText} <ArrowSquareOutIcon size={14} weight="thin" />
    //           </Link>
    //         </Badge>
    //         <h1 className="text-3xl sm:text-4xl">
    //           <span className="text-primary-foreground">{title}</span>
    //           {subtitle && <><br />{subtitle}</>}
    //         </h1>
    //         <p className="prose prose-sm max-w-lg">
    //           {description}
    //         </p>
    //       </div>
    //     </div>
    //   </div>
    // </section>
    <section className="pt-6">
        <div className="container-base">
          <div className="flex md:flex-row flex-col justify-between md:gap-16 gap-6">
            <div className="space-y-2 max-w-lg md:w-full">
              <Badge variant="primary">
                <Link href={badgeHref} target="_blank" className="flex items-center gap-1">
                  {badgeText} <ArrowSquareOutIcon size={14} weight="thin" />
                </Link>
              </Badge>
              <h1 className="text-3xl sm:text-4xl">
                <span className="text-primary-foreground">{title}</span> {subtitle}
              </h1>
              <p className="prose prose-sm">{description}</p>
            </div>
            <div className="hidden md:block md:max-w-[500px] h-full aspect-auto bg-foreground/5 rounded-md overflow-hidden">
            <video autoPlay muted className="w-full h-full object-cover" poster={videoPosterUrl}>
              <source src={videoUrl} type="video/mp4" />
            </video>
          </div>
          </div>
        </div>
      </section>
  )
}
