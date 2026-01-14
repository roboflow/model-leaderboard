interface TaskCardProps {
  title: string
  description: string
  videoUrl: string
  videoPosterUrl: string
}

export function TaskCard({ title, description, videoUrl, videoPosterUrl }: TaskCardProps) {
  return (
    <section className="pb-0">
      <div className="container-base mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 border border-border rounded-md overflow-hidden">
          <div className="p-6 sm:p-12 space-y-2">
            <h2 className="text-xl sm:text-2xl text-primary-foreground">{title}</h2>
            <p className="prose prose-sm max-w-lg">
              {description}
            </p>
          </div>
          <div className="w-full h-full aspect-video bg-foreground/5">
            <video autoPlay muted className="w-full h-full object-cover" poster={videoPosterUrl}>
              <source src={videoUrl} type="video/mp4" />
            </video>
          </div>
        </div>
      </div>
    </section>
  )
}
