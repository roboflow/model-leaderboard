import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Methodology - Computer Vision Model Leaderboard',
  description: 'Learn about the methodology behind the Computer Vision Model Leaderboard, including how benchmarks are calculated.',
  openGraph: {
    title: 'Methodology - Computer Vision Model Leaderboard',
    description: 'Learn about the methodology behind the Computer Vision Model Leaderboard, including how benchmarks are calculated.',
  },
  twitter: {
    title: 'Methodology - Computer Vision Model Leaderboard',
    description: 'Learn about the methodology behind the Computer Vision Model Leaderboard, including how benchmarks are calculated.',
  },
}

export default function MethodologyPage() {
  return (
    <section className="py-pt-6 sm:pt-12">
      <div className="container-narrow">
        <div className="space-y-6">
          <h1 className="text-3xl sm:text-4xl">Methodology</h1>
          <div className="prose prose-sm max-w-lg">
            <p>The Roboflow computer vision model leaderboard benchmarks popular object detection models against the Microsoft COCO dataset. The Microsoft COCO dataset is commonly used to evaluate and compare the performance of object detection models.</p>

            <p>Benchmark data in the table was computed independently by the Roboflow team, following public inference instructions from each model vendor. We aim to achieve as close to the original benchmark results as possible by following all instructions.</p>

            <p>This project is open source, with code available that we use for benchmarking. This means you can verify the results of the data in the leaderboard table.</p>

            <p>We used the validation set of the COCO dataset, to evaluate model performance on common objects. This means that the benchmark is less relevant for evaluating domain adaptiveness: how a new architecture does on a specific domain.</p>

            <p>The Roboflow 100 benchmark was designed to measure model performance across domains. If you are interested in learning more about domain-specific model benchmarking, refer to the Roboflow 100 website.</p>
          </div>
       </div>
      </div>
    </section>
  )
}
