import type { Metadata } from 'next'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import Link from "next/link"

export const metadata: Metadata = {
  title: 'FAQ - Computer Vision Model Leaderboard',
  description: 'Frequently asked questions about the Computer Vision Model Leaderboard, covering metrics, models, and usage.',
  openGraph: {
    title: 'FAQ - Computer Vision Model Leaderboard',
    description: 'Frequently asked questions about the Computer Vision Model Leaderboard, covering metrics, models, and usage.',
  },
  twitter: {
    title: 'FAQ - Computer Vision Model Leaderboard',
    description: 'Frequently asked questions about the Computer Vision Model Leaderboard, covering metrics, models, and usage.',
  },
}

declare global {
  namespace JSX {
    interface IntrinsicElements {
      math: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
        xmlns?: string
      }
      mrow: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement>
      mi: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement>
      mo: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement>
      mn: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement>
      mfrac: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
        linethickness?: string
      }
    }
  }
}



export default function FAQPage() {
  return (
    <>
      <section className="pt-6 sm:pt-12">
        <div className="container-narrow space-y-6">
          <div className="space-y-6">
            <h1 className="text-3xl sm:text-4xl">Frequently Asked Questions</h1>
            <h2 className="prose prose-sm">
              Get answers to common questions about benchmarks, datasets, metrics, and how the leaderboard works.
            </h2>
          </div>
        </div>
      </section>

      <section>
        <div className="container-narrow">
          <Accordion type="single" collapsible>
            <AccordionItem value="item-1">
              <AccordionTrigger className="text-lg font-medium hover:no-underline">What is mAP?</AccordionTrigger>
              <AccordionContent className="prose prose-sm">
                <p>Mean Average Precision (mAP) is a metric used to evaluate the object detection models. It is the average of the precision-recall curves at different IoU thresholds.</p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-2">
              <AccordionTrigger className="text-lg font-medium hover:no-underline">What is the difference between mAP 50, mAP 75 and mAP 50:95?</AccordionTrigger>
              <AccordionContent className="prose prose-sm">
              <p>mAP can be evaluated at multiple <Link href="https://blog.roboflow.com/mean-average-precision/#:~:text=The%20Intersection%20over%20Union" target="_blank" className="link-primary">IoU thresholds</Link>. mAP 50, for example, evaluates it while considering detections that overlap with an IoU of 0.5 or greater - everything else is a false positive. mAP 50:95 is an average of all considered IoU thresholds - 0.5, 0.6, … 0.8, 0.9, 0.95. It is the primary metric showing how well the model performs, across increasing levels of rigour.</p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-3">
              <AccordionTrigger className="text-lg font-medium hover:no-underline">What do the small, medium, and large labels next to the mAP scores mean?</AccordionTrigger>
              <AccordionContent className="prose prose-sm">
              <p>The small, medium, and large labels next to the mAP scores indicate the size of the objects in the images. This is important because object detection models can struggle with detecting small objects. The COCO dataset has three categories of object sizes: small (less than 32x32 pixels), medium (between 32x32 and 96x96 pixels), and large (greater than 96x96 pixels). You can learn more about the definition on the <Link href="https://cocodataset.org/#detection-eval" target="_blank" className="link-primary">COCO detection evaluation description page</Link>.</p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-4">
              <AccordionTrigger className="text-lg font-medium hover:no-underline">What is F1 Score?</AccordionTrigger>
              <AccordionContent className="prose prose-sm">
                <p>F1 Score is a metric used to evaluate the object detection models. It is the harmonic mean of precision and recall. Recall measures how many of the target objects were detected. Even if the model detected more cars than there were in the image, but captured all targets among them - the recall will be 100%. Precision measures how many of the detected objects are correct. If the model found a box on some cars in an image, but classified 20% as bicycles, precision is 80%, regardless of how many were found. What if you want high recall and high precision? F1 Score simply combines the two into a single metric. With a high F1 score, you can be sure that model produced both high precision and recall in its results.</p>

                <p>Here is the formula for F1 Score, where P is precision and R is recall:</p>

                <div className="flex justify-center my-6">
                  <div className="text-3xl font-mono text-center">
                    <span className="font-bold">F1 = </span>
                    <div className="inline-block align-middle mx-2">
                      <div className="border-b-2 border-foreground pb-1 text-xl text-center">2PR</div>
                      <div className="text-xl text-center pt-1">P + R</div>
                    </div>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-5">
              <AccordionTrigger className="text-lg font-medium hover:no-underline">Is this project open source?</AccordionTrigger>
              <AccordionContent className="prose prose-sm">
                <p>Yes, this project is open source. You can find the code on <Link href="https://github.com/roboflow/model-leaderboard" target="_blank" className="link-primary">GitHub</Link>.</p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-6">
              <AccordionTrigger className="text-lg font-medium hover:no-underline">What is the slider icon next to model names?</AccordionTrigger>
              <AccordionContent className="prose prose-sm">
                <p>Hovering over it shows the parameters used to run the model. We aim to make the parameters as similar as possible to the ones used by the original authors.</p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-7">
              <AccordionTrigger className="text-lg font-medium hover:no-underline">What parameters were used to run each model?</AccordionTrigger>
              <AccordionContent className="prose prose-sm">
                <p>We aim to make the parameters as similar as possible to the ones used by the original authors. Hover over the slider icon next to the model name to see the parameters used.</p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-8">
              <AccordionTrigger className="text-lg font-medium hover:no-underline">Can I suggest a model to benchmark?</AccordionTrigger>
              <AccordionContent className="prose prose-sm">
                <p>Yes! If there is a model that you would like to see benchmarked, you may <Link href="https://github.com/roboflow/model-leaderboard/pulls" target="_blank" className="link-primary">open an PR</Link> with the model instructions. Please have a look at the <Link href="https://github.com/roboflow/model-leaderboard?tab=readme-ov-file#model-leaderboard-website" target="_blank" className="link-primary">README</Link> to learn about the structure of the repository.</p>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </section>
    </>
  )
}
