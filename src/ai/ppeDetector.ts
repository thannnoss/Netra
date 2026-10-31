
import * as ort from 'onnxruntime-web'
import { loadModel } from './onnxClient'

const MODEL_SIZE = 640
const LETTERBOX_COLOR = 114
const CONFIDENCE_THRESHOLD = 0.25
const NMS_THRESHOLD = 0.45
const NUM_CLASSES = 17
const NUM_CANDIDATES = 8400

export const CLASS_NAMES = [
  'helmet',
  'gloves',
  'vest',
  'boots',
  'goggles',
  'none',
  'Person',
  'no_helmet',
  'no_goggle',
  'no_gloves',
  'no_boots',
  'Mask',
  'NO-Mask',
  'NO-Safety Vest',
  'Safety Cone',
  'machinery',
  'vehicle',
] as const

export type Detection = {
  classId: number
  className: string
  confidence: number
  bbox: [number, number, number, number]
}

export type LetterboxInfo = {
  originalWidth: number
  originalHeight: number
  scale: number
  padX: number
  padY: number
}

export type RawDetectionOutput = {
  data: Float32Array
  dimensions: number[]
  letterbox: LetterboxInfo
}

function getSourceDimensions(
  source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
) {
  if (source instanceof HTMLImageElement) {
    return {
      width: source.naturalWidth,
      height: source.naturalHeight,
    }
  }

  if (source instanceof HTMLVideoElement) {
    return {
      width: source.videoWidth,
      height: source.videoHeight,
    }
  }

  return {
    width: source.width,
    height: source.height,
  }
}

function preprocessImage(
  source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
) {
  const { width, height } = getSourceDimensions(source)

  if (!width || !height) {
    throw new Error('Invalid image dimensions.')
  }

  const scale = Math.min(
    MODEL_SIZE / width,
    MODEL_SIZE / height
  )

  const resizedWidth = Math.round(width * scale)
  const resizedHeight = Math.round(height * scale)

  const padX = (MODEL_SIZE - resizedWidth) / 2
  const padY = (MODEL_SIZE - resizedHeight) / 2

  const canvas = document.createElement('canvas')
  canvas.width = MODEL_SIZE
  canvas.height = MODEL_SIZE

  const context = canvas.getContext('2d')

  if (!context) {
    throw new Error('Could not create image canvas.')
  }

  context.fillStyle = `rgb(${LETTERBOX_COLOR}, ${LETTERBOX_COLOR}, ${LETTERBOX_COLOR})`
  context.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE)

  context.drawImage(
    source,
    0,
    0,
    width,
    height,
    padX,
    padY,
    resizedWidth,
    resizedHeight
  )

  const imageData = context.getImageData(
    0,
    0,
    MODEL_SIZE,
    MODEL_SIZE
  )

  const floatData = new Float32Array(
    3 * MODEL_SIZE * MODEL_SIZE
  )

  const planeSize = MODEL_SIZE * MODEL_SIZE

  for (let i = 0; i < planeSize; i++) {
    const pixelIndex = i * 4

    floatData[i] = imageData.data[pixelIndex] / 255
    floatData[planeSize + i] =
      imageData.data[pixelIndex + 1] / 255
    floatData[2 * planeSize + i] =
      imageData.data[pixelIndex + 2] / 255
  }

  return {
    tensor: new ort.Tensor(
      'float32',
      floatData,
      [1, 3, MODEL_SIZE, MODEL_SIZE]
    ),
    letterbox: {
      originalWidth: width,
      originalHeight: height,
      scale,
      padX,
      padY,
    },
  }
}

function calculateIoU(
  a: Detection['bbox'],
  b: Detection['bbox']
): number {
  const x1 = Math.max(a[0], b[0])
  const y1 = Math.max(a[1], b[1])
  const x2 = Math.min(a[2], b[2])
  const y2 = Math.min(a[3], b[3])

  const intersection =
    Math.max(0, x2 - x1) * Math.max(0, y2 - y1)

  const areaA =
    Math.max(0, a[2] - a[0]) *
    Math.max(0, a[3] - a[1])

  const areaB =
    Math.max(0, b[2] - b[0]) *
    Math.max(0, b[3] - b[1])

  const union = areaA + areaB - intersection

  return union > 0 ? intersection / union : 0
}

function applyNMS(detections: Detection[]): Detection[] {
  const results: Detection[] = []

  const classGroups = new Map<number, Detection[]>()

  for (const detection of detections) {
    const group = classGroups.get(detection.classId) || []
    group.push(detection)
    classGroups.set(detection.classId, group)
  }

  for (const group of classGroups.values()) {
    group.sort((a, b) => b.confidence - a.confidence)

    while (group.length > 0) {
      const best = group.shift()!
      results.push(best)

      for (let i = group.length - 1; i >= 0; i--) {
        if (
          calculateIoU(best.bbox, group[i].bbox) >
          NMS_THRESHOLD
        ) {
          group.splice(i, 1)
        }
      }
    }
  }

  return results.sort(
    (a, b) => b.confidence - a.confidence
  )
}

export function decodeDetections(
  output: RawDetectionOutput
): Detection[] {
  const { data, dimensions, letterbox } = output

  if (
    dimensions.length !== 3 ||
    dimensions[0] !== 1 ||
    dimensions[1] !== NUM_CLASSES + 4 ||
    dimensions[2] !== NUM_CANDIDATES
  ) {
    throw new Error(
      `Unexpected output shape: [${dimensions.join(', ')}]`
    )
  }

  const detections: Detection[] = []

  const getValue = (row: number, candidate: number) =>
    data[row * NUM_CANDIDATES + candidate]

  for (let i = 0; i < NUM_CANDIDATES; i++) {
    let bestClass = -1
    let bestConfidence = 0

    for (let classId = 0; classId < NUM_CLASSES; classId++) {
      const confidence = getValue(classId + 4, i)

      if (confidence > bestConfidence) {
        bestConfidence = confidence
        bestClass = classId
      }
    }

    if (
      bestClass < 0 ||
      bestConfidence < CONFIDENCE_THRESHOLD
    ) {
      continue
    }

    const centerX = getValue(0, i)
    const centerY = getValue(1, i)
    const width = getValue(2, i)
    const height = getValue(3, i)

    const x1 = (centerX - width / 2 - letterbox.padX) /
      letterbox.scale
    const y1 = (centerY - height / 2 - letterbox.padY) /
      letterbox.scale
    const x2 = (centerX + width / 2 - letterbox.padX) /
      letterbox.scale
    const y2 = (centerY + height / 2 - letterbox.padY) /
      letterbox.scale

    const bbox: Detection['bbox'] = [
      Math.max(0, Math.min(letterbox.originalWidth, x1)),
      Math.max(0, Math.min(letterbox.originalHeight, y1)),
      Math.max(0, Math.min(letterbox.originalWidth, x2)),
      Math.max(0, Math.min(letterbox.originalHeight, y2)),
    ]

    if (bbox[2] <= bbox[0] || bbox[3] <= bbox[1]) {
      continue
    }

    detections.push({
      classId: bestClass,
      className: CLASS_NAMES[bestClass],
      confidence: bestConfidence,
      bbox,
    })
  }

  return applyNMS(detections)
}

export async function runRawInference(
  source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
): Promise<RawDetectionOutput> {
  const session = await loadModel()
  const { tensor, letterbox } = preprocessImage(source)

  const results = await session.run({
    images: tensor,
  })

  const output = results.output0

  if (!output) {
    throw new Error('Model output "output0" was not found.')
  }

  if (!(output.data instanceof Float32Array)) {
    throw new Error('Unexpected model output data type.')
  }

  return {
    data: output.data,
    dimensions: [...output.dims],
    letterbox,
  }
}

export async function runPpeInference(
  source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
) {
  const rawOutput = await runRawInference(source)
  const detections = decodeDetections(rawOutput)

  return {
    detections,
    imageWidth: rawOutput.letterbox.originalWidth,
    imageHeight: rawOutput.letterbox.originalHeight,
  }
}
