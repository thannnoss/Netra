import * as ort from 'onnxruntime-web'

let session: ort.InferenceSession | null = null
let sessionPromise: Promise<ort.InferenceSession> | null = null

export async function loadModel(): Promise<ort.InferenceSession> {
  if (session) return session

  if (!sessionPromise) {
    sessionPromise = (async () => {
      const response = await fetch('/models/best.onnx')

      if (!response.ok) {
        throw new Error('Could not load best.onnx')
      }

      const modelBuffer = await response.arrayBuffer()

      const loadedSession = await ort.InferenceSession.create(
        modelBuffer,
        {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        }
      )

      session = loadedSession

console.log('NETRA model loaded successfully')
console.log('Model inputs:', loadedSession.inputNames)
console.log('Model outputs:', loadedSession.outputNames)

return loadedSession
    })().catch((error) => {
      sessionPromise = null
      throw error
    })
  }

  return sessionPromise
}