import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import jsPDF from 'jspdf'
import {
  ShieldCheck,
  UploadCloud,
  Users,
  CheckCircle2,
  AlertTriangle,
  HardHat,
  Shirt,
  Glasses,
  Hand,
  Footprints,
  ScanSearch,
  LoaderCircle,
  X,
  Image as ImageIcon,
  Camera,
  CameraOff,
  History,
  FileDown,
  Trash2,
} from 'lucide-react'
import './App.css'
import { runPpeInference } from './ai/ppeDetector'

type Worker = {
  id: number
  status: 'COMPLIANT' | 'VIOLATION' | 'REVIEW'
  ppe: {
    helmet: boolean
    vest: boolean
    glasses: boolean
    gloves: boolean
    shoes: boolean
  }
  evidence?: {
    helmet: 'DETECTED' | 'NOT_DETECTED' | 'EXPLICITLY_MISSING'
    vest: 'DETECTED' | 'NOT_DETECTED' | 'EXPLICITLY_MISSING'
    glasses: 'DETECTED' | 'NOT_DETECTED' | 'EXPLICITLY_MISSING'
    gloves: 'DETECTED' | 'NOT_DETECTED' | 'EXPLICITLY_MISSING'
    shoes: 'DETECTED' | 'NOT_DETECTED' | 'EXPLICITLY_MISSING'
  }
  confidence?: number
  bbox?: [number, number, number, number]
}

type ViolationSeverity = 'CRITICAL' | 'HIGH' | 'MEDIUM'
type ViolationFilter = 'ALL' | ViolationSeverity

type Violation = {
  workerId: number
  missingItems: string[]
  severity: ViolationSeverity
  timestamp: string
}

type PpeRequirements = {
  helmet: boolean
  vest: boolean
  glasses: boolean
  gloves: boolean
  shoes: boolean
}

type SavedInspection = {
  id: string
  timestamp: string
  inputMode: 'image' | 'camera' | 'video'
  sourceName: string
  workers: Worker[]
  requirements: PpeRequirements
  modelVersion?: string
  evidenceImages?: string[]
  cameraSession?: {
    startedAt: string
    endedAt: string
    durationSeconds: number
    scanCount: number
    detectionCounts: Record<string, number>
  }
}

const DB_NAME = 'netra-inspection-db'
const DB_VERSION = 3
const MODEL_VERSION = 'ppe-yolov8n-17class-browser-v1'
const STORE_NAME = 'inspections'

const openInspectionDatabase = (): Promise<IDBDatabase> =>
  new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    request.onupgradeneeded = (event) => {
      const db = request.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' })
      } else if ((event as IDBVersionChangeEvent).oldVersion < 3) {
        // Remove records created by the previous inspection pipeline.
        request.transaction?.objectStore(STORE_NAME).clear()
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error || new Error('Could not open inspection database.'))
  })

const getSavedInspections = async (): Promise<SavedInspection[]> => {
  const db = await openInspectionDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const request = tx.objectStore(STORE_NAME).getAll()
    request.onsuccess = () => {
      db.close()
      resolve((request.result as SavedInspection[]).filter((item) => item.modelVersion === MODEL_VERSION).sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)))
    }
    request.onerror = () => { db.close(); reject(request.error || new Error('Could not load history.')) }
  })
}

const deleteInspectionFromDatabase = async (id: string): Promise<void> => {
  const db = await openInspectionDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.objectStore(STORE_NAME).delete(id)
    tx.oncomplete = () => { db.close(); resolve() }
    tx.onerror = () => { db.close(); reject(tx.error || new Error('Could not delete inspection.')) }
  })
}


const saveInspectionToDatabase = async (record: SavedInspection): Promise<void> => {
  const db = await openInspectionDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.objectStore(STORE_NAME).put(record)
    tx.oncomplete = () => { db.close(); resolve() }
    tx.onerror = () => { db.close(); reject(tx.error || new Error('Could not save inspection.')) }
  })
}

type PpeKey = keyof PpeRequirements

const POSITIVE_CLASS_TO_PPE: Record<string, PpeKey> = {
  helmet: 'helmet',
  vest: 'vest',
  goggles: 'glasses',
  gloves: 'gloves',
  boots: 'shoes',
}

const NEGATIVE_CLASS_TO_PPE: Record<string, PpeKey> = {
  no_helmet: 'helmet',
  'NO-Safety Vest': 'vest',
  no_goggle: 'glasses',
  no_gloves: 'gloves',
  no_boots: 'shoes',
}

const associateDetectionsWithWorkers = (
  detections: Awaited<ReturnType<typeof runPpeInference>>['detections'],
  requirements: PpeRequirements
): Worker[] => {
  const people = detections.filter((item) => item.className === 'Person')
  type EvidenceStatus = 'DETECTED' | 'NOT_DETECTED' | 'EXPLICITLY_MISSING'
  const evidenceByPerson: Record<PpeKey, EvidenceStatus>[] = people.map(() => ({
    helmet: 'NOT_DETECTED',
    vest: 'NOT_DETECTED',
    glasses: 'NOT_DETECTED',
    gloves: 'NOT_DETECTED',
    shoes: 'NOT_DETECTED',
  }))

  const assignToPerson = (bbox: [number, number, number, number]) => {
    const cx = (bbox[0] + bbox[2]) / 2
    const cy = (bbox[1] + bbox[3]) / 2
    let bestIndex = -1
    let bestScore = Number.POSITIVE_INFINITY

    people.forEach((person, index) => {
      const [x1, y1, x2, y2] = person.bbox
      const width = Math.max(1, x2 - x1)
      const height = Math.max(1, y2 - y1)
      const expanded = {
        left: x1 - width * 0.08,
        right: x2 + width * 0.08,
        top: y1 - height * 0.05,
        bottom: y2 + height * 0.05,
      }

      if (
        cx < expanded.left || cx > expanded.right ||
        cy < expanded.top || cy > expanded.bottom
      ) return

      const dx = (cx - (x1 + x2) / 2) / width
      const dy = (cy - (y1 + y2) / 2) / height
      const score = dx * dx + dy * dy
      if (score < bestScore) {
        bestScore = score
        bestIndex = index
      }
    })

    return bestIndex
  }

  for (const detection of detections) {
    const key = POSITIVE_CLASS_TO_PPE[detection.className]
    const negativeKey = NEGATIVE_CLASS_TO_PPE[detection.className]
    if (!key && !negativeKey) continue

    const personIndex = assignToPerson(detection.bbox)
    if (personIndex < 0) continue

    const ppeKey = key || negativeKey
    const nextEvidence = key ? 'DETECTED' : 'EXPLICITLY_MISSING'
    const currentEvidence = evidenceByPerson[personIndex][ppeKey]

    // Explicit negative evidence takes precedence over a conflicting positive box.
    if (nextEvidence === 'EXPLICITLY_MISSING' || currentEvidence === 'NOT_DETECTED') {
      evidenceByPerson[personIndex][ppeKey] = nextEvidence
    }
  }

  return people.map((person, index) => {
    const evidence = evidenceByPerson[index]
    const ppe = {
      helmet: evidence.helmet === 'DETECTED',
      vest: evidence.vest === 'DETECTED',
      glasses: evidence.glasses === 'DETECTED',
      gloves: evidence.gloves === 'DETECTED',
      shoes: evidence.shoes === 'DETECTED',
    }

    const requiredKeys = (Object.keys(requirements) as PpeKey[])
      .filter((key) => requirements[key])
    const hasExplicitViolation = requiredKeys.some(
      (key) => evidence[key] === 'EXPLICITLY_MISSING'
    )
    const hasUnknown = requiredKeys.some(
      (key) => evidence[key] === 'NOT_DETECTED'
    )

    const status: Worker['status'] = hasExplicitViolation
      ? 'VIOLATION'
      : hasUnknown
        ? 'REVIEW'
        : 'COMPLIANT'

    return {
      id: index + 1,
      status,
      ppe,
      evidence,
      confidence: person.confidence,
      bbox: person.bbox,
    }
  })
}




const activeSessionEvidenceSlot = (completedScans: number) => completedScans < 3 ? completedScans : 3

function App() {
  const mountRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const videoFileRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const liveOverlayRef = useRef<HTMLCanvasElement>(null)
  const transitionTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const liveScanTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const liveScanningRef = useRef(false)
  const liveSessionRef = useRef<{ startedAt: string; startedMs: number; scanCount: number; detectionCounts: Record<string, number>; latestWorkers: Worker[]; evidenceImages: string[] } | null>(null)

  const [launching, setLaunching] = useState(false)
  const [dashboard, setDashboard] = useState(false)

  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null)
  const [workers, setWorkers] = useState<Worker[]>([])
  const [expandedWorkers, setExpandedWorkers] = useState<number[]>([])
  const [inspectionTimestamp, setInspectionTimestamp] = useState<string | null>(null)
  const [violationFilter, setViolationFilter] = useState<ViolationFilter>('ALL')
  const [postInspectionView, setPostInspectionView] = useState<
    'violations' | 'analytics' | 'history' | null
  >(null)
  const [savedInspections, setSavedInspections] = useState<SavedInspection[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null)

  const [helmetRequired, setHelmetRequired] = useState(true)
  const [vestRequired, setVestRequired] = useState(true)
  const [glassesRequired, setGlassesRequired] = useState(true)
  const [glovesRequired, setGlovesRequired] = useState(true)
  const [shoesRequired, setShoesRequired] = useState(true)

 const [analyzing, setAnalyzing] = useState(false)
  const [message, setMessage] = useState('')
  const [inputMode, setInputMode] = useState<'image' | 'camera' | 'video'>('image')
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null)
  const [cameraError, setCameraError] = useState('')
  const [liveScanning, setLiveScanning] = useState(false)
  const [liveScanCount, setLiveScanCount] = useState(0)
  const [videoProgress, setVideoProgress] = useState(0)

  // ORIGINAL THREE.JS WAREHOUSE HALLWAY
  useEffect(() => {
    const container = mountRef.current
    if (!container) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x172326)
    scene.fog = new THREE.FogExp2(0x172326, 0.009)

    const camera = new THREE.PerspectiveCamera(
      58,
      window.innerWidth / window.innerHeight,
      0.1,
      180
    )

    camera.position.set(0, 3.2, 12)
    camera.lookAt(0, 2, -20)

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    })

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.35))
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.25
    renderer.outputColorSpace = THREE.SRGBColorSpace

    container.appendChild(renderer.domElement)

    const floorMaterial = new THREE.MeshStandardMaterial({
      color: 0x303b3c,
      roughness: 0.65,
      metalness: 0.25,
    })

    const concreteMaterial = new THREE.MeshStandardMaterial({
      color: 0x394345,
      roughness: 0.85,
      metalness: 0.1,
    })

    const steelMaterial = new THREE.MeshStandardMaterial({
      color: 0x46585a,
      roughness: 0.5,
      metalness: 0.65,
    })

    const darkSteelMaterial = new THREE.MeshStandardMaterial({
      color: 0x202d2f,
      roughness: 0.6,
      metalness: 0.65,
    })

    const rackMaterial = new THREE.MeshStandardMaterial({
      color: 0x506063,
      roughness: 0.55,
      metalness: 0.6,
    })

    const crateMaterial = new THREE.MeshStandardMaterial({
      color: 0x455656,
      roughness: 0.85,
      metalness: 0.1,
    })

    const cyanMaterial = new THREE.MeshStandardMaterial({
      color: 0x39cfc2,
      emissive: 0x0b756c,
      emissiveIntensity: 1.2,
    })

    const warmMaterial = new THREE.MeshStandardMaterial({
      color: 0xffe0ae,
      emissive: 0xffbd70,
      emissiveIntensity: 2.2,
    })

    scene.add(new THREE.HemisphereLight(0xd1e1dc, 0x303a3b, 1.8))

    const mainLight = new THREE.DirectionalLight(0xe0e8dc, 1.8)
    mainLight.position.set(-8, 14, 8)
    mainLight.castShadow = true
    mainLight.shadow.mapSize.set(1024, 1024)
    scene.add(mainLight)

    const backGlow = new THREE.PointLight(0x168c89, 35, 65, 2)
    backGlow.position.set(0, 5, -65)
    scene.add(backGlow)

    const sideGlow = new THREE.PointLight(0x557f83, 24, 45, 2)
    sideGlow.position.set(-8, 4, -24)
    scene.add(sideGlow)

    const addBox = (
      width: number,
      height: number,
      depth: number,
      x: number,
      y: number,
      z: number,
      material: THREE.Material,
      castShadow = true,
      receiveShadow = true
    ) => {
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(width, height, depth),
        material
      )

      mesh.position.set(x, y, z)
      mesh.castShadow = castShadow
      mesh.receiveShadow = receiveShadow
      scene.add(mesh)

      return mesh
    }

    // FLOOR
    addBox(24, 0.5, 145, 0, -0.3, -48, floorMaterial, false, true)

    const floorLines = new THREE.Group()

    const lineMaterial = new THREE.MeshBasicMaterial({
      color: 0x8caeaa,
      transparent: true,
      opacity: 0.28,
    })

    for (let z = 8; z > -115; z -= 8) {
      const line = new THREE.Mesh(
        new THREE.BoxGeometry(0.035, 0.012, 1.5),
        lineMaterial
      )

      line.position.set(0, -0.035, z)
      floorLines.add(line)
    }

    for (const x of [-5.5, 5.5]) {
      const lane = new THREE.Mesh(
        new THREE.BoxGeometry(0.045, 0.015, 135),
        lineMaterial
      )

      lane.position.set(x, -0.03, -48)
      floorLines.add(lane)
    }

    scene.add(floorLines)

    // WAREHOUSE WALLS AND ROOF
    addBox(0.5, 11, 145, -12, 5.2, -48, concreteMaterial, false)
    addBox(0.5, 11, 145, 12, 5.2, -48, concreteMaterial, false)
    addBox(24, 0.5, 145, 0, 11, -48, darkSteelMaterial, false)

    // CEILING
    for (let z = 8; z > -120; z -= 7) {
      addBox(24, 0.24, 0.3, 0, 10.35, z, steelMaterial)
      addBox(0.22, 1.2, 0.22, -9.8, 9.7, z, steelMaterial)
      addBox(0.22, 1.2, 0.22, 9.8, 9.7, z, steelMaterial)
    }

    // PILLARS
    for (let z = 7; z > -112; z -= 14) {
      for (const x of [-9.7, 9.7]) {
        addBox(0.8, 10.1, 0.8, x, 5, z, steelMaterial)
        addBox(1.1, 0.35, 1.1, x, 0.18, z, darkSteelMaterial)
        addBox(1.1, 0.35, 1.1, x, 9.8, z, darkSteelMaterial)
        addBox(0.84, 0.08, 0.84, x, 7.5, z, cyanMaterial, false)
      }
    }

    // STORAGE RACKS
    for (const side of [-1, 1]) {
      for (let z = 2; z > -105; z -= 13) {
        const x = side * 8

        for (const offset of [-1.3, 1.3]) {
          addBox(0.2, 6.5, 0.2, x + offset, 3.25, z, rackMaterial)
        }

        for (const y of [1.2, 3.3, 5.4]) {
          addBox(3, 0.18, 4.4, x, y, z - 1.4, darkSteelMaterial)
          addBox(2.8, 0.1, 0.12, x, y + 0.16, z + 0.75, rackMaterial)
        }

        for (let level = 0; level < 3; level++) {
          const crate = addBox(
            1.45,
            1.25,
            2.5,
            x,
            1.9 + level * 2.05,
            z - 1.4,
            crateMaterial
          )

          crate.rotation.y = side * 0.025

          addBox(
            1.1,
            0.035,
            0.035,
            x,
            1.9 + level * 2.05,
            z - 0.1,
            steelMaterial,
            false
          )
        }
      }
    }

    // CONVEYORS
    for (const z of [-18, -35, -53, -72, -90]) {
      addBox(3, 0.8, 5.5, 0, 0.5, z, darkSteelMaterial)
      addBox(2.8, 0.12, 5.2, 0, 0.98, z, steelMaterial, false)

      for (const x of [-1.4, 1.4]) {
        addBox(0.1, 0.35, 5.4, x, 1.15, z, rackMaterial)
      }

      const status = new THREE.Mesh(
        new THREE.BoxGeometry(0.12, 0.12, 0.12),
        cyanMaterial
      )

      status.position.set(1.48, 1.3, z - 2)
      scene.add(status)
    }

    // LIGHTS
    const beams: THREE.Mesh[] = []
    const lamps: THREE.Mesh[] = []

    for (let z = 5; z > -105; z -= 15) {
      for (const x of [-5.5, 5.5]) {
        const lamp = new THREE.Mesh(
          new THREE.BoxGeometry(1.5, 0.12, 0.7),
          warmMaterial
        )

        lamp.position.set(x, 10.45, z)
        scene.add(lamp)
        lamps.push(lamp)

        const point = new THREE.PointLight(0xffd49a, 25, 20, 2)
        point.position.set(x, 9.8, z)
        scene.add(point)

        const beam = new THREE.Mesh(
          new THREE.CylinderGeometry(0.15, 2.8, 9, 24, 1, true),
          new THREE.MeshBasicMaterial({
            color: 0xffd6a0,
            transparent: true,
            opacity: 0.045,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            side: THREE.DoubleSide,
          })
        )

        beam.position.set(x, 5.9, z)
        scene.add(beam)
        beams.push(beam)
      }
    }

    // DISTANT LIGHTS
    for (let z = -10; z > -110; z -= 18) {
      const light = new THREE.PointLight(0x36bcb2, 12, 16, 2)
      light.position.set(0, 5, z)
      scene.add(light)
    }

    // DUST
    const particleCount = 650
    const particlePositions = new Float32Array(particleCount * 3)

    for (let i = 0; i < particleCount; i++) {
      particlePositions[i * 3] = (Math.random() - 0.5) * 22
      particlePositions[i * 3 + 1] = Math.random() * 10
      particlePositions[i * 3 + 2] = -Math.random() * 120 + 12
    }

    const dustGeometry = new THREE.BufferGeometry()

    dustGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(particlePositions, 3)
    )

    const dustMaterial = new THREE.PointsMaterial({
      color: 0xd0e9df,
      size: 0.045,
      transparent: true,
      opacity: 0.48,
      sizeAttenuation: true,
      depthWrite: false,
    })

    const dust = new THREE.Points(dustGeometry, dustMaterial)
    scene.add(dust)

    // ATMOSPHERE
    const atmosphere = new THREE.Mesh(
      new THREE.PlaneGeometry(24, 11),
      new THREE.MeshBasicMaterial({
        color: 0x547a76,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
      })
    )

    atmosphere.position.set(0, 5, -60)
    scene.add(atmosphere)

    // ANIMATION
    const clock = new THREE.Clock()
    let animationId = 0
    let elapsed = 0
    let disposed = false

    const animate = () => {
      if (disposed) return

      animationId = requestAnimationFrame(animate)

      const delta = Math.min(clock.getDelta(), 0.033)
      elapsed += delta

      camera.position.z = 12 - elapsed * 0.72
      camera.position.x = Math.sin(elapsed * 0.14) * 0.14
      camera.position.y = 3.2 + Math.sin(elapsed * 0.22) * 0.035

      camera.lookAt(
        Math.sin(elapsed * 0.14) * 0.12,
        2.1 + Math.sin(elapsed * 0.2) * 0.025,
        camera.position.z - 27
      )

      lamps.forEach((lamp, index) => {
        const material = lamp.material as THREE.MeshStandardMaterial
        material.emissiveIntensity =
          1.8 + Math.sin(elapsed * 1.5 + index) * 0.3
      })

      beams.forEach((beam, index) => {
        beam.rotation.z = Math.sin(elapsed * 0.25 + index) * 0.025

        const material = beam.material as THREE.MeshBasicMaterial
        material.opacity =
          0.035 + (Math.sin(elapsed * 0.7 + index) + 1) * 0.012
      })

      const positions = dust.geometry.attributes.position.array as Float32Array

      for (let i = 0; i < particleCount; i++) {
        const index = i * 3

        positions[index + 1] += Math.sin(elapsed + i) * delta * 0.018
        positions[index + 2] += delta * 0.35

        if (positions[index + 2] > 13) {
          positions[index + 2] = -110
          positions[index] = (Math.random() - 0.5) * 22
          positions[index + 1] = Math.random() * 10
        }
      }

      dust.geometry.attributes.position.needsUpdate = true
      renderer.render(scene, camera)
    }

    animate()

    // RESIZE
    const resize = () => {
      const width = window.innerWidth
      const height = window.innerHeight

      camera.aspect = width / height
      camera.updateProjectionMatrix()

      renderer.setSize(width, height)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.35))
    }

    window.addEventListener('resize', resize)

    // CLEANUP
    return () => {
      disposed = true
      cancelAnimationFrame(animationId)
      window.removeEventListener('resize', resize)

      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Points) {
          object.geometry.dispose()

          const materials = Array.isArray(object.material)
            ? object.material
            : [object.material]

          materials.forEach((material) => material.dispose())
        }
      })

      renderer.dispose()

      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  // LIVE CAMERA PREVIEW
  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream
      void videoRef.current.play().catch(() => {
        setCameraError(
          'Unable to start the camera preview. Please check browser permissions.'
        )
      })
    }
  }, [cameraStream])

  const stopCamera = () => {
    if (liveScanningRef.current) void stopLiveInspection()
    cameraStream?.getTracks().forEach((track) => track.stop())
    setCameraStream(null)
  }

  const startCamera = async () => {
    setCameraError('')

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        'Camera access is unavailable. Use HTTPS or localhost in a supported browser.'
      )
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      })
      setCameraStream(stream)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'NotAllowedError') {
        setCameraError(
          'Camera permission was denied. Allow camera access in your browser settings and try again.'
        )
      } else if (
        error instanceof DOMException &&
        error.name === 'NotFoundError'
      ) {
        setCameraError('No camera was found on this device.')
      } else {
        setCameraError(
          'Could not access the camera. Check that it is connected and not being used by another app.'
        )
      }
    }
  }

  useEffect(() => {
    return () => {
      liveScanningRef.current = false
      if (liveScanTimer.current) clearTimeout(liveScanTimer.current)
      cameraStream?.getTracks().forEach((track) => track.stop())
    }
  }, [cameraStream])

  const refreshInspectionHistory = async () => {
    setHistoryLoading(true)
    setHistoryError('')
    try {
      setSavedInspections(await getSavedInspections())
    } catch {
      setHistoryError('Could not load local inspection history. Check browser storage permissions.')
    } finally {
      setHistoryLoading(false)
    }
  }

  useEffect(() => {
    void refreshInspectionHistory()
  }, [])

  // INTRO TRANSITION
  const initiateWork = () => {
    if (launching) return

    setLaunching(true)

    transitionTimer.current = setTimeout(() => {
      setDashboard(true)
    }, 450)
  }

  useEffect(() => {
    return () => {
      if (transitionTimer.current) {
        clearTimeout(transitionTimer.current)
      }
    }
  }, [])

  // FILE HANDLING
  const handleFile = (file?: File) => {
    if (!file) return

    stopCamera()
    setInputMode('image')

    if (!file.type.startsWith('image/')) {
      setMessage('Please select a valid image file.')
      return
    }

    setSelectedFile(file)
    setAnnotatedImage(null)
    setWorkers([])
    setExpandedWorkers([])
    setInspectionTimestamp(null)
    setViolationFilter('ALL')
    setPostInspectionView(null)
    setSelectedHistoryId(null)
    setMessage('')

    setPreview((oldPreview) => {
      if (oldPreview?.startsWith('blob:')) {
        URL.revokeObjectURL(oldPreview)
      }

      return URL.createObjectURL(file)
    })
  }

  const handleVideoFile = (file?: File) => {
    if (!file) return
    stopCamera()
    if (!file.type.startsWith('video/')) {
      setMessage('Please select a valid video file.')
      return
    }
    if (file.size > 50 * 1024 * 1024) {
      setMessage('Video exceeds the 50 MB limit. Choose a smaller file.')
      return
    }
    const objectUrl = URL.createObjectURL(file)
    const probe = document.createElement('video')
    probe.preload = 'metadata'
    probe.onloadedmetadata = () => {
      if (!Number.isFinite(probe.duration) || probe.duration < 10 || probe.duration > 30) {
        setMessage(`Video duration must be between 10 and 30 seconds. Selected: ${Number.isFinite(probe.duration) ? `${probe.duration.toFixed(1)}s` : 'unknown'}.`)
        URL.revokeObjectURL(objectUrl)
        return
      }
      setSelectedFile(file)
      setPreview((oldPreview) => { if (oldPreview?.startsWith('blob:')) URL.revokeObjectURL(oldPreview); return objectUrl })
      setInputMode('video')
      setWorkers([])
      setExpandedWorkers([])
      setInspectionTimestamp(null)
      setViolationFilter('ALL')
      setPostInspectionView(null)
      setSelectedHistoryId(null)
      setVideoProgress(0)
      setMessage(`Video ready · ${(file.size / (1024 * 1024)).toFixed(1)} MB · ${probe.duration.toFixed(1)} seconds.`)
    }
    probe.onerror = () => {
      URL.revokeObjectURL(probe.src)
      URL.revokeObjectURL(objectUrl)
      setMessage('Could not read this video. Try MP4 or WebM.')
    }
    probe.src = objectUrl
  }

  const resetFile = () => {
    setSelectedFile(null)
    setAnnotatedImage(null)
    setWorkers([])
    setExpandedWorkers([])
    setInspectionTimestamp(null)
    setViolationFilter('ALL')
    setPostInspectionView(null)
    setSelectedHistoryId(null)
    setMessage('')

    setPreview((oldPreview) => {
      if (oldPreview?.startsWith('blob:')) {
        URL.revokeObjectURL(oldPreview)
      }

      return null
    })

    if (fileRef.current) fileRef.current.value = ''
    if (videoFileRef.current) videoFileRef.current.value = ''
    setVideoProgress(0)
  }

  const clearLiveOverlay = () => {
    const canvas = liveOverlayRef.current
    const context = canvas?.getContext('2d')
    if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height)
  }

  const drawLiveOverlay = (
    video: HTMLVideoElement,
    detections: Awaited<ReturnType<typeof runPpeInference>>['detections']
  ) => {
    const canvas = liveOverlayRef.current
    const context = canvas?.getContext('2d')
    if (!canvas || !context || !video.videoWidth || !video.videoHeight) return

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
    }
    context.clearRect(0, 0, canvas.width, canvas.height)

    const scale = Math.max(1, canvas.width / 900)
    const fontSize = Math.max(12, Math.round(14 * scale))
    context.lineWidth = Math.max(2, 2 * scale)
    context.font = `600 ${fontSize}px Inter, Arial, sans-serif`
    context.textBaseline = 'top'

    for (const detection of detections) {
      const [x1, y1, x2, y2] = detection.bbox
      const missing = detection.className.startsWith('NO-') || detection.className.startsWith('no_')
      const color = detection.className === 'Person' ? '#38bdf8' : missing ? '#fb5252' : '#34d399'
      const label = `${detection.className} ${(detection.confidence * 100).toFixed(0)}%`
      const labelWidth = context.measureText(label).width + 14 * scale
      const labelHeight = fontSize + 8 * scale
      const labelY = Math.max(0, y1 - labelHeight)

      context.strokeStyle = color
      context.strokeRect(x1, y1, x2 - x1, y2 - y1)
      context.fillStyle = color
      context.fillRect(x1, labelY, Math.min(labelWidth, canvas.width - x1), labelHeight)
      context.fillStyle = '#07110e'
      context.fillText(label, x1 + 7 * scale, labelY + 4 * scale)
    }
  }

  // Continuous camera inspection: sample frames sequentially to avoid overlapping ONNX runs.
  const startLiveInspection = () => {
    if (!cameraStream || !videoRef.current) {
      setMessage('Start the camera and wait for the preview before scanning.')
      return
    }
    if (liveScanningRef.current) return

    const startedAt = new Date().toISOString()
    liveSessionRef.current = {
      startedAt,
      startedMs: Date.now(),
      scanCount: 0,
      detectionCounts: {},
      latestWorkers: [],
      evidenceImages: [],
    }
    clearLiveOverlay()
    liveScanningRef.current = true
    setLiveScanning(true)
    setLiveScanCount(0)
    setWorkers([])
    setInspectionTimestamp(null)
    setSelectedHistoryId(null)
    setPostInspectionView(null)
    setMessage('Live inspection running. NETRA is scanning camera frames. Select STOP & GENERATE REPORT when finished.')

    const scanFrame = async () => {
      if (!liveScanningRef.current) return
      const video = videoRef.current
      const session = liveSessionRef.current
      if (!video || video.readyState < 2 || !session) {
        setMessage('Waiting for a ready camera frame...')
        liveScanTimer.current = setTimeout(() => void scanFrame(), 500)
        return
      }

      try {
        const result = await runPpeInference(video)
        if (!liveScanningRef.current || !liveSessionRef.current) return
        drawLiveOverlay(video, result.detections)
        // Save a small set of annotated frames for the final PDF (first 3 + latest).
        const snapshotCanvas = document.createElement('canvas')
        const snapshotScale = Math.min(1, 1280 / video.videoWidth)
        snapshotCanvas.width = Math.max(1, Math.round(video.videoWidth * snapshotScale))
        snapshotCanvas.height = Math.max(1, Math.round(video.videoHeight * snapshotScale))
        const snapshotContext = snapshotCanvas.getContext('2d')
        if (snapshotContext) {
          snapshotContext.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height)
          snapshotContext.save()
          snapshotContext.scale(snapshotScale, snapshotScale)
          const stroke = Math.max(2, video.videoWidth / 500)
          const font = Math.max(13, Math.round(video.videoWidth / 65))
          snapshotContext.lineWidth = stroke
          snapshotContext.font = `600 ${font}px Arial, sans-serif`
          snapshotContext.textBaseline = 'top'
          for (const detection of result.detections) {
            const [x1, y1, x2, y2] = detection.bbox
            const missing = detection.className.startsWith('NO-') || detection.className.startsWith('no_')
            const color = detection.className === 'Person' ? '#38bdf8' : missing ? '#ef4444' : '#34d399'
            const label = `${detection.className} ${(detection.confidence * 100).toFixed(1)}%`
            const labelHeight = font + 8
            const labelY = Math.max(0, y1 - labelHeight)
            snapshotContext.strokeStyle = color
            snapshotContext.strokeRect(x1, y1, x2 - x1, y2 - y1)
            snapshotContext.fillStyle = color
            snapshotContext.fillRect(x1, labelY, Math.min(snapshotContext.measureText(label).width + 12, video.videoWidth - x1), labelHeight)
            snapshotContext.fillStyle = '#07110e'
            snapshotContext.fillText(label, x1 + 6, labelY + 4)
          }
          snapshotContext.restore()
          const snapshot = snapshotCanvas.toDataURL('image/jpeg', 0.76)
          if (activeSessionEvidenceSlot(session.scanCount) < 3) session.evidenceImages.push(snapshot)
          else {
            if (session.evidenceImages.length < 4) session.evidenceImages.push(snapshot)
            else session.evidenceImages[3] = snapshot
          }
        }
        const requirements: PpeRequirements = {
          helmet: helmetRequired,
          vest: vestRequired,
          glasses: glassesRequired,
          gloves: glovesRequired,
          shoes: shoesRequired,
        }
        const frameWorkers = associateDetectionsWithWorkers(result.detections, requirements)
        const activeSession = liveSessionRef.current
        activeSession.scanCount += 1
        for (const detection of result.detections) {
          activeSession.detectionCounts[detection.className] =
            (activeSession.detectionCounts[detection.className] || 0) + 1
        }
        // Keep the most recent frame's worker snapshot. This is not persistent person tracking.
        activeSession.latestWorkers = frameWorkers
        setWorkers(frameWorkers)
        setLiveScanCount(activeSession.scanCount)
        setMessage(`LIVE SCANNING · ${activeSession.scanCount} frames analyzed · ${frameWorkers.length} workers in latest frame`)
      } catch (error) {
        console.error('NETRA live frame inference failed:', error)
        setMessage(`Live scan error: ${error instanceof Error ? error.message : 'Unknown error'}. Retrying...`)
      }

      if (liveScanningRef.current) {
        liveScanTimer.current = setTimeout(() => void scanFrame(), 1200)
      }
    }

    void scanFrame()
  }

  const stopLiveInspection = async () => {
    if (!liveScanningRef.current) return
    liveScanningRef.current = false
    clearLiveOverlay()
    if (liveScanTimer.current) {
      clearTimeout(liveScanTimer.current)
      liveScanTimer.current = null
    }
    setLiveScanning(false)

    const session = liveSessionRef.current
    liveSessionRef.current = null
    if (!session || session.scanCount === 0) {
      setMessage('Live inspection stopped. No successful frames were analyzed, so no report was saved.')
      return
    }

    const endedAt = new Date().toISOString()
    const durationSeconds = Math.max(1, Math.round((Date.now() - session.startedMs) / 1000))
    const requirements: PpeRequirements = {
      helmet: helmetRequired,
      vest: vestRequired,
      glasses: glassesRequired,
      gloves: glovesRequired,
      shoes: shoesRequired,
    }
    const record: SavedInspection = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: endedAt,
      inputMode: 'camera',
      sourceName: `Live camera session (${session.scanCount} frames)`,
      workers: session.latestWorkers,
      requirements,
      evidenceImages: session.evidenceImages,
      modelVersion: MODEL_VERSION,
      cameraSession: {
        startedAt: session.startedAt,
        endedAt,
        durationSeconds,
        scanCount: session.scanCount,
        detectionCounts: session.detectionCounts,
      },
    }

    setWorkers(record.workers)
    setInspectionTimestamp(endedAt)
    setSelectedHistoryId(record.id)
    setViolationFilter('ALL')
    setPostInspectionView('violations')
    try {
      await saveInspectionToDatabase(record)
      setSavedInspections((current) => [record, ...current].sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)))
      setMessage(`Live inspection saved: ${session.scanCount} frames over ${durationSeconds}s. Download the session PDF from Inspection Results.`)
    } catch (error) {
      console.error('Could not save live inspection:', error)
      setMessage('Live scan ended, but the report could not be saved to browser history. You can still download the current report.')
    }
  }

  // Analyze a short uploaded video locally in the browser by sampling frames sequentially.
  const handleAnalyzeVideo = async () => {
    if (!selectedFile || inputMode !== 'video') {
      setMessage('Upload a valid 10–30 second video first.')
      return
    }
    if (selectedFile.size > 50 * 1024 * 1024) {
      setMessage('Video exceeds the 50 MB limit.')
      return
    }

    const video = document.createElement('video')
    const analysisUrl = URL.createObjectURL(selectedFile)
    video.preload = 'auto'
    video.muted = true
    video.playsInline = true

    const waitFor = (eventName: 'loadedmetadata' | 'loadeddata' | 'seeked') =>
      new Promise<void>((resolve, reject) => {
        let timeoutId: ReturnType<typeof setTimeout>
        const cleanup = () => {
          window.clearTimeout(timeoutId)
          video.removeEventListener(eventName, onDone)
          video.removeEventListener('error', onError)
        }
        const onDone = () => { cleanup(); resolve() }
        const onError = () => {
          cleanup()
          const mediaError = video.error
          const reason = mediaError?.code === 3
            ? 'The browser could not decode this video. Try exporting it as MP4 (H.264 video + AAC audio).'
            : mediaError?.code === 4
              ? 'This video format or codec is not supported by this browser. Try MP4 (H.264).'
              : 'The browser could not read this video. Try MP4 (H.264).'
          reject(new Error(reason))
        }
        video.addEventListener(eventName, onDone, { once: true })
        video.addEventListener('error', onError, { once: true })
        timeoutId = window.setTimeout(() => {
          cleanup()
          reject(new Error('Video loading timed out. Try a smaller MP4 (H.264) file.'))
        }, 15000)
      })

    try {
      setAnalyzing(true)
      setVideoProgress(0)
      setMessage('Preparing video frames for PPE analysis...')

      const metadataReady = waitFor('loadedmetadata')
      video.src = analysisUrl
      video.load()
      await metadataReady

      if (!Number.isFinite(video.duration) || video.duration < 10 || video.duration > 30) {
        throw new Error('Video must be between 10 and 30 seconds long.')
      }

      // Wait until the first decoded frame is available before running inference.
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        const frameReady = waitFor('loadeddata')
        await frameReady
      }

      const requirements: PpeRequirements = {
        helmet: helmetRequired, vest: vestRequired, glasses: glassesRequired,
        gloves: glovesRequired, shoes: shoesRequired,
      }
      const startedAt = new Date().toISOString()
      const durationSeconds = Math.round(video.duration)
      const sampleInterval = 1.5
      const sampleTimes: number[] = []
      for (let t = 0.25; t < video.duration - 0.1; t += sampleInterval) sampleTimes.push(t)
      if (sampleTimes.length === 0) sampleTimes.push(0.1)

      const detectionCounts: Record<string, number> = {}
      let latestWorkers: Worker[] = []
      let completed = 0
      const evidenceImages: string[] = []

      const seekTo = async (time: number) => {
        if (Math.abs(video.currentTime - time) < 0.03 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) return
        const seeked = waitFor('seeked')
        video.currentTime = time
        await seeked
        if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
          await waitFor('loadeddata')
        }
      }

      for (let index = 0; index < sampleTimes.length; index += 1) {
        await seekTo(sampleTimes[index])
        if (video.videoWidth === 0 || video.videoHeight === 0) {
          throw new Error('A video frame could not be decoded. Try MP4 (H.264).')
        }
        const result = await runPpeInference(video)
        latestWorkers = associateDetectionsWithWorkers(result.detections, requirements)
        for (const detection of result.detections) {
          detectionCounts[detection.className] = (detectionCounts[detection.className] || 0) + 1
        }

        // Preserve the first three samples and the final sample for compact PDF evidence.
        if (index < 3 || index === sampleTimes.length - 1) {
          const canvas = document.createElement('canvas')
          const scale = Math.min(1, 1280 / result.imageWidth)
          canvas.width = Math.max(1, Math.round(result.imageWidth * scale))
          canvas.height = Math.max(1, Math.round(result.imageHeight * scale))
          const context = canvas.getContext('2d')
          if (context) {
            context.drawImage(video, 0, 0, canvas.width, canvas.height)
            context.save()
            context.scale(scale, scale)
            context.lineWidth = Math.max(2, result.imageWidth / 500)
            const font = Math.max(13, Math.round(result.imageWidth / 65))
            context.font = `600 ${font}px Arial, sans-serif`
            context.textBaseline = 'top'
            for (const detection of result.detections) {
              const [x1, y1, x2, y2] = detection.bbox
              const missing = detection.className.startsWith('NO-') || detection.className.startsWith('no_')
              const color = detection.className === 'Person' ? '#38bdf8' : missing ? '#ef4444' : '#34d399'
              const label = `${detection.className} ${(detection.confidence * 100).toFixed(1)}%`
              const labelHeight = font + 8
              const labelY = Math.max(0, y1 - labelHeight)
              context.strokeStyle = color
              context.strokeRect(x1, y1, x2 - x1, y2 - y1)
              context.fillStyle = color
              context.fillRect(x1, labelY, Math.min(context.measureText(label).width + 12, result.imageWidth - x1), labelHeight)
              context.fillStyle = '#07110e'
              context.fillText(label, x1 + 6, labelY + 4)
            }
            context.restore()
            const image = canvas.toDataURL('image/jpeg', 0.76)
            if (index === sampleTimes.length - 1 && evidenceImages.length >= 4) evidenceImages[3] = image
            else if (evidenceImages.length < 4) evidenceImages.push(image)
          }
        }

        completed += 1
        setVideoProgress(Math.round((completed / sampleTimes.length) * 100))
        setWorkers(latestWorkers)
        setMessage(`ANALYZING VIDEO · ${completed}/${sampleTimes.length} frames · ${Math.round((completed / sampleTimes.length) * 100)}%`)
      }

      const endedAt = new Date().toISOString()
      const record: SavedInspection = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        timestamp: endedAt,
        inputMode: 'video',
        sourceName: selectedFile.name,
        workers: latestWorkers,
        requirements,
        modelVersion: MODEL_VERSION,
        evidenceImages,
        cameraSession: { startedAt, endedAt, durationSeconds, scanCount: completed, detectionCounts },
      }
      setWorkers(latestWorkers)
      setInspectionTimestamp(endedAt)
      setSelectedHistoryId(record.id)
      setViolationFilter('ALL')
      setPostInspectionView('violations')
      await saveInspectionToDatabase(record)
      setSavedInspections((current) => [record, ...current].sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)))
      setMessage(`Video inspection complete · ${completed} frames analyzed across ${durationSeconds} seconds. PDF evidence is ready.`)
    } catch (error) {
      console.error('NETRA video analysis failed:', error)
      setMessage(`Video analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      video.pause()
      video.removeAttribute('src')
      video.load()
      URL.revokeObjectURL(analysisUrl)
      setAnalyzing(false)
    }
  }

  // Run browser-side inference for a single uploaded image.
  const handleAnalyze = async () => {
    const inspectionReady =
      inputMode === 'image'
        ? Boolean(selectedFile)
        : Boolean(cameraStream)

    if (!inspectionReady) {
      setMessage(inputMode === 'image' ? 'Please upload an image first.' : 'Please start the camera first.')
      return
    }

    let objectUrl: string | null = null

    try {
      setAnalyzing(true)
      setMessage('Detecting workers and associating PPE. Please wait...')

      let source: HTMLImageElement | HTMLVideoElement

      if (inputMode === 'image' && selectedFile) {
        const image = new Image()
        objectUrl = URL.createObjectURL(selectedFile)
        image.src = objectUrl
        await image.decode()
        source = image
      } else {
        const video = videoRef.current
        if (!video || video.readyState < 2) {
          throw new Error('Camera frame is not ready.')
        }
        source = video
      }

      const result = await runPpeInference(source)
      const requirements: PpeRequirements = {
        helmet: helmetRequired,
        vest: vestRequired,
        glasses: glassesRequired,
        gloves: glovesRequired,
        shoes: shoesRequired,
      }
      const detectedWorkers = associateDetectionsWithWorkers(
        result.detections,
        requirements
      )

      let annotatedEvidence: string | undefined

      // Draw detections on the original image. Keep source dimensions so the
      // browser can scale the complete image without cropping the overlay.
      if (inputMode === 'image') {
        const canvas = document.createElement('canvas')
        canvas.width = result.imageWidth
        canvas.height = result.imageHeight
        const context = canvas.getContext('2d')
        if (!context) throw new Error('Could not create detection overlay canvas.')

        context.drawImage(source, 0, 0, result.imageWidth, result.imageHeight)
        const lineWidth = Math.max(2, result.imageWidth / 450)
        const fontSize = Math.max(12, Math.round(result.imageWidth / 55))
        context.lineWidth = lineWidth
        context.font = `600 ${fontSize}px Arial, sans-serif`
        context.textBaseline = 'top'

        for (const detection of result.detections) {
          const [x1, y1, x2, y2] = detection.bbox
          const isPerson = detection.className === 'Person'
          const isMissing = detection.className.startsWith('NO-') ||
            detection.className.startsWith('no_')
          const color = isPerson ? '#38bdf8' : isMissing ? '#ef4444' : '#34d399'
          const label = `${detection.className} ${(detection.confidence * 100).toFixed(1)}%`
          const labelHeight = fontSize + 8
          const labelWidth = Math.min(
            context.measureText(label).width + 12,
            Math.max(0, result.imageWidth - x1)
          )
          const labelY = Math.max(0, y1 - labelHeight)

          context.strokeStyle = color
          context.strokeRect(x1, y1, x2 - x1, y2 - y1)
          context.fillStyle = color
          context.fillRect(x1, labelY, labelWidth, labelHeight)
          context.fillStyle = '#07110e'
          context.fillText(label, x1 + 6, labelY + 4)
        }

        annotatedEvidence = canvas.toDataURL('image/jpeg', 0.86)
        const annotatedPreview = canvas.toDataURL('image/png')
        setAnnotatedImage(annotatedPreview)
        setPreview(annotatedPreview)
      }

      const timestamp = new Date().toISOString()
      const record: SavedInspection = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        timestamp,
        inputMode,
        sourceName: inputMode === 'image'
          ? selectedFile?.name || 'Uploaded image'
          : 'Live camera capture',
        workers: detectedWorkers,
        requirements,
        modelVersion: MODEL_VERSION,
        evidenceImages: annotatedEvidence ? [annotatedEvidence] : undefined,
      }

      setWorkers(detectedWorkers)
      setExpandedWorkers([])
      setInspectionTimestamp(timestamp)
      setSelectedHistoryId(record.id)
      setViolationFilter('ALL')
      setPostInspectionView('violations')

      await saveInspectionToDatabase(record)
      setSavedInspections((current) =>
        [record, ...current.filter((item) => item.id !== record.id)]
          .sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp))
      )

      console.log('NETRA detections:', result.detections)
      console.log('NETRA worker results:', detectedWorkers)
      console.log('Total detections:', result.detections.length)
      console.log('Workers identified:', detectedWorkers.length)

      setMessage(
        detectedWorkers.length
          ? `Inspection completed: ${detectedWorkers.length} workers identified. Review each worker's PPE evidence before acting.`
          : 'Inspection completed, but no people were detected. No worker records were created.'
      )
    } catch (error) {
      console.error('NETRA detection failed:', error)
      setMessage(`Detection failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      if (objectUrl) URL.revokeObjectURL(objectUrl)
      setAnalyzing(false)
    }
  }

  const compliantCount = workers.filter(
    (worker) => worker.status === 'COMPLIANT'
  ).length

  const violationCount = workers.filter(
    (worker) => worker.status !== 'COMPLIANT'
  ).length

  const ppeChecks = [
    { key: 'helmet', label: 'Safety Helmet', required: helmetRequired, icon: <HardHat size={16} /> },
    { key: 'vest', label: 'Safety Vest', required: vestRequired, icon: <Shirt size={16} /> },
    { key: 'glasses', label: 'Safety Glasses', required: glassesRequired, icon: <Glasses size={16} /> },
    { key: 'gloves', label: 'Safety Gloves', required: glovesRequired, icon: <Hand size={16} /> },
    { key: 'shoes', label: 'Safety Shoes', required: shoesRequired, icon: <Footprints size={16} /> },
  ] as const

  const getMissingPpe = (worker: Worker) =>
    ppeChecks
      .filter((item) => item.required && !worker.ppe[item.key])
      .map((item) => item.label)

  const getExplicitlyMissingPpe = (worker: Worker) =>
    ppeChecks
      .filter((item) => item.required && worker.evidence?.[item.key] === 'EXPLICITLY_MISSING')
      .map((item) => item.label)

  // VIOLATION MANAGEMENT
  // Severity is a triage heuristic, not a validated industrial risk score.
  const violations: Violation[] = inspectionTimestamp
    ? workers
        .filter((worker) => worker.status === 'VIOLATION')
        .map((worker) => {
          const missingItems = getExplicitlyMissingPpe(worker)

          const severity: ViolationSeverity =
            helmetRequired && worker.evidence?.helmet === 'EXPLICITLY_MISSING'
              ? 'CRITICAL'
              : missingItems.length >= 2
                ? 'HIGH'
                : 'MEDIUM'

          return {
            workerId: worker.id,
            missingItems,
            severity,
            timestamp: inspectionTimestamp,
          }
        })
    : []

  const filteredViolations =
    violationFilter === 'ALL'
      ? violations
      : violations.filter((violation) => violation.severity === violationFilter)

  const severityCounts = {
    CRITICAL: violations.filter((violation) => violation.severity === 'CRITICAL').length,
    HIGH: violations.filter((violation) => violation.severity === 'HIGH').length,
    MEDIUM: violations.filter((violation) => violation.severity === 'MEDIUM').length,
  }

  // ANALYTICS CALCULATIONS
  const complianceRate =
    workers.length > 0
      ? Math.round((compliantCount / workers.length) * 100)
      : 0

  const ppeAnalytics = [
    {
      label: 'Safety Helmet',
      required: helmetRequired,
      passed: workers.filter((worker) => worker.ppe.helmet).length,
    },
    {
      label: 'Safety Vest',
      required: vestRequired,
      passed: workers.filter((worker) => worker.ppe.vest).length,
    },
    {
      label: 'Safety Glasses',
      required: glassesRequired,
      passed: workers.filter((worker) => worker.ppe.glasses).length,
    },
    {
      label: 'Safety Gloves',
      required: glovesRequired,
      passed: workers.filter((worker) => worker.ppe.gloves).length,
    },
    {
      label: 'Safety Shoes',
      required: shoesRequired,
      passed: workers.filter((worker) => worker.ppe.shoes).length,
    },
  ].map((item) => ({
    ...item,
    rate:
      item.required && workers.length > 0
        ? Math.round((item.passed / workers.length) * 100)
        : 0,
  }))

  const formatViolationTime = (timestamp: string) =>
    new Date(timestamp).toLocaleString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })

  const openSavedInspection = (record: SavedInspection) => {
    stopCamera()
    setSelectedHistoryId(record.id)
    setWorkers(record.workers)
    setInspectionTimestamp(record.timestamp)
    setInputMode(record.inputMode)
    setHelmetRequired(record.requirements.helmet)
    setVestRequired(record.requirements.vest)
    setGlassesRequired(record.requirements.glasses)
    setGlovesRequired(record.requirements.gloves)
    setShoesRequired(record.requirements.shoes)
    setExpandedWorkers([])
    setViolationFilter('ALL')
    setPostInspectionView('violations')
    setMessage(`Loaded saved inspection: ${record.sourceName}.`)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleDeleteInspection = async (id: string) => {
    if (!window.confirm('Permanently delete this inspection from this browser?')) return
    try {
      await deleteInspectionFromDatabase(id)
      setSavedInspections((items) => items.filter((item) => item.id !== id))
      if (selectedHistoryId === id) setSelectedHistoryId(null)
      setMessage('Inspection deleted from local history.')
    } catch {
      setHistoryError('Unable to delete this inspection.')
    }
  }

  const downloadInspectionReport = (record: SavedInspection) => {
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
    const pageWidth = pdf.internal.pageSize.getWidth()
    const pageHeight = pdf.internal.pageSize.getHeight()
    const margin = 15
    const contentWidth = pageWidth - margin * 2
    let y = 56
    const total = record.workers.length
    const compliant = record.workers.filter((worker) => worker.status === 'COMPLIANT').length
    const violations = record.workers.filter((worker) => worker.status === 'VIOLATION').length
    const review = record.workers.filter((worker) => worker.status === 'REVIEW').length
    const rate = total ? Math.round((compliant / total) * 100) : 0
    const requirements = record.requirements
    const ppe = [
      { key: 'helmet' as const, label: 'Helmet' },
      { key: 'vest' as const, label: 'Vest' },
      { key: 'glasses' as const, label: 'Goggles' },
      { key: 'gloves' as const, label: 'Gloves' },
      { key: 'shoes' as const, label: 'Safety boots' },
    ]
    const colors = {
      ink: [31, 48, 45] as [number, number, number],
      muted: [104, 119, 114] as [number, number, number],
      green: [22, 126, 100] as [number, number, number],
      red: [190, 65, 61] as [number, number, number],
      amber: [176, 127, 40] as [number, number, number],
      pale: [239, 246, 243] as [number, number, number],
      line: [220, 229, 225] as [number, number, number],
    }
    const addHeader = (subtitle: string) => {
      pdf.setFillColor(20, 43, 40)
      pdf.rect(0, 0, pageWidth, 39, 'F')
      pdf.setTextColor(225, 246, 237)
      pdf.setFont('helvetica', 'bold')
      pdf.setFontSize(21)
      pdf.text('NETRA', margin, 17)
      pdf.setFont('helvetica', 'normal')
      pdf.setFontSize(8)
      pdf.text(subtitle.toUpperCase(), margin, 25)
      pdf.setFontSize(8)
      pdf.text(`REPORT ID  ${record.id.slice(-12).toUpperCase()}`, pageWidth - margin, 17, { align: 'right' })
    }
    const section = (title: string) => {
      if (y > pageHeight - 28) { pdf.addPage(); addHeader('Inspection report'); y = 52 }
      pdf.setFont('helvetica', 'bold')
      pdf.setFontSize(11)
      pdf.setTextColor(...colors.green)
      pdf.text(title.toUpperCase(), margin, y)
      y += 7
      pdf.setDrawColor(...colors.line)
      pdf.line(margin, y - 3, pageWidth - margin, y - 3)
    }
    const textLine = (value: string, x = margin, maxWidth = contentWidth, size = 9, color = colors.ink) => {
      pdf.setFont('helvetica', 'normal')
      pdf.setFontSize(size)
      pdf.setTextColor(...color)
      const lines = pdf.splitTextToSize(value, maxWidth)
      const height = lines.length * 4.5
      if (y + height > pageHeight - 16) { pdf.addPage(); addHeader('Inspection report'); y = 52 }
      pdf.text(lines, x, y)
      y += height + 2
    }
    const card = (x: number, top: number, width: number, label: string, value: string, accent: [number, number, number]) => {
      pdf.setFillColor(247, 250, 248)
      pdf.roundedRect(x, top, width, 20, 2, 2, 'F')
      pdf.setFillColor(...accent)
      pdf.roundedRect(x, top, 2, 20, 1, 1, 'F')
      pdf.setFont('helvetica', 'normal')
      pdf.setFontSize(7)
      pdf.setTextColor(...colors.muted)
      pdf.text(label.toUpperCase(), x + 6, top + 7)
      pdf.setFont('helvetica', 'bold')
      pdf.setFontSize(15)
      pdf.setTextColor(...colors.ink)
      pdf.text(value, x + 6, top + 15)
    }

    addHeader('Industrial safety inspection report')
    pdf.setFont('helvetica', 'bold')
    pdf.setFontSize(16)
    pdf.setTextColor(...colors.ink)
    pdf.text('Inspection overview', margin, y)
    y += 8
    textLine(`${formatViolationTime(record.timestamp)}  •  ${record.inputMode === 'camera' ? 'Live camera' : record.inputMode === 'video' ? 'Uploaded video' : 'Uploaded image'}`, margin, contentWidth, 9, colors.muted)
    textLine(`Source: ${record.sourceName}`, margin, contentWidth, 8, colors.muted)
    const gap = 4
    const cardWidth = (contentWidth - gap * 3) / 4
    const cardY = y + 1
    card(margin, cardY, cardWidth, 'Workers', String(total), colors.ink)
    card(margin + cardWidth + gap, cardY, cardWidth, 'Compliant', String(compliant), colors.green)
    card(margin + (cardWidth + gap) * 2, cardY, cardWidth, 'Violations', String(violations), colors.red)
    card(margin + (cardWidth + gap) * 3, cardY, cardWidth, 'Review', String(review), colors.amber)
    y = cardY + 27
    pdf.setFillColor(239, 246, 243)
    pdf.roundedRect(margin, y, contentWidth, 12, 2, 2, 'F')
    pdf.setFont('helvetica', 'bold')
    pdf.setFontSize(9)
    pdf.setTextColor(...colors.green)
    pdf.text(`COMPLIANCE RATE  ${rate}%`, margin + 5, y + 7.5)
    y += 20

    section('PPE requirements')
    const enabled = ppe.filter((item) => requirements[item.key]).map((item) => item.label)
    textLine(enabled.length ? `Required: ${enabled.join('  •  ')}` : 'No PPE requirements were enabled for this inspection.')
    const notRequired = ppe.filter((item) => !requirements[item.key]).map((item) => item.label)
    if (notRequired.length) textLine(`Not required: ${notRequired.join('  •  ')}`, margin, contentWidth, 8, colors.muted)

    if (record.cameraSession) {
      section('Camera session')
      textLine(`${record.inputMode === 'video' ? 'Video duration' : 'Session duration'}: ${record.cameraSession.durationSeconds}s     Frames analyzed: ${record.cameraSession.scanCount}`)
      textLine(`${record.inputMode === 'video' ? 'Video processing started' : 'Started'}: ${formatViolationTime(record.cameraSession.startedAt)}`)
      textLine(`${record.inputMode === 'video' ? 'Video processing completed' : 'Stopped'}: ${formatViolationTime(record.cameraSession.endedAt)}`)
      textLine('Worker results represent the final analyzed frame. The system does not track unique worker identities across frames.', margin, contentWidth, 8, colors.muted)
    }

    section('Worker results')
    if (!record.workers.length) {
      textLine('No workers were identified in the analyzed frame. This does not prove the scene was safe; review the visual evidence.')
    } else {
      const colX = [margin, margin + 22, margin + 53, margin + 91, pageWidth - margin]
      pdf.setFillColor(239, 246, 243)
      pdf.rect(margin, y - 4, contentWidth, 9, 'F')
      pdf.setFont('helvetica', 'bold')
      pdf.setFontSize(7)
      pdf.setTextColor(...colors.muted)
      pdf.text('WORKER', colX[0] + 2, y + 2)
      pdf.text('STATUS', colX[1], y + 2)
      pdf.text('MISSING / UNCONFIRMED PPE', colX[2], y + 2)
      pdf.text('CONF.', colX[3], y + 2)
      y += 11
      record.workers.forEach((worker) => {
        const uncertain = ppe.filter((item) => requirements[item.key] && !worker.ppe[item.key]).map((item) => item.label)
        const statusColor = worker.status === 'COMPLIANT' ? colors.green : worker.status === 'VIOLATION' ? colors.red : colors.amber
        const rowText = uncertain.length ? uncertain.join(', ') : 'None'
        const wrapped = pdf.splitTextToSize(rowText, 36)
        const rowHeight = Math.max(9, wrapped.length * 3.5 + 3)
        if (y + rowHeight > pageHeight - 17) { pdf.addPage(); addHeader('Inspection report'); y = 52 }
        pdf.setDrawColor(...colors.line)
        pdf.line(margin, y + rowHeight - 1, pageWidth - margin, y + rowHeight - 1)
        pdf.setFont('helvetica', 'bold')
        pdf.setFontSize(8)
        pdf.setTextColor(...colors.ink)
        pdf.text(`Worker ${worker.id}`, colX[0] + 2, y + 4)
        pdf.setTextColor(...statusColor)
        pdf.text(worker.status, colX[1], y + 4)
        pdf.setFont('helvetica', 'normal')
        pdf.setFontSize(7)
        pdf.setTextColor(...colors.ink)
        pdf.text(wrapped, colX[2], y + 4)
        pdf.text(worker.confidence != null ? `${Math.round(worker.confidence * 100)}%` : '—', colX[3], y + 4)
        y += rowHeight
      })
    }

    if (record.evidenceImages?.length) {
      pdf.addPage()
      addHeader('Annotated visual evidence')
      y = 51
      pdf.setFont('helvetica', 'bold')
      pdf.setFontSize(14)
      pdf.setTextColor(...colors.ink)
      pdf.text('Visual inspection evidence', margin, y)
      y += 7
      textLine('Bounding boxes show model detections. Blue = person, green = detected PPE, red = explicitly missing PPE.', margin, contentWidth, 8, colors.muted)
      record.evidenceImages.forEach((dataUrl, index) => {
        try {
          const props = pdf.getImageProperties(dataUrl)
          const maxW = contentWidth
          const maxH = record.evidenceImages!.length > 1 ? 92 : 150
          const scale = Math.min(maxW / props.width, maxH / props.height)
          const w = props.width * scale
          const h = props.height * scale
          if (y + h + 12 > pageHeight - 12) { pdf.addPage(); addHeader('Annotated visual evidence'); y = 51 }
          pdf.setFont('helvetica', 'bold')
          pdf.setFontSize(8)
          pdf.setTextColor(...colors.green)
          pdf.text(`FRAME ${index + 1}${record.cameraSession ? ' · SELECTED SAMPLE' : ''}`, margin, y)
          y += 4
          pdf.addImage(dataUrl, 'JPEG', margin, y, w, h, undefined, 'FAST')
          y += h + 10
        } catch (error) {
          console.warn('Could not embed inspection evidence image:', error)
        }
      })
    }

    // Footer on every page, added after all content is laid out.
    const pages = pdf.getNumberOfPages()
    for (let page = 1; page <= pages; page += 1) {
      pdf.setPage(page)
      pdf.setDrawColor(...colors.line)
      pdf.line(margin, pageHeight - 10, pageWidth - margin, pageHeight - 10)
      pdf.setFont('helvetica', 'normal')
      pdf.setFontSize(7)
      pdf.setTextColor(...colors.muted)
      pdf.text('NETRA · AI-assisted screening only · Verify findings before safety action', margin, pageHeight - 5)
      pdf.text(`${page} / ${pages}`, pageWidth - margin, pageHeight - 5, { align: 'right' })
    }
    pdf.save(`NETRA-Inspection-${record.timestamp.replace(/[:.]/g, '-')}.pdf`)
  }

  // Let the supervisor export the inspection currently shown in the results panel.
  // Prefer the saved IndexedDB record so the PDF exactly matches the stored inspection.
  const downloadCurrentInspectionReport = () => {
    if (!inspectionTimestamp) {
      setMessage('Run an inspection before downloading its report.')
      return
    }

    const savedRecord = selectedHistoryId
      ? savedInspections.find((record) => record.id === selectedHistoryId)
      : undefined

    const currentRecord: SavedInspection = savedRecord || {
      id: selectedHistoryId || `${Date.parse(inspectionTimestamp)}`,
      timestamp: inspectionTimestamp,
      inputMode,
      sourceName: inputMode === 'image' ? selectedFile?.name || 'Inspection image' : inputMode === 'video' ? selectedFile?.name || 'Uploaded video' : 'Live camera capture',
      workers,
      requirements: {
        helmet: helmetRequired,
        vest: vestRequired,
        glasses: glassesRequired,
        gloves: glovesRequired,
        shoes: shoesRequired,
      },
      modelVersion: MODEL_VERSION,
      evidenceImages: preview?.startsWith('data:image/') ? [preview] : undefined,
    }

    downloadInspectionReport(currentRecord)
  }

  const downloadWorkerReport = (worker: Worker) => {
    const pdf = new jsPDF()
    const pageWidth = pdf.internal.pageSize.getWidth()
    const pageHeight = pdf.internal.pageSize.getHeight()
    const margin = 16
    let y = 20

    const line = (value: string, size = 10, bold = false, color: [number, number, number] = [45, 60, 57]) => {
      pdf.setFont('helvetica', bold ? 'bold' : 'normal')
      pdf.setFontSize(size)
      pdf.setTextColor(...color)
      const wrapped = pdf.splitTextToSize(value, pageWidth - margin * 2)
      const needed = wrapped.length * size * 0.5 + 7
      if (y + needed > pageHeight - 18) { pdf.addPage(); y = 20 }
      pdf.text(wrapped, margin, y)
      y += needed
    }

    const requirements: PpeRequirements = {
      helmet: helmetRequired,
      vest: vestRequired,
      glasses: glassesRequired,
      gloves: glovesRequired,
      shoes: shoesRequired,
    }
    const ppe = [
      { key: 'helmet' as const, label: 'Safety Helmet' },
      { key: 'vest' as const, label: 'Safety Vest' },
      { key: 'glasses' as const, label: 'Safety Glasses' },
      { key: 'gloves' as const, label: 'Safety Gloves' },
      { key: 'shoes' as const, label: 'Safety Shoes' },
    ]
    const missing = ppe.filter((item) => requirements[item.key] && !worker.ppe[item.key])
    const severity: ViolationSeverity | 'NONE' = missing.length === 0 ? 'NONE' : requirements.helmet && !worker.ppe.helmet ? 'CRITICAL' : missing.length >= 2 ? 'HIGH' : 'MEDIUM'

    pdf.setFillColor(20, 43, 40)
    pdf.rect(0, 0, pageWidth, 42, 'F')
    pdf.setTextColor(220, 245, 236)
    pdf.setFont('helvetica', 'bold')
    pdf.setFontSize(23)
    pdf.text('NETRA', margin, 20)
    pdf.setFont('helvetica', 'normal')
    pdf.setFontSize(9)
    pdf.text('INDIVIDUAL WORKER COMPLIANCE REPORT', margin, 29)
    pdf.setFontSize(8)
    pdf.text('SAVED INSPECTION RECORD', margin, 36)
    y = 56

    line(`WORKER ${worker.id}`, 15, true, [27, 105, 91])
    line(`Inspection date: ${inspectionTimestamp ? formatViolationTime(inspectionTimestamp) : 'Not available'}`)
    line(`Inspection input: ${inputMode.toUpperCase()}`)
    line(`Overall status: ${worker.status}`, 12, true, worker.status === 'COMPLIANT' ? [35, 135, 99] : worker.status === 'REVIEW' ? [190, 145, 55] : [190, 72, 67])
    line(`Missing required PPE: ${missing.length}`)
    line(`Violation severity: ${severity}`, 11, true, severity === 'NONE' ? [35, 135, 99] : [190, 72, 67])

    line('PPE COMPLIANCE DETAILS', 13, true, [27, 105, 91])
    ppe.forEach((item) => {
      const required = requirements[item.key]
      const detected = worker.ppe[item.key]
      const evidence = worker.evidence?.[item.key]
      const result = !required ? 'NOT REQUIRED' : evidence === 'EXPLICITLY_MISSING' ? 'MODEL FLAGGED MISSING' : detected ? 'DETECTED' : 'NOT DETECTED — REVIEW'
      line(`${item.label}: ${result}`, 10, required && !detected, required && !detected ? [190, 72, 67] : [45, 60, 57])
    })

    line('IMPORTANT DISCLAIMER', 11, true, [190, 95, 50])
    line('This report contains a previously saved inspection record. Verify all findings before taking workplace safety action.', 9)
    pdf.save(`NETRA-Worker-${worker.id}-${(inspectionTimestamp || new Date().toISOString()).replace(/[:.]/g, '-')}.pdf`)
  }

  const toggleWorkerDetails = (workerId: number) => {
    setExpandedWorkers((current) =>
      current.includes(workerId)
        ? current.filter((id) => id !== workerId)
        : [...current, workerId]
    )
  }

  return (
    <>
      {/* CINEMATIC INTRO */}
      {!dashboard && (
        <main className={`cinematic-page ${launching ? 'intro-launching' : ''}`}>
          <div className="intro-warehouse-scene" ref={mountRef} />

          <div className="cinematic-vignette" />
          <div className="cinematic-grain" />
          <div className="cinematic-scanline" />

          {/* RETICLE KEPT; UNWANTED OVERLAY TEXT REMOVED */}
          <div className="scene-reticle">
            <span />
            <span />
            <span />
            <span />
          </div>

          <section className="netra-title">
            <div className="title-overline">
              <span className="title-line" />
              INTELLIGENT VISION SYSTEM
              <span className="title-line" />
            </div>

            <h1 className="title-word">
              <span>N</span>
              <span>E</span>
              <span>T</span>
              <span>R</span>
              <span>A</span>
            </h1>

            <div className="title-underline" />

            <p className="title-tagline">
              INDUSTRIAL SAFETY <span>·</span> COMPLIANCE
            </p>

            <div className="title-footer">
              <span>VISION THAT PREVENTS</span>
              <span>. / 2026</span>
            </div>
          </section>

          <section className="initiate-section">
            <div className="initiate-label">SYSTEM READY FOR INSPECTION</div>

            <button
              className="initiate-button"
              onClick={initiateWork}
              disabled={launching}
            >
              LET'S INITIATE THE WORK
              <span>↗</span>
            </button>

            <div className="initiate-note">
              AI-POWERED PERSONAL PROTECTIVE EQUIPMENT MONITORING
            </div>
          </section>

          <div className="scene-bottom">
            <div>
              <span className="scene-caption">NETRA / VISUAL ENGINE</span>
              <span className="scene-subcaption">
                INDUSTRIAL ENVIRONMENT · 3D REAL-TIME RENDER
              </span>
            </div>

            <div className="scene-progress">
              <span />
            </div>
          </div>
        </main>
      )}

      {/* DASHBOARD */}
      {dashboard && (
        <main className="netra-dashboard">
          <header className="dashboard-header">
            <div className="dashboard-brand">
              <div className="dashboard-brand-icon">
                <ShieldCheck size={22} />
              </div>

              <div>
                <h2>NETRA</h2>
                <span>INDUSTRIAL SAFETY INTELLIGENCE</span>
              </div>
            </div>

            <div className="dashboard-header-right">
              <span className="system-indicator">
                <i />
                SYSTEM ONLINE
              </span>

              <span className="dashboard-version">VISION ENGINE / 01</span>
            </div>
          </header>

          <div className="dashboard-content">
            <section className="dashboard-heading">
              <div>
                <div className="dashboard-eyebrow">
                  <span /> PPE MONITORING SYSTEM
                </div>

                <h1>Inspection Dashboard</h1>

                <p>
                  Upload a workplace image to identify workers and verify
                  personal protective equipment compliance.
                </p>
              </div>

              <div className="dashboard-heading-tag">
                <ScanSearch size={17} />
                AI VISION ANALYSIS
              </div>
            </section>

            {/* STATISTICS */}
            <section className="dashboard-stats">
              <div className="stat-card">
                <div className="stat-icon">
                  <Users size={19} />
                </div>

                <div>
                  <span>WORKERS DETECTED</span>
                  <strong>{workers.length}</strong>
                </div>

                <small>01</small>
              </div>

              <div className="stat-card compliant-stat">
                <div className="stat-icon">
                  <CheckCircle2 size={19} />
                </div>

                <div>
                  <span>COMPLIANT</span>
                  <strong>{compliantCount}</strong>
                </div>

                <small>02</small>
              </div>

              <div className="stat-card violation-stat">
                <div className="stat-icon">
                  <AlertTriangle size={19} />
                </div>

                <div>
                  <span>NEEDS ATTENTION</span>
                  <strong>{violationCount}</strong>
                </div>

                <small>03</small>
              </div>
            </section>

            {/* MAIN WORKSPACE */}
            <section className="inspection-layout">
              <div className="dashboard-panel inspection-panel">
                <div className="panel-heading">
                  <div>
                    <span className="panel-index">01 / INPUT</span>
                    <h2>New Inspection</h2>
                  </div>

                  <div
                    className="input-mode-switch"
                    aria-label="Inspection input mode"
                  >
                    <button
                      type="button"
                      className={`input-mode-button ${
                        inputMode === 'image' ? 'active' : ''
                      }`}
                      onClick={() => {
                        stopCamera()
                        setCameraError('')
                        if (inputMode === 'video') resetFile()
                        setInputMode('image')
                      }}
                      aria-pressed={inputMode === 'image'}
                    >
                      <ImageIcon size={15} /> IMAGE
                    </button>

                    <button
                      type="button"
                      className={`input-mode-button ${
                        inputMode === 'camera' ? 'active' : ''
                      }`}
                      onClick={() => {
                        setInputMode('camera')
                        setMessage('')
                      }}
                      aria-pressed={inputMode === 'camera'}
                    >
                      <Camera size={15} /> LIVE CAMERA
                    </button>
                    <button
                      type="button"
                      className={`input-mode-button ${inputMode === 'video' ? 'active' : ''}`}
                      onClick={() => { stopCamera(); if (inputMode !== 'video') resetFile(); setInputMode('video'); setMessage('Upload a 10–30 second video (maximum 50 MB).') }}
                      aria-pressed={inputMode === 'video'}
                    >
                      <UploadCloud size={15} /> VIDEO
                    </button>
                  </div>
                </div>

                {inputMode === 'image' ? (
                  <div
                    className={`upload-area ${preview ? 'has-preview' : ''}`}
                    style={preview ? { height: 'auto', minHeight: 0, overflow: 'visible' } : undefined}
                    onDragOver={(event) => event.preventDefault()}
                    onDrop={(event) => {
                      event.preventDefault()
                      handleFile(event.dataTransfer.files[0])
                    }}
                    onClick={() => {
                      if (!preview) fileRef.current?.click()
                    }}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' && !preview) {
                        fileRef.current?.click()
                      }
                    }}
                  >
                    <input
                      ref={fileRef}
                      type="file"
                      accept="image/*"
                      hidden
                      onChange={(event) =>
                        handleFile(event.target.files?.[0])
                      }
                    />

                    {preview ? (
                      <div className="upload-preview">
                        <img src={preview} alt="Inspection image with PPE detection bounding boxes" style={{ display: 'block', width: '100%', maxHeight: '440px', objectFit: 'contain', objectPosition: 'center', background: '#07110e', borderRadius: '12px' }} />

                        <button
                          className="remove-image"
                          onClick={(event) => {
                            event.stopPropagation()
                            resetFile()
                          }}
                          aria-label="Remove image"
                        >
                          <X size={17} />
                        </button>

                        <div className="preview-file-name">
                          <ImageIcon size={14} />
                          {selectedFile?.name || 'Annotated inspection image'}
                        </div>
                      </div>
                    ) : (
                      <div className="upload-placeholder">
                        <div className="upload-icon">
                          <UploadCloud size={27} />
                        </div>

                        <h3>Upload workplace image</h3>

                        <p>
                          Drag and drop your image here or browse files
                        </p>

                        <span>PNG, JPG, JPEG · IMAGE FILES</span>
                      </div>
                    )}
                  </div>
                ) : inputMode === 'video' ? (
                  <div className={`upload-area ${preview ? 'has-preview' : ''}`} style={preview ? { height: 'auto', minHeight: 0, overflow: 'visible' } : undefined} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); handleVideoFile(event.dataTransfer.files[0]) }} onClick={() => { if (!preview) videoFileRef.current?.click() }}>
                    <input ref={videoFileRef} type="file" accept="video/mp4,video/webm,video/quicktime,video/*" hidden onChange={(event) => handleVideoFile(event.target.files?.[0])} />
                    {preview ? (
                      <div className="upload-preview">
                        <video src={preview} controls muted playsInline style={{ width: '100%', maxHeight: '360px', objectFit: 'contain', background: '#07110e', borderRadius: '12px' }} />
                        <button className="remove-image" onClick={(event) => { event.stopPropagation(); resetFile() }} aria-label="Remove video"><X size={17} /></button>
                        <div className="preview-file-name"><Camera size={14} />{selectedFile?.name} · {selectedFile ? (selectedFile.size / (1024 * 1024)).toFixed(1) : '0'} MB</div>
                      </div>
                    ) : (
                      <div className="upload-placeholder">
                        <div className="upload-icon"><UploadCloud size={27} /></div>
                        <h3>Upload short workplace video</h3>
                        <p>Drag and drop a video here or browse files</p>
                        <span>MP4, WEBM, MOV · 10–30 SECONDS · MAX 50 MB</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div
                    className={`camera-area ${
                      cameraStream ? 'camera-active' : ''
                    }`}
                  >
                    {cameraStream ? (
                      <>
                        <video
                          ref={videoRef}
                          className="camera-preview"
                          autoPlay
                          muted
                          playsInline
                        />
                        <canvas
                          ref={liveOverlayRef}
                          className="camera-detection-overlay"
                          aria-label="Live PPE detection bounding boxes"
                          style={{
                            position: 'absolute',
                            inset: 0,
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            pointerEvents: 'none',
                            zIndex: 2,
                          }}
                        />

                        <div className="camera-live-indicator">
                          <i /> CAMERA ACTIVE
                        </div>

                        <div
                          className="camera-frame-corners"
                          aria-hidden="true"
                        >
                          <span />
                          <span />
                          <span />
                          <span />
                        </div>
                      </>
                    ) : (
                      <div className="camera-placeholder">
                        <div className="upload-icon">
                          <Camera size={27} />
                        </div>

                        <h3>Live camera monitoring</h3>

                        <p>
                          Start your camera to preview the workplace feed.
                        </p>

                        <button
                          type="button"
                          className="camera-action-button"
                          onClick={startCamera}
                        >
                          <Camera size={16} /> START CAMERA
                        </button>
                      </div>
                    )}

                    {cameraStream && (
                      <button
                        type="button"
                        className="camera-stop-button"
                        onClick={stopCamera}
                      >
                        <CameraOff size={15} /> STOP CAMERA
                      </button>
                    )}

                    {cameraError && (
                      <div className="camera-error">{cameraError}</div>
                    )}

                    {!cameraError && cameraStream && (
                      <div className="camera-helper-note">
                        LIVE CAMERA PREVIEW · START CONTINUOUS INSPECTION TO SCAN UNTIL STOPPED
                      </div>
                    )}
                  </div>
                )}

                <div className="requirements-heading">
                  <div>
                    <span className="panel-index">02 / CONFIGURATION</span>
                    <h3>Safety Requirements</h3>
                  </div>
                </div>

                <div className="requirements-list">
                  {/* HELMET */}
                  <div className="requirement-row">
                    <div className="requirement-icon">
                      <HardHat size={20} />
                    </div>

                    <div className="requirement-copy">
                      <strong>Safety Helmet</strong>
                      <span>Helmet detection required</span>
                    </div>

                    <button
                      className={`toggle-switch ${
                        helmetRequired ? 'active' : ''
                      }`}
                      onClick={() => setHelmetRequired(!helmetRequired)}
                      aria-label="Toggle helmet requirement"
                      aria-pressed={helmetRequired}
                    >
                      <span />
                    </button>
                  </div>

                  {/* VEST */}
                  <div className="requirement-row">
                    <div className="requirement-icon">
                      <Shirt size={20} />
                    </div>

                    <div className="requirement-copy">
                      <strong>Safety Vest</strong>
                      <span>Reflective vest detection required</span>
                    </div>

                    <button
                      className={`toggle-switch ${
                        vestRequired ? 'active' : ''
                      }`}
                      onClick={() => setVestRequired(!vestRequired)}
                      aria-label="Toggle vest requirement"
                      aria-pressed={vestRequired}
                    >
                      <span />
                    </button>
                  </div>

                  {/* GLASSES */}
                  <div className="requirement-row">
                    <div className="requirement-icon">
                      <Glasses size={20} />
                    </div>

                    <div className="requirement-copy">
                      <strong>Safety Glasses</strong>
                      <span>Eye protection detection required</span>
                    </div>

                    <button
                      className={`toggle-switch ${
                        glassesRequired ? 'active' : ''
                      }`}
                      onClick={() => setGlassesRequired(!glassesRequired)}
                      aria-label="Toggle glasses requirement"
                      aria-pressed={glassesRequired}
                    >
                      <span />
                    </button>
                  </div>

                  {/* GLOVES */}
                  <div className="requirement-row">
                    <div className="requirement-icon">
                      <Hand size={20} />
                    </div>

                    <div className="requirement-copy">
                      <strong>Safety Gloves</strong>
                      <span>Hand protection detection required</span>
                    </div>

                    <button
                      className={`toggle-switch ${
                        glovesRequired ? 'active' : ''
                      }`}
                      onClick={() => setGlovesRequired(!glovesRequired)}
                      aria-label="Toggle gloves requirement"
                      aria-pressed={glovesRequired}
                    >
                      <span />
                    </button>
                  </div>

                  {/* SHOES */}
                  <div className="requirement-row">
                    <div className="requirement-icon">
                      <Footprints size={20} />
                    </div>

                    <div className="requirement-copy">
                      <strong>Safety Shoes</strong>
                      <span>Protective footwear detection required</span>
                    </div>

                    <button
                      className={`toggle-switch ${
                        shoesRequired ? 'active' : ''
                      }`}
                      onClick={() => setShoesRequired(!shoesRequired)}
                      aria-label="Toggle shoes requirement"
                      aria-pressed={shoesRequired}
                    >
                      <span />
                    </button>
                  </div>
                </div>

                {inputMode === 'image' ? (
                  <button className="run-inspection" onClick={handleAnalyze} disabled={analyzing || !selectedFile}>
                    {analyzing ? <><LoaderCircle className="spin" size={18} /> RUNNING INSPECTION...</> : <><ScanSearch size={18} /> RUN INSPECTION <span>↗</span></>}
                  </button>
                ) : inputMode === 'video' ? (
                  <button className="run-inspection" onClick={handleAnalyzeVideo} disabled={analyzing || !selectedFile}>
                    {analyzing ? <><LoaderCircle className="spin" size={18} /> ANALYZING VIDEO · {videoProgress}%</> : <><ScanSearch size={18} /> ANALYZE VIDEO <span>↗</span></>}
                  </button>
                ) : (
                  <button
                    className="run-inspection"
                    onClick={liveScanning ? () => void stopLiveInspection() : startLiveInspection}
                    disabled={!cameraStream || analyzing}
                  >
                    {liveScanning ? <><CameraOff size={18} /> STOP & GENERATE REPORT · {liveScanCount} FRAMES</> : <><ScanSearch size={18} /> START CONTINUOUS INSPECTION <span>↗</span></>}
                  </button>
                )}

                {message && (
                  <div
                    className={`inspection-message ${
                      message.toLowerCase().includes('failed') ||
                      message.toLowerCase().includes('unable') ||
                      message.toLowerCase().includes('please')
                        ? 'error-message'
                        : ''
                    }`}
                  >
                    {message}
                  </div>
                )}
              </div>

              {/* RESULTS */}
              <div className="dashboard-panel results-panel">
                <div className="panel-heading">
                  <div>
                    <span className="panel-index">03 / OUTPUT</span>
                    <h2>Inspection Results</h2>
                  </div>

                  <span className="results-live">
                    <i />
                    ANALYSIS
                  </span>
                </div>

                {!workers.length && !inspectionTimestamp ? (
                  <div className="empty-results">
                    <div className="empty-results-icon">
                      <ScanSearch size={34} />
                    </div>

                    <h3>Awaiting Inspection</h3>

                    <p>
                      Worker and PPE results will appear here after an image or camera inspection.
                    </p>

                    <div className="empty-result-line">
                      <span />
                      NO ANALYSIS DATA AVAILABLE
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="results-summary">
                      <div>
                        <span>ANALYSIS COMPLETE</span>
                        <strong>{workers.length} Workers Identified</strong>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <button
                          type="button"
                          onClick={downloadCurrentInspectionReport}
                          disabled={!inspectionTimestamp}
                          aria-label="Download complete inspection PDF report"
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '7px',
                            padding: '9px 12px',
                            border: '1px solid rgba(130, 190, 180, 0.32)',
                            borderRadius: '8px',
                            background: 'rgba(35, 75, 70, 0.32)',
                            color: '#dcebe6',
                            cursor: inspectionTimestamp ? 'pointer' : 'not-allowed',
                            fontSize: '11px',
                            fontWeight: 700,
                            whiteSpace: 'nowrap',
                          }}
                        >
                          <FileDown size={15} /> DOWNLOAD REPORT PDF
                        </button>
                        <CheckCircle2 size={23} />
                      </div>
                    </div>

                    <p style={{ fontSize: '11px', lineHeight: 1.5, opacity: 0.68, margin: '8px 0 12px' }}>
                      PPE marked as missing means it was not detected for that worker in this frame. Occlusion, camera angle and model recall can affect results. Verify findings before taking safety action.
                    </p>
                    {inputMode === 'image' && annotatedImage && (
                      <section
                        aria-label="Annotated image detection evidence"
                        style={{
                          margin: '14px 0 18px',
                          padding: '12px',
                          border: '1px solid rgba(130, 190, 180, 0.24)',
                          borderRadius: '12px',
                          background: 'rgba(7, 17, 14, 0.72)',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
                          <strong style={{ fontSize: '12px', letterSpacing: '0.08em' }}>ANNOTATED IMAGE EVIDENCE</strong>
                          <span style={{ fontSize: '10px', opacity: 0.65 }}>DETECTION OVERLAY</span>
                        </div>
                        <img
                          src={annotatedImage}
                          alt="Uploaded workplace image annotated with PPE detection bounding boxes and confidence labels"
                          style={{ display: 'block', width: '100%', height: 'auto', maxHeight: '520px', objectFit: 'contain', objectPosition: 'center', background: '#07110e', borderRadius: '8px' }}
                        />
                        <p style={{ fontSize: '10px', lineHeight: 1.5, opacity: 0.62, margin: '8px 0 0' }}>Blue: person · Green: detected PPE · Red: explicitly missing PPE</p>
                      </section>
                    )}
                    <div className="worker-list">
                      {workers.map((worker) => {
                        const isExpanded = expandedWorkers.includes(worker.id)
                        const missingItems = getMissingPpe(worker)

                        return (
                          <div className="worker-card" key={worker.id}>
                            <div className="worker-row">
                              <div className="worker-avatar">
                                <Users size={18} />
                              </div>

                              <div className="worker-details">
                                <strong>Worker {worker.id}</strong>
                                <span>
                                  {worker.status === 'COMPLIANT'
                                    ? 'All required PPE detected'
                                    : worker.status === 'REVIEW'
                                      ? `Not detected / verify: ${missingItems.join(', ') || 'No required PPE evidence'}`
                                      : `Explicit PPE violation: ${missingItems.join(', ')}`}
                                </span>
                              </div>

                              <span
                                className={`worker-status ${
                                  worker.status === 'COMPLIANT'
                                    ? 'status-compliant'
                                    : worker.status === 'REVIEW'
                                      ? 'status-review'
                                      : 'status-violation'
                                }`}
                              >
                                {worker.status === 'COMPLIANT'
                                  ? 'COMPLIANT'
                                  : worker.status === 'REVIEW'
                                    ? 'REVIEW'
                                    : 'VIOLATION'}
                              </span>

                              <button
                                type="button"
                                onClick={() => downloadWorkerReport(worker)}
                                aria-label={`Download Worker ${worker.id} PDF report`}
                                style={{
                                  marginLeft: '8px',
                                  padding: '7px 10px',
                                  border: '1px solid rgba(130, 190, 180, 0.22)',
                                  borderRadius: '8px',
                                  background: 'rgba(35, 75, 70, 0.22)',
                                  color: 'inherit',
                                  cursor: 'pointer',
                                  fontSize: '11px',
                                  whiteSpace: 'nowrap',
                                }}
                              >
                                <FileDown size={13} style={{ verticalAlign: 'middle', marginRight: '5px' }} />
                                WORKER PDF
                              </button>

                              <button
                                type="button"
                                onClick={() => toggleWorkerDetails(worker.id)}
                                aria-expanded={isExpanded}
                                aria-label={`${isExpanded ? 'Hide' : 'Show'} Worker ${worker.id} PPE details`}
                                style={{
                                  marginLeft: '10px',
                                  padding: '7px 10px',
                                  border: '1px solid rgba(130, 190, 180, 0.22)',
                                  borderRadius: '8px',
                                  background: 'rgba(35, 75, 70, 0.22)',
                                  color: 'inherit',
                                  cursor: 'pointer',
                                  fontSize: '11px',
                                  whiteSpace: 'nowrap',
                                }}
                              >
                                {isExpanded ? 'HIDE DETAILS −' : 'VIEW DETAILS +'}
                              </button>
                            </div>

                            {isExpanded && (
                              <div
                                className="worker-ppe-details"
                                style={{
                                  display: 'grid',
                                  gridTemplateColumns: 'repeat(auto-fit, minmax(145px, 1fr))',
                                  gap: '9px',
                                  padding: '14px',
                                  borderTop: '1px solid rgba(130, 190, 180, 0.14)',
                                  background: 'rgba(10, 25, 25, 0.28)',
                                }}
                              >
                                {ppeChecks.map((item) => {
                                  const result = !item.required
                                    ? 'NOT REQUIRED'
                                    : worker.ppe[item.key]
                                      ? 'PASS'
                                      : 'FAIL'

                                  const resultColor = result === 'PASS'
                                    ? '#65d6a6'
                                    : result === 'FAIL'
                                      ? '#ff8585'
                                      : '#aab8b5'

                                  return (
                                    <div
                                      key={item.key}
                                      style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'space-between',
                                        gap: '8px',
                                        padding: '10px',
                                        border: '1px solid rgba(130, 190, 180, 0.14)',
                                        borderRadius: '8px',
                                        background: 'rgba(255, 255, 255, 0.025)',
                                      }}
                                    >
                                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
                                        <span style={{ color: '#9ac9c0', display: 'flex' }}>
                                          {item.icon}
                                        </span>
                                        <span style={{ fontSize: '12px', color: '#d5e2de' }}>
                                          {item.label}
                                        </span>
                                      </div>

                                      <strong
                                        style={{
                                          color: resultColor,
                                          fontSize: '10px',
                                          letterSpacing: '0.04em',
                                          whiteSpace: 'nowrap',
                                        }}
                                      >
                                        {result}
                                      </strong>
                                    </div>
                                  )
                                })}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>

                    <div className="results-footer">
                      <span>
                        {compliantCount} COMPLIANT · {violationCount} NEED ATTENTION
                      </span>
                    </div>
                  </>
                )}
              </div>
            </section>

            {/* POST-INSPECTION NAVIGATION */}
            {!analyzing && (inspectionTimestamp || savedInspections.length > 0) && (
              <section
                style={{
                  marginTop: '26px',
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                  gap: '16px',
                }}
              >
                <button
                  type="button"
                  onClick={() => setPostInspectionView('violations')}
                  aria-pressed={postInspectionView === 'violations'}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '16px',
                    padding: '22px',
                    textAlign: 'left',
                    borderRadius: '12px',
                    border:
                      postInspectionView === 'violations'
                        ? '1px solid #83cfc0'
                        : '1px solid rgba(130, 190, 180, 0.2)',
                    background:
                      postInspectionView === 'violations'
                        ? 'rgba(91, 190, 169, 0.12)'
                        : 'rgba(20, 40, 39, 0.7)',
                    color: '#dcebe6',
                    cursor: 'pointer',
                  }}
                >
                  <AlertTriangle size={25} color="#f0b66d" />

                  <span style={{ flex: 1 }}>
                    <strong style={{ display: 'block', fontSize: '15px' }}>
                      Violation Management
                    </strong>
                    <small style={{ color: '#91aaa4', display: 'block', marginTop: '6px' }}>
                      Review detected violations and severity
                    </small>
                  </span>

                  <strong style={{ color: '#f0b66d', fontSize: '22px' }}>
                    {violationCount}
                  </strong>
                </button>

                <button
                  type="button"
                  onClick={() => setPostInspectionView('analytics')}
                  aria-pressed={postInspectionView === 'analytics'}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '16px',
                    padding: '22px',
                    textAlign: 'left',
                    borderRadius: '12px',
                    border:
                      postInspectionView === 'analytics'
                        ? '1px solid #83cfc0'
                        : '1px solid rgba(130, 190, 180, 0.2)',
                    background:
                      postInspectionView === 'analytics'
                        ? 'rgba(91, 190, 169, 0.12)'
                        : 'rgba(20, 40, 39, 0.7)',
                    color: '#dcebe6',
                    cursor: 'pointer',
                  }}
                >
                  <ScanSearch size={25} color="#65cbb5" />

                  <span style={{ flex: 1 }}>
                    <strong style={{ display: 'block', fontSize: '15px' }}>
                      Analytics Dashboard
                    </strong>
                    <small style={{ color: '#91aaa4', display: 'block', marginTop: '6px' }}>
                      Explore compliance rates and PPE statistics
                    </small>
                  </span>

                  <strong style={{ color: '#65cbb5', fontSize: '22px' }}>
                    {complianceRate}%
                  </strong>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setPostInspectionView('history')
                    void refreshInspectionHistory()
                  }}
                  aria-pressed={postInspectionView === 'history'}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '16px', padding: '22px',
                    textAlign: 'left', borderRadius: '12px',
                    border: postInspectionView === 'history' ? '1px solid #83cfc0' : '1px solid rgba(130, 190, 180, 0.2)',
                    background: postInspectionView === 'history' ? 'rgba(91, 190, 169, 0.12)' : 'rgba(20, 40, 39, 0.7)',
                    color: '#dcebe6', cursor: 'pointer',
                  }}
                >
                  <History size={25} color="#65cbb5" />
                  <span style={{ flex: 1 }}>
                    <strong style={{ display: 'block', fontSize: '15px' }}>Inspection History</strong>
                    <small style={{ color: '#91aaa4', display: 'block', marginTop: '6px' }}>Reopen inspections and download PDF reports</small>
                  </span>
                  <strong style={{ color: '#65cbb5', fontSize: '22px' }}>{savedInspections.length}</strong>
                </button>
              </section>
            )}

            {/* VIOLATION MANAGEMENT */}
            {postInspectionView === 'violations' && (
              <section
                className="dashboard-panel"
                style={{ marginTop: '24px', padding: '24px' }}
              >
                <div className="panel-heading" style={{ marginBottom: '22px' }}>
                  <div>
                    <span className="panel-index">04 / MONITORING</span>
                    <h2>Violation Management</h2>
                  </div>
                  <span className="results-live">
                    {violations.length} ACTIVE VIOLATIONS
                  </span>
                </div>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                    gap: '12px',
                    marginBottom: '22px',
                  }}
                >
                  {([
                    { key: 'CRITICAL', label: 'CRITICAL', color: '#ff8585' },
                    { key: 'HIGH', label: 'HIGH', color: '#f0b66d' },
                    { key: 'MEDIUM', label: 'MEDIUM', color: '#e8d47c' },
                  ] as const).map((item) => (
                    <div
                      key={item.key}
                      style={{
                        padding: '16px',
                        border: '1px solid rgba(130, 190, 180, 0.16)',
                        borderRadius: '10px',
                        background: 'rgba(255, 255, 255, 0.025)',
                      }}
                    >
                      <span
                        style={{
                          display: 'block',
                          color: '#91aaa4',
                          fontSize: '10px',
                          letterSpacing: '0.12em',
                          marginBottom: '8px',
                        }}
                      >
                        {item.label}
                      </span>
                      <strong style={{ color: item.color, fontSize: '26px' }}>
                        {severityCounts[item.key]}
                      </strong>
                    </div>
                  ))}
                </div>

                {/* VIOLATION FILTERS */}
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '8px',
                    marginBottom: '18px',
                  }}
                >
                  {(['ALL', 'CRITICAL', 'HIGH', 'MEDIUM'] as ViolationFilter[]).map(
                    (filter) => (
                      <button
                        key={filter}
                        type="button"
                        onClick={() => setViolationFilter(filter)}
                        aria-pressed={violationFilter === filter}
                        style={{
                          padding: '8px 13px',
                          borderRadius: '6px',
                          border:
                            violationFilter === filter
                              ? '1px solid #83cfc0'
                              : '1px solid rgba(130, 190, 180, 0.2)',
                          background:
                            violationFilter === filter
                              ? 'rgba(91, 190, 169, 0.15)'
                              : 'transparent',
                          color:
                            violationFilter === filter ? '#b9eee2' : '#9bb2ac',
                          fontSize: '10px',
                          letterSpacing: '0.08em',
                          cursor: 'pointer',
                        }}
                      >
                        {filter}
                      </button>
                    )
                  )}
                </div>

                {/* VIOLATION RECORDS */}
                {!inspectionTimestamp ? (
                  <div
                    style={{
                      padding: '28px 12px',
                      textAlign: 'center',
                      border: '1px dashed rgba(130, 190, 180, 0.2)',
                      borderRadius: '8px',
                      color: '#91aaa4',
                      fontSize: '12px',
                    }}
                  >
                    Complete an inspection to view detected violations.
                  </div>
                ) : filteredViolations.length === 0 ? (
                  <div
                    style={{
                      padding: '28px 12px',
                      textAlign: 'center',
                      border: '1px dashed rgba(130, 190, 180, 0.2)',
                      borderRadius: '8px',
                      color: '#91aaa4',
                      fontSize: '12px',
                    }}
                  >
                    {violations.length === 0
                      ? 'No PPE violations detected in this inspection.'
                      : 'No violations match this severity filter.'}
                  </div>
                ) : (
                  <div style={{ display: 'grid', gap: '10px' }}>
                    {filteredViolations.map((violation) => {
                      const severityColor =
                        violation.severity === 'CRITICAL'
                          ? '#ff8585'
                          : violation.severity === 'HIGH'
                            ? '#f0b66d'
                            : '#e8d47c'

                      return (
                        <div
                          key={violation.workerId}
                          style={{
                            display: 'flex',
                            flexWrap: 'wrap',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: '14px',
                            padding: '15px',
                            border: '1px solid rgba(130, 190, 180, 0.15)',
                            borderRadius: '8px',
                            background: 'rgba(255, 255, 255, 0.02)',
                          }}
                        >
                          <div style={{ minWidth: '180px', flex: '1 1 220px' }}>
                            <strong
                              style={{
                                display: 'block',
                                color: '#dcebe6',
                                fontSize: '13px',
                                marginBottom: '7px',
                              }}
                            >
                              Worker {violation.workerId}
                            </strong>

                            <span style={{ color: '#9bb2ac', fontSize: '11px' }}>
                              Missing: {violation.missingItems.join(', ')}
                            </span>

                            <span
                              style={{
                                display: 'block',
                                color: '#718d85',
                                fontSize: '10px',
                                marginTop: '7px',
                              }}
                            >
                              {formatViolationTime(violation.timestamp)}
                            </span>
                          </div>

                          <span
                            style={{
                              color: severityColor,
                              border: `1px solid ${severityColor}55`,
                              background: `${severityColor}12`,
                              borderRadius: '5px',
                              padding: '7px 10px',
                              fontSize: '10px',
                              letterSpacing: '0.1em',
                              fontWeight: 700,
                            }}
                          >
                            {violation.severity}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                )}

                <div
                  style={{
                    marginTop: '18px',
                    color: '#718d85',
                    fontSize: '9px',
                    letterSpacing: '0.1em',
                  }}
                >
                  MODEL OUTPUT · CONFIRMED NEGATIVE PPE LABELS
                </div>
              </section>
            )}

            {/* ANALYTICS DASHBOARD */}
            {postInspectionView === 'analytics' && (
              <section
                className="dashboard-panel"
                style={{
                  marginTop: '24px',
                  padding: '24px',
                }}
              >
                <div className="panel-heading" style={{ marginBottom: '24px' }}>
                  <div>
                    <span className="panel-index">05 / ANALYTICS</span>
                    <h2>Analytics Dashboard</h2>
                  </div>

                  <span className="results-live">
                    INSPECTION ANALYTICS
                  </span>
                </div>

                {/* OVERVIEW CARDS */}
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
                    gap: '14px',
                    marginBottom: '28px',
                  }}
                >
                  {[
                    {
                      label: 'TOTAL WORKERS',
                      value: workers.length,
                      color: '#dcebe6',
                    },
                    {
                      label: 'COMPLIANT WORKERS',
                      value: compliantCount,
                      color: '#65d6a6',
                    },
                    {
                      label: 'NON-COMPLIANT',
                      value: violationCount,
                      color: '#ff8585',
                    },
                    {
                      label: 'COMPLIANCE RATE',
                      value: `${complianceRate}%`,
                      color: '#65cbb5',
                    },
                  ].map((item) => (
                    <div
                      key={item.label}
                      style={{
                        padding: '18px',
                        border: '1px solid rgba(130, 190, 180, 0.16)',
                        borderRadius: '10px',
                        background: 'rgba(255, 255, 255, 0.025)',
                      }}
                    >
                      <span
                        style={{
                          display: 'block',
                          color: '#91aaa4',
                          fontSize: '10px',
                          letterSpacing: '0.1em',
                          marginBottom: '12px',
                        }}
                      >
                        {item.label}
                      </span>

                      <strong
                        style={{
                          color: item.color,
                          fontSize: '28px',
                        }}
                      >
                        {item.value}
                      </strong>
                    </div>
                  ))}
                </div>

                {/* WORKER COMPLIANCE BREAKDOWN */}
                <div style={{ marginBottom: '30px' }}>
                  <h3
                    style={{
                      color: '#dcebe6',
                      fontSize: '15px',
                      marginBottom: '16px',
                    }}
                  >
                    Worker Compliance Breakdown
                  </h3>

                  {[
                    {
                      label: 'Compliant',
                      count: compliantCount,
                      color: '#65d6a6',
                    },
                    {
                      label: 'Non-compliant',
                      count: violationCount,
                      color: '#ff8585',
                    },
                  ].map((item) => {
                    const percentage =
                      workers.length > 0
                        ? (item.count / workers.length) * 100
                        : 0

                    return (
                      <div key={item.label} style={{ marginBottom: '16px' }}>
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            marginBottom: '8px',
                            color: '#b9ccc5',
                            fontSize: '12px',
                          }}
                        >
                          <span>{item.label}</span>
                          <strong>
                            {item.count} workers ({Math.round(percentage)}%)
                          </strong>
                        </div>

                        <div
                          style={{
                            height: '9px',
                            borderRadius: '20px',
                            background: 'rgba(255,255,255,0.07)',
                            overflow: 'hidden',
                          }}
                        >
                          <div
                            style={{
                              width: `${percentage}%`,
                              height: '100%',
                              background: item.color,
                              borderRadius: '20px',
                              transition: 'width 0.4s ease',
                            }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* PPE-WISE COMPLIANCE */}
                <div style={{ marginBottom: '30px' }}>
                  <h3
                    style={{
                      color: '#dcebe6',
                      fontSize: '15px',
                      marginBottom: '16px',
                    }}
                  >
                    PPE-wise Compliance
                  </h3>

                  <div style={{ display: 'grid', gap: '16px' }}>
                    {ppeAnalytics.map((item) => (
                      <div key={item.label}>
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            gap: '10px',
                            marginBottom: '8px',
                          }}
                        >
                          <span style={{ color: '#b9ccc5', fontSize: '12px' }}>
                            {item.label}
                          </span>

                          {item.required ? (
                            <strong style={{ color: '#65cbb5', fontSize: '12px' }}>
                              {item.passed}/{workers.length} · {item.rate}%
                            </strong>
                          ) : (
                            <span style={{ color: '#91aaa4', fontSize: '10px' }}>
                              NOT REQUIRED
                            </span>
                          )}
                        </div>

                        <div
                          style={{
                            height: '8px',
                            borderRadius: '20px',
                            background: 'rgba(255,255,255,0.07)',
                            overflow: 'hidden',
                          }}
                        >
                          <div
                            style={{
                              width: `${item.rate}%`,
                              height: '100%',
                              background: '#65cbb5',
                              borderRadius: '20px',
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* VIOLATION SEVERITY DISTRIBUTION */}
                <div>
                  <h3
                    style={{
                      color: '#dcebe6',
                      fontSize: '15px',
                      marginBottom: '16px',
                    }}
                  >
                    Violation Severity Distribution
                  </h3>

                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                      gap: '12px',
                    }}
                  >
                    {[
                      {
                        label: 'CRITICAL',
                        count: severityCounts.CRITICAL,
                        color: '#ff8585',
                      },
                      {
                        label: 'HIGH',
                        count: severityCounts.HIGH,
                        color: '#f0b66d',
                      },
                      {
                        label: 'MEDIUM',
                        count: severityCounts.MEDIUM,
                        color: '#e8d47c',
                      },
                    ].map((item) => (
                      <div
                        key={item.label}
                        style={{
                          padding: '16px',
                          border: '1px solid rgba(130, 190, 180, 0.16)',
                          borderRadius: '9px',
                          background: 'rgba(255,255,255,0.025)',
                        }}
                      >
                        <span
                          style={{
                            display: 'block',
                            color: '#91aaa4',
                            fontSize: '10px',
                            letterSpacing: '0.1em',
                            marginBottom: '10px',
                          }}
                        >
                          {item.label}
                        </span>

                        <strong style={{ color: item.color, fontSize: '25px' }}>
                          {item.count}
                        </strong>
                      </div>
                    ))}
                  </div>
                </div>

                <div
                  style={{
                    marginTop: '22px',
                    color: '#718d85',
                    fontSize: '9px',
                    letterSpacing: '0.1em',
                  }}
                >
                  MODEL ANALYTICS · CURRENT INSPECTION
                </div>
              </section>
            )}

            {/* INSPECTION HISTORY */}
            {postInspectionView === 'history' && (
              <section className="dashboard-panel" style={{ marginTop: '24px', padding: '24px' }}>
                <div className="panel-heading" style={{ marginBottom: '22px' }}>
                  <div><span className="panel-index">06 / RECORDS</span><h2>Inspection History</h2></div>
                  <span className="results-live">{savedInspections.length} SAVED RECORDS</span>
                </div>

                <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '20px' }}>
                  <p style={{ color: '#91aaa4', fontSize: '12px', lineHeight: 1.7, margin: 0 }}>Saved locally in this browser. Records are not synced between devices.</p>
                  <button type="button" className="camera-action-button" onClick={() => void refreshInspectionHistory()} disabled={historyLoading}>
                    {historyLoading ? <LoaderCircle size={15} /> : <History size={15} />} REFRESH
                  </button>
                </div>

                {historyError && <div style={{ padding: '14px', marginBottom: '16px', border: '1px solid rgba(255, 133, 133, 0.35)', borderRadius: '8px', color: '#ff9999', fontSize: '12px' }}>{historyError}</div>}

                {historyLoading ? (
                  <div style={{ padding: '35px 12px', textAlign: 'center', color: '#91aaa4' }}>Loading saved inspections...</div>
                ) : savedInspections.length === 0 ? (
                  <div style={{ padding: '40px 15px', textAlign: 'center', border: '1px dashed rgba(130, 190, 180, 0.2)', borderRadius: '10px', color: '#91aaa4' }}>
                    <History size={30} style={{ marginBottom: '12px', opacity: 0.65 }} />
                    <h3 style={{ color: '#dcebe6', fontSize: '15px', marginBottom: '8px' }}>No inspection records yet</h3>
                    <p style={{ fontSize: '12px' }}>Complete an inspection to create your first saved record.</p>
                  </div>
                ) : (
                  <div style={{ display: 'grid', gap: '12px' }}>
                    {savedInspections.map((record) => {
                      const total = record.workers.length
                      const compliant = record.workers.filter((worker) => worker.status === 'COMPLIANT').length
                      const violationTotal = total - compliant
                      const rate = total ? Math.round((compliant / total) * 100) : 0
                      return (
                        <article key={record.id} style={{ padding: '18px', border: selectedHistoryId === record.id ? '1px solid #83cfc0' : '1px solid rgba(130, 190, 180, 0.16)', borderRadius: '10px', background: selectedHistoryId === record.id ? 'rgba(91, 190, 169, 0.08)' : 'rgba(255,255,255,0.02)' }}>
                          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start', justifyContent: 'space-between', gap: '14px' }}>
                            <div style={{ flex: '1 1 220px' }}>
                              <strong style={{ display: 'block', color: '#dcebe6', fontSize: '14px', marginBottom: '7px' }}>{record.sourceName}</strong>
                              <span style={{ color: '#91aaa4', fontSize: '11px' }}>{formatViolationTime(record.timestamp)}</span>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '15px', marginTop: '13px', fontSize: '11px' }}>
                                <span style={{ color: '#dcebe6' }}>{total} workers</span>
                                <span style={{ color: '#65d6a6' }}>{compliant} compliant</span>
                                <span style={{ color: '#ff8585' }}>{violationTotal} violations</span>
                                <strong style={{ color: '#65cbb5' }}>{rate}% compliance</strong>
                              </div>
                              <span style={{ display: 'inline-block', marginTop: '12px', color: '#718d85', fontSize: '9px', letterSpacing: '0.1em' }}>{record.inputMode.toUpperCase()} MODE</span>
                            </div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                              <button type="button" className="camera-action-button" onClick={() => openSavedInspection(record)}><History size={14} /> OPEN</button>
                              <button type="button" className="camera-action-button" onClick={() => downloadInspectionReport(record)}><FileDown size={14} /> PDF</button>
                              <button type="button" className="camera-action-button" onClick={() => void handleDeleteInspection(record.id)} aria-label={`Delete inspection ${record.sourceName}`} style={{ color: '#ff9999' }}><Trash2 size={14} /></button>
                            </div>
                          </div>
                        </article>
                      )
                    })}
                  </div>
                )}
                <div style={{ marginTop: '20px', color: '#718d85', fontSize: '9px', letterSpacing: '0.1em', lineHeight: 1.8 }}>LOCAL BROWSER HISTORY · NO CLOUD BACKUP</div>
              </section>
            )}

            <footer className="dashboard-footer">
              <span>NETRA / INDUSTRIAL SAFETY INTELLIGENCE</span>
              <span>POWERED BY COMPUTER VISION</span>
            </footer>
          </div>
        </main>
      )}
    </>
  )
}

export default App