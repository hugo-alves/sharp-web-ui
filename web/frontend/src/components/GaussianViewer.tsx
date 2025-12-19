import { useEffect, useRef, useState, useCallback } from 'react';
import * as SPLAT from 'gsplat';
import { getSplatUrl } from '../api/client';

interface GaussianViewerProps {
  jobId: string;
  className?: string;
}

// Detect if we're on a mobile device
const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);

/**
 * Custom camera controls following antimatter15/splat conventions:
 * - Horizontal drag: direct (drag right = look right)
 * - Vertical drag: inverted (drag up = look up, drag down = look down)
 * - Two-finger: pinch to zoom, drag to pan, rotate for roll
 * - Three-finger: roll/tilt
 */
class NaturalCameraControls {
  private camera: any; // Camera type doesn't expose position/rotation in typings
  private canvas: HTMLElement;

  // Camera state (spherical coordinates around target)
  private alpha = 0;      // horizontal angle (yaw)
  private beta = 0.5;     // vertical angle (pitch), start slightly above
  private radius = 5;     // distance from target
  private target = { x: 0, y: 0, z: 0 };

  // Interaction state
  private isDragging = false;
  private isPanning = false;
  private lastX = 0;
  private lastY = 0;
  private lastPinchDist = 0;
  private lastPinchAngle = 0;
  private rollAngle = 0;

  // Settings
  private orbitSpeed = 0.005;
  private panSpeed = 0.01;
  private zoomSpeed = 0.001;
  private minRadius = 0.5;
  private maxRadius = 50;
  private dampening = 0.1;

  // Target values for smooth interpolation
  private targetAlpha = 0;
  private targetBeta = 0.5;
  private targetRadius = 5;
  private targetRoll = 0;

  constructor(camera: SPLAT.Camera, canvas: HTMLElement) {
    this.camera = camera;
    this.canvas = canvas;

    this.setupEventListeners();
    this.updateCamera();
  }

  private setupEventListeners() {
    // Mouse events
    this.canvas.addEventListener('mousedown', this.onMouseDown);
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('mousemove', this.onMouseMove);
    window.addEventListener('mouseup', this.onMouseUp);
    this.canvas.addEventListener('wheel', this.onWheel, { passive: false });

    // Touch events
    this.canvas.addEventListener('touchstart', this.onTouchStart, { passive: false });
    this.canvas.addEventListener('touchmove', this.onTouchMove, { passive: false });
    this.canvas.addEventListener('touchend', this.onTouchEnd);
  }

  private onMouseDown = (e: MouseEvent) => {
    e.preventDefault();
    this.isDragging = true;
    this.isPanning = e.button === 2 || e.shiftKey; // Right-click or shift for pan
    this.lastX = e.clientX;
    this.lastY = e.clientY;
  };

  private onMouseMove = (e: MouseEvent) => {
    if (!this.isDragging) return;

    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;

    if (this.isPanning) {
      this.pan(dx, dy);
    } else {
      this.orbit(dx, dy);
    }

    this.lastX = e.clientX;
    this.lastY = e.clientY;
  };

  private onMouseUp = () => {
    this.isDragging = false;
    this.isPanning = false;
  };

  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    this.zoom(e.deltaY);
  };

  private onTouchStart = (e: TouchEvent) => {
    e.preventDefault();

    if (e.touches.length === 1) {
      // Single finger - orbit
      this.isDragging = true;
      this.isPanning = false;
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      // Two fingers - pinch zoom + pan + roll
      this.isDragging = true;
      this.isPanning = true;
      const [t1, t2] = [e.touches[0], e.touches[1]];
      this.lastX = (t1.clientX + t2.clientX) / 2;
      this.lastY = (t1.clientY + t2.clientY) / 2;
      this.lastPinchDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
      this.lastPinchAngle = Math.atan2(t2.clientY - t1.clientY, t2.clientX - t1.clientX);
    } else if (e.touches.length === 3) {
      // Three fingers - roll only
      this.isDragging = true;
      const touches = Array.from(e.touches);
      const cx = touches.reduce((s, t) => s + t.clientX, 0) / 3;
      const cy = touches.reduce((s, t) => s + t.clientY, 0) / 3;
      this.lastPinchAngle = Math.atan2(touches[0].clientY - cy, touches[0].clientX - cx);
    }
  };

  private onTouchMove = (e: TouchEvent) => {
    if (!this.isDragging) return;
    e.preventDefault();

    if (e.touches.length === 1) {
      // Single finger orbit
      const dx = e.touches[0].clientX - this.lastX;
      const dy = e.touches[0].clientY - this.lastY;
      this.orbit(dx, dy);
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      const [t1, t2] = [e.touches[0], e.touches[1]];
      const centerX = (t1.clientX + t2.clientX) / 2;
      const centerY = (t1.clientY + t2.clientY) / 2;
      const pinchDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
      const pinchAngle = Math.atan2(t2.clientY - t1.clientY, t2.clientX - t1.clientX);

      // Pinch to zoom
      const zoomDelta = (this.lastPinchDist - pinchDist) * 5;
      this.zoom(zoomDelta);

      // Two-finger drag to pan
      const dx = centerX - this.lastX;
      const dy = centerY - this.lastY;
      this.pan(dx, dy);

      // Two-finger rotation for roll
      let deltaAngle = pinchAngle - this.lastPinchAngle;
      while (deltaAngle > Math.PI) deltaAngle -= 2 * Math.PI;
      while (deltaAngle < -Math.PI) deltaAngle += 2 * Math.PI;
      this.targetRoll += deltaAngle;

      this.lastX = centerX;
      this.lastY = centerY;
      this.lastPinchDist = pinchDist;
      this.lastPinchAngle = pinchAngle;
    } else if (e.touches.length === 3) {
      // Three-finger roll
      const touches = Array.from(e.touches);
      const cx = touches.reduce((s, t) => s + t.clientX, 0) / 3;
      const cy = touches.reduce((s, t) => s + t.clientY, 0) / 3;
      const currentAngle = Math.atan2(touches[0].clientY - cy, touches[0].clientX - cx);

      let deltaAngle = currentAngle - this.lastPinchAngle;
      while (deltaAngle > Math.PI) deltaAngle -= 2 * Math.PI;
      while (deltaAngle < -Math.PI) deltaAngle += 2 * Math.PI;
      this.targetRoll += deltaAngle;

      this.lastPinchAngle = currentAngle;
    }
  };

  private onTouchEnd = (e: TouchEvent) => {
    if (e.touches.length === 0) {
      this.isDragging = false;
      this.isPanning = false;
    } else if (e.touches.length === 1) {
      // Switched from multi-touch to single touch
      this.isPanning = false;
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      const [t1, t2] = [e.touches[0], e.touches[1]];
      this.lastX = (t1.clientX + t2.clientX) / 2;
      this.lastY = (t1.clientY + t2.clientY) / 2;
      this.lastPinchDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
      this.lastPinchAngle = Math.atan2(t2.clientY - t1.clientY, t2.clientX - t1.clientX);
    }
  };

  private orbit(dx: number, dy: number) {
    // Horizontal: drag right = look right (direct)
    this.targetAlpha += dx * this.orbitSpeed;

    // Vertical: drag up = look up (inverted Y - this is the key!)
    // Negative because screen Y increases downward
    this.targetBeta -= dy * this.orbitSpeed;

    // Clamp vertical angle to avoid gimbal lock
    this.targetBeta = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.targetBeta));
  }

  private pan(dx: number, dy: number) {
    const speed = this.panSpeed * this.radius;

    // Calculate right and up vectors from camera orientation
    const cosA = Math.cos(this.alpha);
    const sinA = Math.sin(this.alpha);

    // Right vector (perpendicular to view direction in XZ plane)
    const rightX = cosA;
    const rightZ = sinA;

    // Up vector (always Y-up for simplicity)
    // Move target opposite to drag direction for natural feel
    this.target.x -= dx * speed * rightX;
    this.target.z -= dx * speed * rightZ;
    this.target.y += dy * speed;
  }

  private zoom(delta: number) {
    this.targetRadius += delta * this.zoomSpeed * this.radius;
    this.targetRadius = Math.max(this.minRadius, Math.min(this.maxRadius, this.targetRadius));
  }

  update() {
    // Smooth interpolation
    this.alpha += (this.targetAlpha - this.alpha) * this.dampening;
    this.beta += (this.targetBeta - this.beta) * this.dampening;
    this.radius += (this.targetRadius - this.radius) * this.dampening;
    this.rollAngle += (this.targetRoll - this.rollAngle) * this.dampening;

    this.updateCamera();
  }

  private updateCamera() {
    // Convert spherical to Cartesian
    const cosBeta = Math.cos(this.beta);
    const x = this.target.x + this.radius * cosBeta * Math.sin(this.alpha);
    const y = this.target.y + this.radius * Math.sin(this.beta);
    const z = this.target.z + this.radius * cosBeta * Math.cos(this.alpha);

    // Set camera position
    this.camera.position.x = x;
    this.camera.position.y = y;
    this.camera.position.z = z;

    // Calculate look-at rotation with roll
    // First, get the direction to target
    const dx = this.target.x - x;
    const dy = this.target.y - y;
    const dz = this.target.z - z;

    // Calculate pitch and yaw from direction
    const pitch = Math.atan2(dy, Math.sqrt(dx * dx + dz * dz));
    const yaw = Math.atan2(dx, dz);

    // Create rotation quaternion (yaw, pitch, roll)
    // Using ZYX order: roll around Z, then pitch around X, then yaw around Y
    const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2);
    const cp = Math.cos(-pitch / 2), sp = Math.sin(-pitch / 2);
    const cr = Math.cos(this.rollAngle / 2), sr = Math.sin(this.rollAngle / 2);

    const qw = cr * cp * cy + sr * sp * sy;
    const qx = sr * cp * cy - cr * sp * sy;
    const qy = cr * sp * cy + sr * cp * sy;
    const qz = cr * cp * sy - sr * sp * cy;

    this.camera.rotation.x = qx;
    this.camera.rotation.y = qy;
    this.camera.rotation.z = qz;
    this.camera.rotation.w = qw;
  }

  dispose() {
    this.canvas.removeEventListener('mousedown', this.onMouseDown);
    window.removeEventListener('mousemove', this.onMouseMove);
    window.removeEventListener('mouseup', this.onMouseUp);
    this.canvas.removeEventListener('wheel', this.onWheel);
    this.canvas.removeEventListener('touchstart', this.onTouchStart);
    this.canvas.removeEventListener('touchmove', this.onTouchMove);
    this.canvas.removeEventListener('touchend', this.onTouchEnd);
  }
}

export function GaussianViewer({ jobId, className = '' }: GaussianViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<SPLAT.WebGLRenderer | null>(null);
  const sceneRef = useRef<SPLAT.Scene | null>(null);
  const cameraRef = useRef<SPLAT.Camera | null>(null);
  const controlsRef = useRef<NaturalCameraControls | null>(null);
  const animationIdRef = useRef<number | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [splatCount, setSplatCount] = useState<number | null>(null);

  // For iOS pseudo-fullscreen (since iOS Safari doesn't support Fullscreen API)
  const [isPseudoFullscreen, setIsPseudoFullscreen] = useState(false);

  // Handle fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement !== null);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
    };
  }, []);

  // Lock body scroll when in pseudo-fullscreen mode (for iOS)
  useEffect(() => {
    if (isPseudoFullscreen) {
      document.body.style.overflow = 'hidden';
      document.body.style.position = 'fixed';
      document.body.style.width = '100%';
      document.body.style.height = '100%';
    } else {
      document.body.style.overflow = '';
      document.body.style.position = '';
      document.body.style.width = '';
      document.body.style.height = '';
    }
    return () => {
      document.body.style.overflow = '';
      document.body.style.position = '';
      document.body.style.width = '';
      document.body.style.height = '';
    };
  }, [isPseudoFullscreen]);

  const toggleFullscreen = useCallback(async () => {
    if (!containerRef.current) return;

    // For iOS, use pseudo-fullscreen since Safari doesn't support Fullscreen API
    if (isIOS) {
      setIsPseudoFullscreen(prev => !prev);
      // Trigger resize after state change
      setTimeout(() => {
        if (rendererRef.current && containerRef.current) {
          rendererRef.current.setSize(
            containerRef.current.clientWidth,
            containerRef.current.clientHeight
          );
        }
      }, 50);
      return;
    }

    try {
      const elem = containerRef.current as any;
      if (!document.fullscreenElement && !(document as any).webkitFullscreenElement) {
        if (elem.requestFullscreen) {
          await elem.requestFullscreen();
        } else if (elem.webkitRequestFullscreen) {
          await elem.webkitRequestFullscreen();
        }
      } else {
        if (document.exitFullscreen) {
          await document.exitFullscreen();
        } else if ((document as any).webkitExitFullscreen) {
          await (document as any).webkitExitFullscreen();
        }
      }
    } catch (err) {
      // Fallback to pseudo-fullscreen if native fails
      console.error('Fullscreen error, using fallback:', err);
      setIsPseudoFullscreen(prev => !prev);
    }
  }, []);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'f' || e.key === 'F') {
        toggleFullscreen();
      }
      // Handle Escape for pseudo-fullscreen mode
      if (e.key === 'Escape' && isPseudoFullscreen) {
        setIsPseudoFullscreen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toggleFullscreen, isPseudoFullscreen]);

  // Initialize and load splat
  useEffect(() => {
    if (!canvasRef.current) return;

    let disposed = false;

    async function initViewer() {
      if (!canvasRef.current || disposed) return;

      setLoading(true);
      setError(null);

      try {
        // Clean up previous resources
        if (animationIdRef.current) {
          cancelAnimationFrame(animationIdRef.current);
          animationIdRef.current = null;
        }
        if (rendererRef.current) {
          rendererRef.current.dispose();
          rendererRef.current = null;
        }

        console.log('Creating gsplat.js viewer...');

        // Create scene, camera, renderer
        const scene = new SPLAT.Scene();
        const camera = new SPLAT.Camera();
        const renderer = new SPLAT.WebGLRenderer(canvasRef.current);

        // Use custom controls with natural feel (following antimatter15/splat conventions)
        const controls = new NaturalCameraControls(camera, canvasRef.current);

        sceneRef.current = scene;
        cameraRef.current = camera;
        rendererRef.current = renderer;
        controlsRef.current = controls;

        // Load the .splat file (converted from SHARP PLY format)
        const splatUrl = getSplatUrl(jobId);
        console.log('Loading splat from:', splatUrl);

        // Fetch and load the splat using SPLAT.Loader (for .splat format)
        await SPLAT.Loader.LoadAsync(splatUrl, scene, (progress) => {
          console.log('Loading progress:', Math.round(progress * 100) + '%');
        });

        if (disposed) {
          renderer.dispose();
          return;
        }

        // Get splat count from scene
        const splatData = scene.objects[0];
        if (splatData && 'splatCount' in splatData) {
          setSplatCount((splatData as any).splatCount);
        }

        console.log('Loaded splat successfully');

        // Animation loop
        const frame = () => {
          if (disposed) return;
          controls.update();
          renderer.render(scene, camera);
          animationIdRef.current = requestAnimationFrame(frame);
        };

        animationIdRef.current = requestAnimationFrame(frame);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load Gaussian splat:', err);
        if (!disposed) {
          setError(`Failed to load 3D preview: ${err instanceof Error ? err.message : 'Unknown error'}`);
          setLoading(false);
        }
      }
    }

    initViewer();

    return () => {
      disposed = true;
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
        animationIdRef.current = null;
      }
      if (controlsRef.current) {
        controlsRef.current.dispose();
        controlsRef.current = null;
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
        rendererRef.current = null;
      }
      sceneRef.current = null;
      cameraRef.current = null;
    };
  }, [jobId]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (rendererRef.current && containerRef.current) {
        rendererRef.current.setSize(
          containerRef.current.clientWidth,
          containerRef.current.clientHeight
        );
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  if (error) {
    return (
      <div className={`flex items-center justify-center bg-gray-900 ${className}`}>
        <div className="text-center p-4">
          <p className="text-red-400 mb-2">{error}</p>
          <p className="text-gray-500 text-sm">Try downloading the PLY file and viewing in an external viewer</p>
        </div>
      </div>
    );
  }

  // Combined fullscreen state (native or pseudo)
  const isInFullscreen = isFullscreen || isPseudoFullscreen;

  return (
    <div
      ref={containerRef}
      className={`relative ${isInFullscreen ? 'fixed inset-0 w-screen h-screen' : className}`}
      style={{
        backgroundColor: '#111827',
        minHeight: isInFullscreen ? '100vh' : '500px',
        zIndex: isInFullscreen ? 9999 : 'auto',
      }}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ zIndex: 1 }}
      />

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10">
          <div className="text-center">
            <div className="animate-spin h-8 w-8 border-2 border-white border-t-transparent rounded-full mx-auto mb-2" />
            <p className="text-gray-400 text-sm">Loading Gaussian Splats...</p>
          </div>
        </div>
      )}

      {/* Fullscreen button */}
      <button
        onClick={toggleFullscreen}
        className="absolute top-3 right-3 bg-black/70 hover:bg-black/90 text-white p-2.5 rounded-lg transition-colors z-20 touch-manipulation"
        title={isInFullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen (F)'}
        style={{ minWidth: '44px', minHeight: '44px' }} // Better touch target for mobile
      >
        {isInFullscreen ? (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        )}
      </button>

      {/* Stats */}
      {splatCount !== null && !loading && (
        <div className="absolute bottom-3 left-3 bg-black/60 text-white text-xs px-2 py-1 rounded z-20">
          {splatCount.toLocaleString()} gaussians
        </div>
      )}

      {/* Controls help - different for mobile vs desktop */}
      {!loading && !isMobile && (
        <div className="absolute bottom-3 right-3 bg-black/60 text-white text-xs px-2 py-1 rounded space-y-0.5 z-20">
          <div className="font-medium text-gray-300 mb-1">Controls</div>
          <div>Drag: Orbit</div>
          <div>Right-drag/Shift: Pan</div>
          <div>Scroll: Zoom</div>
          <div className="border-t border-white/20 mt-1 pt-1">F: Fullscreen</div>
        </div>
      )}

      {/* Mobile controls help - simplified */}
      {!loading && isMobile && !isInFullscreen && (
        <div className="absolute bottom-3 right-3 bg-black/60 text-white text-xs px-2 py-1 rounded z-20">
          <div>1 finger: Orbit</div>
          <div>2 fingers: Zoom + Pan + Roll</div>
          <div>3 fingers: Roll only</div>
        </div>
      )}
    </div>
  );
}
