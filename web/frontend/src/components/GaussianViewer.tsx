import { useEffect, useRef, useState, useCallback } from 'react';
import * as SPLAT from 'gsplat';
import { getSplatUrl, getCameraMetadata } from '../api/client';

interface GaussianViewerProps {
  jobId: string;
  className?: string;
}

// Detect if we're on a mobile device
const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);

/**
 * Simplified camera controls: Orbit + Zoom only
 * - Drag: Orbit around the scene
 * - Scroll/Pinch: Zoom in/out
 */
class NaturalCameraControls {
  private camera: any;
  private canvas: HTMLElement;

  // Camera state (spherical coordinates around target)
  private alpha = Math.PI;   // horizontal angle (yaw) - PI means facing +Z
  private beta = 0;          // vertical angle (pitch), level
  private radius = 2.5;      // distance from target
  private target = { x: 0, y: 0, z: 2.5 };  // Fixed target point

  // Interaction state
  private isDragging = false;
  private lastX = 0;
  private lastY = 0;
  private lastPinchDist = 0;
  private isShiftHeld = false;

  // Settings - reduced sensitivity for smoother control
  private orbitSpeed = 0.002;
  private zoomSpeed = 0.0005;
  private panSpeed = 0.003;
  private minRadius = 0.5;
  private maxRadius = 50;
  private dampening = 0.08;

  // Target values for smooth interpolation
  private targetAlpha = Math.PI;
  private targetBeta = 0;
  private targetRadius = 2.5;
  private targetTarget = { x: 0, y: 0, z: 2.5 };

  constructor(camera: SPLAT.Camera, canvas: HTMLElement) {
    this.camera = camera;
    this.canvas = canvas;

    this.setupEventListeners();
    this.updateCamera();
  }

  // Store initial depth for reset
  private initialDepth = 2.5;

  /**
   * Set camera to match the original input image perspective.
   */
  setInputImageView(targetDepth: number = 2.5) {
    this.initialDepth = targetDepth;
    this.alpha = Math.PI;
    this.beta = 0;
    this.radius = targetDepth;
    this.target = { x: 0, y: 0, z: targetDepth };

    this.targetAlpha = this.alpha;
    this.targetBeta = this.beta;
    this.targetRadius = this.radius;
    this.targetTarget = { x: 0, y: 0, z: targetDepth };

    this.updateCamera();
  }

  /**
   * Reset camera to original input image perspective.
   */
  reset() {
    this.setInputImageView(this.initialDepth);
  }

  private setupEventListeners() {
    // Mouse events
    this.canvas.addEventListener('mousedown', this.onMouseDown);
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('mousemove', this.onMouseMove);
    window.addEventListener('mouseup', this.onMouseUp);
    this.canvas.addEventListener('wheel', this.onWheel, { passive: false });

    // Keyboard events for Shift key
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);

    // Touch events
    this.canvas.addEventListener('touchstart', this.onTouchStart, { passive: false });
    this.canvas.addEventListener('touchmove', this.onTouchMove, { passive: false });
    this.canvas.addEventListener('touchend', this.onTouchEnd);
  }

  private onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Shift') {
      this.isShiftHeld = true;
    }
  };

  private onKeyUp = (e: KeyboardEvent) => {
    if (e.key === 'Shift') {
      this.isShiftHeld = false;
    }
  };

  private onMouseDown = (e: MouseEvent) => {
    e.preventDefault();
    this.isDragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
  };

  private onMouseMove = (e: MouseEvent) => {
    if (!this.isDragging) return;

    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;

    if (this.isShiftHeld) {
      this.pan(dx, dy);
    } else {
      this.orbit(dx, dy);
    }

    this.lastX = e.clientX;
    this.lastY = e.clientY;
  };

  private onMouseUp = () => {
    this.isDragging = false;
  };

  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    this.zoom(e.deltaY);
  };

  private onTouchStart = (e: TouchEvent) => {
    e.preventDefault();

    if (e.touches.length === 1) {
      this.isDragging = true;
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      // Two fingers - pinch zoom and pan
      const [t1, t2] = [e.touches[0], e.touches[1]];
      this.lastPinchDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
      // Initialize midpoint for pan tracking
      this.lastMidX = (t1.clientX + t2.clientX) / 2;
      this.lastMidY = (t1.clientY + t2.clientY) / 2;
    }
  };

  // Track midpoint for two-finger pan
  private lastMidX = 0;
  private lastMidY = 0;

  private onTouchMove = (e: TouchEvent) => {
    e.preventDefault();

    if (e.touches.length === 1) {
      const dx = e.touches[0].clientX - this.lastX;
      const dy = e.touches[0].clientY - this.lastY;
      this.orbit(dx, dy);
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      const [t1, t2] = [e.touches[0], e.touches[1]];
      const pinchDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);

      // Calculate midpoint between two fingers
      const midX = (t1.clientX + t2.clientX) / 2;
      const midY = (t1.clientY + t2.clientY) / 2;

      // Two-finger pan (based on midpoint movement)
      if (this.lastMidX !== 0 && this.lastMidY !== 0) {
        const panDx = midX - this.lastMidX;
        const panDy = midY - this.lastMidY;
        // Only pan if the gesture is more pan-like than pinch-like
        if (Math.abs(panDx) > 2 || Math.abs(panDy) > 2) {
          this.pan(panDx, panDy);
        }
      }

      // Pinch to zoom
      const zoomDelta = (this.lastPinchDist - pinchDist) * 5;
      this.zoom(zoomDelta);

      this.lastPinchDist = pinchDist;
      this.lastMidX = midX;
      this.lastMidY = midY;
    }
  };

  private onTouchEnd = (e: TouchEvent) => {
    if (e.touches.length === 0) {
      this.isDragging = false;
      // Reset midpoint tracking
      this.lastMidX = 0;
      this.lastMidY = 0;
    } else if (e.touches.length === 1) {
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;
      // Reset midpoint tracking when going from 2 to 1 finger
      this.lastMidX = 0;
      this.lastMidY = 0;
    } else if (e.touches.length === 2) {
      const [t1, t2] = [e.touches[0], e.touches[1]];
      this.lastPinchDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
      this.lastMidX = (t1.clientX + t2.clientX) / 2;
      this.lastMidY = (t1.clientY + t2.clientY) / 2;
    }
  };

  private orbit(dx: number, dy: number) {
    this.targetAlpha += dx * this.orbitSpeed;
    this.targetBeta -= dy * this.orbitSpeed;
    this.targetBeta = Math.max(-Math.PI / 2 + 0.1, Math.min(Math.PI / 2 - 0.1, this.targetBeta));
  }

  private pan(dx: number, dy: number) {
    // Calculate right and up vectors based on current camera orientation
    const cosAlpha = Math.cos(this.alpha);
    const sinAlpha = Math.sin(this.alpha);
    const cosBeta = Math.cos(this.beta);

    // Right vector (perpendicular to view direction in XZ plane)
    const rightX = cosAlpha;
    const rightZ = -sinAlpha;

    // Up vector (Y is up, adjusted for pitch)
    const upY = cosBeta;

    // Scale by radius for consistent feel at different zoom levels
    const scale = this.panSpeed * this.radius;

    // Move target position
    this.targetTarget.x -= dx * scale * rightX;
    this.targetTarget.z -= dx * scale * rightZ;
    this.targetTarget.y += dy * scale * upY;
  }

  private zoom(delta: number) {
    this.targetRadius += delta * this.zoomSpeed * this.radius;
    this.targetRadius = Math.max(this.minRadius, Math.min(this.maxRadius, this.targetRadius));
  }

  update() {
    this.alpha += (this.targetAlpha - this.alpha) * this.dampening;
    this.beta += (this.targetBeta - this.beta) * this.dampening;
    this.radius += (this.targetRadius - this.radius) * this.dampening;
    this.target.x += (this.targetTarget.x - this.target.x) * this.dampening;
    this.target.y += (this.targetTarget.y - this.target.y) * this.dampening;
    this.target.z += (this.targetTarget.z - this.target.z) * this.dampening;
    this.updateCamera();
  }

  private updateCamera() {
    const cosBeta = Math.cos(this.beta);
    const x = this.target.x + this.radius * cosBeta * Math.sin(this.alpha);
    const y = this.target.y + this.radius * Math.sin(this.beta);
    const z = this.target.z + this.radius * cosBeta * Math.cos(this.alpha);

    this.camera.position.x = x;
    this.camera.position.y = y;
    this.camera.position.z = z;

    const dx = this.target.x - x;
    const dy = this.target.y - y;
    const dz = this.target.z - z;

    const pitch = Math.atan2(dy, Math.sqrt(dx * dx + dz * dz));
    const yaw = Math.atan2(dx, dz);

    const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2);
    const cp = Math.cos(-pitch / 2), sp = Math.sin(-pitch / 2);

    this.camera.rotation.x = -sp * cy;
    this.camera.rotation.y = sp * sy;
    this.camera.rotation.z = cp * sy;
    this.camera.rotation.w = cp * cy;
  }

  dispose() {
    this.canvas.removeEventListener('mousedown', this.onMouseDown);
    window.removeEventListener('mousemove', this.onMouseMove);
    window.removeEventListener('mouseup', this.onMouseUp);
    this.canvas.removeEventListener('wheel', this.onWheel);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
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
        if (rendererRef.current && containerRef.current && cameraRef.current) {
          const width = containerRef.current.clientWidth;
          const height = containerRef.current.clientHeight;
          rendererRef.current.setSize(width, height);

          // Update camera aspect ratio
          const camera = cameraRef.current as any;
          if (camera.data) {
            camera.data.setSize(width, height);
          }
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

        // Fetch camera metadata and load splat in parallel
        const [_splatResult, _cameraData] = await Promise.all([
          SPLAT.Loader.LoadAsync(splatUrl, scene, (progress) => {
            console.log('Loading progress:', Math.round(progress * 100) + '%');
          }),
          getCameraMetadata(jobId).catch(err => {
            console.warn('Could not fetch camera metadata:', err);
            return null;
          })
        ]);

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

        // Set camera to match input image perspective
        // The target depth determines how far into the scene we look
        // A reasonable default is 2-3 units, but we can adjust based on the scene
        controls.setInputImageView(2.5);
        console.log('Camera set to input image perspective');

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

  // Handle resize - update both renderer and camera aspect ratio
  useEffect(() => {
    const handleResize = () => {
      if (rendererRef.current && containerRef.current && cameraRef.current) {
        const width = containerRef.current.clientWidth;
        const height = containerRef.current.clientHeight;
        rendererRef.current.setSize(width, height);

        // Update camera aspect ratio to prevent distortion
        const camera = cameraRef.current as any;
        if (camera.data) {
          camera.data.setSize(width, height);
        }
      }
    };

    window.addEventListener('resize', handleResize);

    // Also handle fullscreen changes
    document.addEventListener('fullscreenchange', handleResize);
    document.addEventListener('webkitfullscreenchange', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('fullscreenchange', handleResize);
      document.removeEventListener('webkitfullscreenchange', handleResize);
    };
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

      {/* Reset button */}
      {!loading && (
        <button
          onClick={() => controlsRef.current?.reset()}
          className="absolute top-3 left-3 bg-black/70 hover:bg-black/90 text-white text-xs px-4 py-3 sm:px-3 sm:py-2 rounded-lg transition-colors z-20 touch-manipulation"
          title="Reset to original view"
          style={{ minWidth: '44px', minHeight: '44px' }}
        >
          Reset View
        </button>
      )}

      {/* Controls help */}
      {!loading && !isInFullscreen && (
        <div className="absolute bottom-3 right-3 bg-black/60 text-white text-xs px-2 py-1 rounded z-20">
          <div>{isMobile ? '1 Finger' : 'Drag'}: Orbit</div>
          <div>{isMobile ? '2 Fingers' : 'Shift+Drag'}: Pan</div>
          <div>{isMobile ? 'Pinch' : 'Scroll'}: Zoom</div>
        </div>
      )}
    </div>
  );
}
