declare module 'gsplat' {
  export class Scene {
    objects: any[];
    addObject(object: any): void;
    removeObject(object: any): void;
  }

  export class Camera {
    // gsplat.js Camera - properties managed internally
  }

  export class WebGLRenderer {
    constructor(canvas?: HTMLCanvasElement);
    canvas: HTMLCanvasElement;
    render(scene: Scene, camera: Camera): void;
    setSize(width: number, height: number): void;
    dispose(): void;
  }

  export class OrbitControls {
    constructor(camera: Camera, domElement: HTMLElement);
    update(): void;
    dispose(): void;
  }

  export class Loader {
    static LoadAsync(
      url: string,
      scene: Scene,
      onProgress?: (progress: number) => void
    ): Promise<void>;
    static LoadFromFileAsync(
      file: File,
      scene: Scene,
      onProgress?: (progress: number) => void
    ): Promise<void>;
  }

  export class PLYLoader {
    static LoadAsync(
      url: string,
      scene: Scene,
      onProgress?: (progress: number) => void,
      format?: string
    ): Promise<void>;
    static LoadFromFileAsync(
      file: File,
      scene: Scene,
      onProgress?: (progress: number) => void,
      format?: string
    ): Promise<void>;
  }

  export class Splat {
    splatCount: number;
  }
}
