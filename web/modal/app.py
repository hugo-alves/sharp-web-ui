"""
Modal deployment for SHARP Web UI API.

This module provides a serverless GPU-powered API for the SHARP model,
which converts single images into 3D Gaussian splats.
"""

import modal
import os
import json
import uuid
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from enum import Enum

# Define the Modal app
app = modal.App("sharp-web-ui")

# Create volumes for persistent storage
uploads_volume = modal.Volume.from_name("sharp-uploads", create_if_missing=True)
outputs_volume = modal.Volume.from_name("sharp-outputs", create_if_missing=True)
model_volume = modal.Volume.from_name("sharp-model-cache", create_if_missing=True)

# Define the image with SHARP dependencies
sharp_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6", "libgl1-mesa-glx")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "timm>=1.0.0",
        "pillow>=10.0.0",
        "pillow-heif>=0.13.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "plyfile>=1.0.0",
        "huggingface-hub>=0.20.0",
        "jaxtyping>=0.2.0",
        "fastapi>=0.104.0",
        "python-multipart>=0.0.6",
    )
    .run_commands(
        "pip install git+https://github.com/apple/ml-sharp.git"
    )
)

# Constants
UPLOADS_DIR = Path("/data/uploads")
OUTPUTS_DIR = Path("/data/outputs")
MODEL_CACHE_DIR = Path("/model-cache")
JOBS_FILE = OUTPUTS_DIR / "jobs.json"
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Helper functions for job management
def load_jobs() -> dict:
    """Load jobs from persistent storage."""
    if JOBS_FILE.exists():
        with open(JOBS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_jobs(jobs: dict):
    """Save jobs to persistent storage."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)


def download_model():
    """Download model weights if not cached."""
    import urllib.request
    import ssl

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_CACHE_DIR / "sharp_2572gikvuh.pt"

    if not model_path.exists():
        print(f"Downloading model to {model_path}...")
        try:
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
        except ssl.SSLError:
            # Fallback for SSL issues
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(DEFAULT_MODEL_URL, context=ctx) as response:
                with open(model_path, 'wb') as f:
                    f.write(response.read())
        print("Model downloaded successfully!")

    return model_path


# SHARP Predictor class
class SHARPPredictor:
    """Wrapper for SHARP model inference."""

    _instance = None

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        import torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path or str(download_model())
        self.predictor = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self):
        """Lazy load the model."""
        if self.predictor is not None:
            return

        import torch
        from sharp.models import PredictorParams, create_predictor

        print(f"Loading model from {self.checkpoint_path} to {self.device}...")
        state_dict = torch.load(self.checkpoint_path, weights_only=True, map_location=self.device)
        self.predictor = create_predictor(PredictorParams())
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        self.predictor.to(self.device)
        print("Model loaded successfully!")

    def predict_from_path(self, image_path: Path, output_path: Path) -> dict:
        """Run prediction on an image file."""
        from PIL import Image
        import numpy as np

        # Register HEIF/HEIC support with Pillow
        from pillow_heif import register_heif_opener
        register_heif_opener()

        # Load image using PIL for better compatibility
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image = np.array(img)
        return self.predict_from_array(image, output_path)

    def predict_from_array(self, image: "np.ndarray", output_path: Path) -> dict:
        """Run prediction on a numpy array."""
        import torch
        from sharp.cli.predict import predict_image
        from sharp.utils.gaussians import save_ply

        self._ensure_loaded()

        h, w = image.shape[:2]
        f_px = max(h, w)

        with torch.no_grad():
            gaussians = predict_image(self.predictor, image, f_px, self.device)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # save_ply expects (gaussians, f_px, image_shape, path)
        save_ply(gaussians, f_px, (h, w), output_path)

        return {
            "num_gaussians": int(gaussians.mean_vectors.shape[1]),
            "image_width": w,
            "image_height": h,
            "focal_length_px": float(f_px),
            "output_path": str(output_path),
        }


# FastAPI app
@app.function(
    image=sharp_image,
    gpu="T4",
    timeout=600,
    secrets=[modal.Secret.from_name("sharp-api-key")],
    volumes={
        "/data/uploads": uploads_volume,
        "/data/outputs": outputs_volume,
        "/model-cache": model_volume,
    },
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Security
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse, Response
    from fastapi.security import APIKeyHeader, APIKeyQuery
    from pydantic import BaseModel

    api = FastAPI(title="SHARP Web UI API", version="1.0.0")

    # CORS configuration - allow all origins for public API
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Key authentication
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    api_key_query = APIKeyQuery(name="api_key", auto_error=False)
    session_id_header = APIKeyHeader(name="X-Session-Id", auto_error=False)
    session_id_query = APIKeyQuery(name="session_id", auto_error=False)

    async def verify_api_key(
        api_key_header: str = Security(api_key_header),
        api_key_query: str = Security(api_key_query),
    ):
        api_key = api_key_header or api_key_query
        expected_key = os.environ.get("SHARP_API_KEY")
        if not api_key or api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return api_key

    def get_session_id(
        session_header: str = Security(session_id_header),
        session_query: str = Security(session_id_query),
    ) -> Optional[str]:
        return session_header or session_query

    # Pydantic models
    class Job(BaseModel):
        id: str
        filename: str
        status: str
        created_at: str
        completed_at: Optional[str] = None
        error: Optional[str] = None
        result: Optional[dict] = None

    # Initialize predictor lazily
    predictor = None

    def get_predictor():
        nonlocal predictor
        if predictor is None:
            predictor = SHARPPredictor()
        return predictor

    def process_job(job_id: str):
        """Process a single job."""
        # Reload volumes to see recently committed files
        uploads_volume.reload()
        outputs_volume.reload()

        jobs = load_jobs()
        if job_id not in jobs:
            return

        job = jobs[job_id]
        jobs[job_id]["status"] = JobStatus.PROCESSING.value
        save_jobs(jobs)
        outputs_volume.commit()

        try:
            # Find the uploaded image (check both lowercase and original case)
            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".heic", ".webp", ".JPG", ".JPEG", ".PNG", ".HEIC", ".WEBP"]:
                path = UPLOADS_DIR / f"{job_id}{ext}"
                if path.exists():
                    image_path = path
                    break

            if not image_path:
                raise FileNotFoundError("Uploaded image not found")

            output_path = OUTPUTS_DIR / f"{job_id}.ply"

            pred = get_predictor()
            result = pred.predict_from_path(image_path, output_path)

            jobs = load_jobs()
            jobs[job_id]["status"] = JobStatus.COMPLETED.value
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            jobs[job_id]["result"] = result
            save_jobs(jobs)
            outputs_volume.commit()

        except Exception as e:
            jobs = load_jobs()
            jobs[job_id]["status"] = JobStatus.FAILED.value
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            save_jobs(jobs)
            outputs_volume.commit()

    @api.post("/api/upload")
    async def upload_images(
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...),
        _: str = Depends(verify_api_key),
        session_id: Optional[str] = Depends(get_session_id),
    ):
        """Upload one or more images for processing."""
        uploads_volume.reload()
        outputs_volume.reload()

        created_jobs = []
        jobs = load_jobs()

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

        for file in files:
            if not file.content_type or not file.content_type.startswith("image/"):
                continue

            job_id = str(uuid.uuid4())
            filename = file.filename or f"image_{job_id}"
            ext = Path(filename).suffix or ".jpg"

            image_path = UPLOADS_DIR / f"{job_id}{ext}"

            content = await file.read()
            with open(image_path, "wb") as f:
                f.write(content)

            job = {
                "id": job_id,
                "filename": filename,
                "status": JobStatus.PENDING.value,
                "created_at": datetime.now().isoformat(),
                "session_id": session_id,
            }
            jobs[job_id] = job
            created_jobs.append(job)

        save_jobs(jobs)
        uploads_volume.commit()
        outputs_volume.commit()

        # Process jobs in background to avoid timeout
        for job in created_jobs:
            background_tasks.add_task(process_job, job["id"])

        # Return immediately with pending status
        return {"jobs": created_jobs}

    @api.get("/api/status/{job_id}")
    async def get_job_status(job_id: str, _: str = Depends(verify_api_key)):
        """Get status of a specific job."""
        outputs_volume.reload()
        jobs = load_jobs()
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return jobs[job_id]

    @api.get("/api/jobs")
    async def list_jobs(
        _: str = Depends(verify_api_key),
        session_id: Optional[str] = Depends(get_session_id),
    ):
        """List jobs for the current session only."""
        outputs_volume.reload()
        jobs = load_jobs()
        # Filter by session_id for privacy
        session_jobs = [
            job for job in jobs.values()
            if job.get("session_id") == session_id
        ]
        return {"jobs": session_jobs}

    @api.get("/api/download/{job_id}")
    async def download_ply(job_id: str, _: str = Depends(verify_api_key)):
        """Download PLY file for a completed job."""
        outputs_volume.reload()
        jobs = load_jobs()

        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        if job["status"] != JobStatus.COMPLETED.value:
            raise HTTPException(status_code=400, detail="Job not completed")

        output_path = OUTPUTS_DIR / f"{job_id}.ply"
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found")

        safe_filename = Path(job["filename"]).stem + ".ply"
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=safe_filename,
        )

    @api.get("/api/splat/{job_id}.splat")
    async def get_splat_file(job_id: str, _: str = Depends(verify_api_key)):
        """Get PLY converted to .splat format for gsplat.js viewer."""
        outputs_volume.reload()
        jobs = load_jobs()

        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        if job["status"] != JobStatus.COMPLETED.value:
            raise HTTPException(status_code=400, detail="Job not completed")

        output_path = OUTPUTS_DIR / f"{job_id}.ply"
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found")

        # Check for cached splat file
        splat_path = OUTPUTS_DIR / f"{job_id}.splat"
        if not splat_path.exists():
            # Convert PLY to splat format
            from plyfile import PlyData
            import numpy as np

            ply_data = PlyData.read(output_path)
            vertex = ply_data["vertex"]
            num_vertices = len(vertex["x"])

            positions = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
            scales = np.column_stack([
                np.exp(vertex["scale_0"]),
                np.exp(vertex["scale_1"]),
                np.exp(vertex["scale_2"]),
            ]).astype(np.float32)

            C0 = 0.28209479177387814
            colors = np.column_stack([
                np.clip(vertex["f_dc_0"] * C0 + 0.5, 0, 1) * 255,
                np.clip(vertex["f_dc_1"] * C0 + 0.5, 0, 1) * 255,
                np.clip(vertex["f_dc_2"] * C0 + 0.5, 0, 1) * 255,
                (1 / (1 + np.exp(-vertex["opacity"]))) * 255,
            ]).astype(np.uint8)

            rotations = np.column_stack([
                vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]
            ])
            norms = np.linalg.norm(rotations, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            rotations = (rotations / norms * 128 + 128).astype(np.uint8)

            scale_magnitude = np.prod(scales, axis=1)
            opacity_values = colors[:, 3].astype(np.float32) / 255.0
            sort_indices = np.argsort(-(scale_magnitude * opacity_values))

            positions = positions[sort_indices]
            scales = scales[sort_indices]
            colors = colors[sort_indices]
            rotations = rotations[sort_indices]

            splat_data = np.empty((num_vertices, 32), dtype=np.uint8)
            splat_data[:, 0:12] = positions.view(np.uint8).reshape(-1, 12)
            splat_data[:, 12:24] = scales.view(np.uint8).reshape(-1, 12)
            splat_data[:, 24:28] = colors
            splat_data[:, 28:32] = rotations

            with open(splat_path, "wb") as f:
                f.write(splat_data.tobytes())

            outputs_volume.commit()

        return FileResponse(
            splat_path,
            media_type="application/octet-stream",
            filename=f"{Path(job['filename']).stem}.splat",
        )

    @api.get("/api/thumbnail/{job_id}")
    async def get_thumbnail(job_id: str, _: str = Depends(verify_api_key)):
        """Get the original uploaded image as thumbnail."""
        uploads_volume.reload()
        jobs = load_jobs()

        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        for ext in [".jpg", ".jpeg", ".png", ".heic", ".webp"]:
            image_path = UPLOADS_DIR / f"{job_id}{ext}"
            if image_path.exists():
                return FileResponse(image_path)

        raise HTTPException(status_code=404, detail="Image not found")

    @api.get("/api/download-all")
    async def download_all(_: str = Depends(verify_api_key)):
        """Download all completed jobs as a ZIP file."""
        outputs_volume.reload()
        jobs = load_jobs()

        completed_jobs = [j for j in jobs.values() if j["status"] == JobStatus.COMPLETED.value]

        if not completed_jobs:
            raise HTTPException(status_code=400, detail="No completed jobs")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for job in completed_jobs:
                output_path = OUTPUTS_DIR / f"{job['id']}.ply"
                if output_path.exists():
                    safe_filename = Path(job["filename"]).stem + ".ply"
                    zf.write(output_path, safe_filename)

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=gaussians.zip"},
        )

    @api.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str, _: str = Depends(verify_api_key)):
        """Delete a job and its files."""
        uploads_volume.reload()
        outputs_volume.reload()
        jobs = load_jobs()

        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        for ext in [".jpg", ".jpeg", ".png", ".heic", ".webp"]:
            image_path = UPLOADS_DIR / f"{job_id}{ext}"
            if image_path.exists():
                image_path.unlink()

        for ext in [".ply", ".splat"]:
            output_path = OUTPUTS_DIR / f"{job_id}{ext}"
            if output_path.exists():
                output_path.unlink()

        del jobs[job_id]
        save_jobs(jobs)

        uploads_volume.commit()
        outputs_volume.commit()

        return {"status": "deleted"}

    @api.delete("/api/jobs")
    async def clear_all_jobs(_: str = Depends(verify_api_key)):
        """Clear all jobs and files."""
        uploads_volume.reload()
        outputs_volume.reload()
        jobs = load_jobs()

        for job_id in list(jobs.keys()):
            for ext in [".jpg", ".jpeg", ".png", ".heic", ".webp"]:
                image_path = UPLOADS_DIR / f"{job_id}{ext}"
                if image_path.exists():
                    image_path.unlink()

            for ext in [".ply", ".splat"]:
                output_path = OUTPUTS_DIR / f"{job_id}{ext}"
                if output_path.exists():
                    output_path.unlink()

        save_jobs({})
        uploads_volume.commit()
        outputs_volume.commit()

        return {"status": "cleared"}

    @api.get("/api/health")
    async def health_check():
        """Health check endpoint (no auth required)."""
        return {"status": "healthy", "service": "sharp-web-ui"}

    return api
