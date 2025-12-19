"""FastAPI backend for SHARP web UI."""

import asyncio
import io
import json
import shutil
import uuid
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Make predictor optional for testing without ML dependencies
try:
    from predictor import SHARPPredictor
    HAS_PREDICTOR = True
except ImportError as e:
    print(f"Warning: Could not import predictor ({e}). Upload/processing disabled.")
    SHARPPredictor = None
    HAS_PREDICTOR = False

app = FastAPI(title="SHARP Web UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
JOBS_FILE = DATA_DIR / "jobs.json"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    id: str
    filename: str
    status: JobStatus
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None


jobs: dict[str, Job] = {}
predictor: Optional[SHARPPredictor] = None


def save_jobs():
    """Persist jobs to disk."""
    jobs_data = {job_id: job.model_dump() for job_id, job in jobs.items()}
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs_data, f, indent=2)


def load_jobs():
    """Load jobs from disk."""
    global jobs
    if JOBS_FILE.exists():
        try:
            with open(JOBS_FILE, "r") as f:
                jobs_data = json.load(f)

            for job_id, job_dict in jobs_data.items():
                # Only load completed or failed jobs (skip pending/processing)
                if job_dict["status"] in ["completed", "failed"]:
                    # Verify files still exist for completed jobs
                    if job_dict["status"] == "completed":
                        output_path = OUTPUT_DIR / f"{job_id}.ply"
                        if not output_path.exists():
                            continue  # Skip if output file is missing

                    jobs[job_id] = Job(**job_dict)

            print(f"Loaded {len(jobs)} jobs from disk")
        except Exception as e:
            print(f"Error loading jobs: {e}")

    # Recover any orphaned output files (PLY files without job entries)
    recovered = 0
    for ply_file in OUTPUT_DIR.glob("*.ply"):
        job_id = ply_file.stem
        if job_id not in jobs:
            # Try to find the original upload to get filename
            filename = f"recovered_{job_id}"
            for ext in [".jpg", ".jpeg", ".png", ".heic", ".webp"]:
                upload_path = UPLOAD_DIR / f"{job_id}{ext}"
                if upload_path.exists():
                    filename = f"image{ext}"
                    break

            jobs[job_id] = Job(
                id=job_id,
                filename=filename,
                status=JobStatus.COMPLETED,
                created_at=datetime.fromtimestamp(ply_file.stat().st_mtime).isoformat(),
                completed_at=datetime.fromtimestamp(ply_file.stat().st_mtime).isoformat(),
            )
            recovered += 1

    if recovered > 0:
        print(f"Recovered {recovered} orphaned output files")
        save_jobs()


@app.on_event("startup")
async def startup_event():
    """Load SHARP model and existing jobs on startup."""
    global predictor
    load_jobs()
    if HAS_PREDICTOR:
        predictor = SHARPPredictor.get_instance()
    else:
        print("Running in viewer-only mode (no ML processing available)")


def process_image(job_id: str, image_path: Path, output_path: Path):
    """Process a single image (runs in background)."""
    global predictor, jobs

    try:
        if not HAS_PREDICTOR or predictor is None:
            raise RuntimeError("ML processing not available - running in viewer-only mode")

        jobs[job_id].status = JobStatus.PROCESSING
        save_jobs()

        result = predictor.predict_from_path(image_path, output_path)

        jobs[job_id].status = JobStatus.COMPLETED
        jobs[job_id].completed_at = datetime.now().isoformat()
        jobs[job_id].result = result
        save_jobs()

    except Exception as e:
        jobs[job_id].status = JobStatus.FAILED
        jobs[job_id].error = str(e)
        jobs[job_id].completed_at = datetime.now().isoformat()
        save_jobs()


@app.post("/api/upload")
async def upload_images(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...)
):
    """Upload one or more images for processing."""
    created_jobs = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            continue

        job_id = str(uuid.uuid4())
        filename = file.filename or f"image_{job_id}"
        ext = Path(filename).suffix or ".jpg"

        image_path = UPLOAD_DIR / f"{job_id}{ext}"
        output_path = OUTPUT_DIR / f"{job_id}.ply"

        content = await file.read()
        with open(image_path, "wb") as f:
            f.write(content)

        job = Job(
            id=job_id,
            filename=filename,
            status=JobStatus.PENDING,
            created_at=datetime.now().isoformat(),
        )
        jobs[job_id] = job
        created_jobs.append(job)
        save_jobs()

        background_tasks.add_task(process_image, job_id, image_path, output_path)

    return {"jobs": [job.model_dump() for job in created_jobs]}


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id].model_dump()


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return {"jobs": [job.model_dump() for job in jobs.values()]}


@app.get("/api/download/{job_id}")
async def download_ply(job_id: str):
    """Download PLY file for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    output_path = OUTPUT_DIR / f"{job_id}.ply"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    safe_filename = Path(job.filename).stem + ".ply"
    return FileResponse(
        output_path,
        media_type="application/octet-stream",
        filename=safe_filename
    )


@app.get("/api/download-all")
async def download_all():
    """Download all completed jobs as a ZIP file."""
    completed_jobs = [j for j in jobs.values() if j.status == JobStatus.COMPLETED]

    if not completed_jobs:
        raise HTTPException(status_code=400, detail="No completed jobs")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for job in completed_jobs:
            output_path = OUTPUT_DIR / f"{job.id}.ply"
            if output_path.exists():
                safe_filename = Path(job.filename).stem + ".ply"
                zf.write(output_path, safe_filename)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=gaussians.zip"}
    )


@app.get("/api/splat/{job_id}.splat")
async def get_splat_file(job_id: str):
    """Convert SHARP PLY to .splat format for gsplat.js viewer."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    output_path = OUTPUT_DIR / f"{job_id}.ply"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    # Check for cached converted file
    splat_path = OUTPUT_DIR / f"{job_id}.splat"
    if splat_path.exists():
        return FileResponse(
            splat_path,
            media_type="application/octet-stream",
            filename=f"{Path(job.filename).stem}.splat"
        )

    from plyfile import PlyData
    import numpy as np

    # Read original SHARP PLY
    ply_data = PlyData.read(output_path)
    vertex = ply_data["vertex"]
    num_vertices = len(vertex["x"])

    # Extract data as numpy arrays
    positions = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)

    # Scales need exponential (stored as log in PLY)
    scales = np.column_stack([
        np.exp(vertex["scale_0"]),
        np.exp(vertex["scale_1"]),
        np.exp(vertex["scale_2"])
    ]).astype(np.float32)

    # Convert SH DC coefficients to RGB (0-255)
    # SH to RGB: color = sh * C0 + 0.5, where C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    colors = np.column_stack([
        np.clip(vertex["f_dc_0"] * C0 + 0.5, 0, 1) * 255,
        np.clip(vertex["f_dc_1"] * C0 + 0.5, 0, 1) * 255,
        np.clip(vertex["f_dc_2"] * C0 + 0.5, 0, 1) * 255,
        # Opacity: sigmoid activation then to 0-255
        (1 / (1 + np.exp(-vertex["opacity"]))) * 255
    ]).astype(np.uint8)

    # Normalize quaternions and convert to 0-255 range
    rotations = np.column_stack([
        vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]
    ])
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    rotations = (rotations / norms * 128 + 128).astype(np.uint8)

    # Sort by opacity-weighted scale (descending) for front-to-back rendering
    scale_magnitude = np.prod(scales, axis=1)
    opacity_values = colors[:, 3].astype(np.float32) / 255.0
    sort_indices = np.argsort(-(scale_magnitude * opacity_values))

    # Apply sorting
    positions = positions[sort_indices]
    scales = scales[sort_indices]
    colors = colors[sort_indices]
    rotations = rotations[sort_indices]

    # Write .splat file (32 bytes per splat) - interleave arrays for correct format
    # Format per splat: position(12) + scale(12) + color(4) + rotation(4) = 32 bytes
    splat_data = np.empty((num_vertices, 32), dtype=np.uint8)
    splat_data[:, 0:12] = positions.view(np.uint8).reshape(-1, 12)
    splat_data[:, 12:24] = scales.view(np.uint8).reshape(-1, 12)
    splat_data[:, 24:28] = colors
    splat_data[:, 28:32] = rotations

    with open(splat_path, "wb") as f:
        f.write(splat_data.tobytes())

    return FileResponse(
        splat_path,
        media_type="application/octet-stream",
        filename=f"{Path(job.filename).stem}.splat"
    )


@app.get("/api/preview/{job_id}")
async def get_preview_data(job_id: str):
    """Get point cloud data for 3D preview."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    output_path = OUTPUT_DIR / f"{job_id}.ply"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    from plyfile import PlyData

    ply_data = PlyData.read(output_path)
    vertex = ply_data["vertex"]

    max_points = 50000
    total_points = len(vertex["x"])
    step = max(1, total_points // max_points)

    positions = []
    colors = []

    for i in range(0, total_points, step):
        positions.extend([
            float(vertex["x"][i]),
            float(vertex["y"][i]),
            float(vertex["z"][i])
        ])

        r = float(vertex["f_dc_0"][i])
        g = float(vertex["f_dc_1"][i])
        b = float(vertex["f_dc_2"][i])

        def sh_to_rgb(sh):
            return max(0, min(1, sh * 0.28209479177387814 + 0.5))

        colors.extend([
            sh_to_rgb(r),
            sh_to_rgb(g),
            sh_to_rgb(b)
        ])

    return {
        "positions": positions,
        "colors": colors,
        "num_points": len(positions) // 3,
        "total_gaussians": total_points
    }


@app.get("/api/thumbnail/{job_id}")
async def get_thumbnail(job_id: str):
    """Get the original uploaded image as thumbnail."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    for ext in [".jpg", ".jpeg", ".png", ".heic", ".webp"]:
        image_path = UPLOAD_DIR / f"{job_id}{ext}"
        if image_path.exists():
            return FileResponse(image_path)

    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/camera/{job_id}")
async def get_camera_metadata(job_id: str):
    """Get camera metadata from PLY file for viewer initialization.

    Returns intrinsics, extrinsics, and image size so the viewer can
    display the SPLAT from the same perspective as the original image.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    output_path = OUTPUT_DIR / f"{job_id}.ply"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    from plyfile import PlyData
    import numpy as np

    ply_data = PlyData.read(output_path)

    # Extract supplementary camera data from PLY
    supplement_data = {}
    for element in ply_data.elements:
        if element.name == "intrinsic":
            supplement_data["intrinsic"] = np.asarray(element["intrinsic"])
        elif element.name == "extrinsic":
            supplement_data["extrinsic"] = np.asarray(element["extrinsic"])
        elif element.name == "image_size":
            supplement_data["image_size"] = np.asarray(element["image_size"])

    # Parse intrinsics (3x3 matrix stored as 9 elements)
    intrinsics = supplement_data.get("intrinsic", np.array([512, 0, 320, 0, 512, 240, 0, 0, 1]))
    if len(intrinsics) == 9:
        intrinsics_matrix = intrinsics.reshape((3, 3))
        focal_x = float(intrinsics_matrix[0, 0])
        focal_y = float(intrinsics_matrix[1, 1])
        cx = float(intrinsics_matrix[0, 2])
        cy = float(intrinsics_matrix[1, 2])
    else:
        # Legacy format
        focal_x = float(intrinsics[0]) if len(intrinsics) > 0 else 512
        focal_y = float(intrinsics[1]) if len(intrinsics) > 1 else 512
        cx = float(intrinsics[2]) if len(intrinsics) > 2 else 320
        cy = float(intrinsics[3]) if len(intrinsics) > 3 else 240

    # Parse extrinsics (4x4 matrix stored as 16 elements)
    extrinsics = supplement_data.get("extrinsic", np.eye(4).flatten())
    extrinsics_matrix = np.eye(4)
    if len(extrinsics) == 16:
        extrinsics_matrix = extrinsics.reshape((4, 4))
    elif len(extrinsics) == 12:
        extrinsics_matrix[:3] = extrinsics.reshape((3, 4))

    # Parse image size
    image_size = supplement_data.get("image_size", np.array([640, 480]))
    width = int(image_size[0])
    height = int(image_size[1])

    # Calculate vertical field of view from focal length
    # FOV = 2 * atan(sensor_size / (2 * focal_length))
    fov_y = float(2 * np.arctan(height / (2 * focal_y)))
    fov_x = float(2 * np.arctan(width / (2 * focal_x)))

    # Extract camera position from extrinsics (inverse of camera transform)
    # For identity extrinsics, camera is at origin looking down +Z
    camera_position = extrinsics_matrix[:3, 3].tolist()

    # Extract rotation (first 3x3 block)
    rotation_matrix = extrinsics_matrix[:3, :3].tolist()

    return {
        "intrinsics": {
            "focal_x": focal_x,
            "focal_y": focal_y,
            "principal_x": cx,
            "principal_y": cy,
        },
        "extrinsics": {
            "position": camera_position,
            "rotation": rotation_matrix,
        },
        "image_size": {
            "width": width,
            "height": height,
        },
        "fov": {
            "horizontal": fov_x,
            "vertical": fov_y,
        }
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    for ext in [".jpg", ".jpeg", ".png", ".heic", ".webp"]:
        image_path = UPLOAD_DIR / f"{job_id}{ext}"
        if image_path.exists():
            image_path.unlink()

    output_path = OUTPUT_DIR / f"{job_id}.ply"
    if output_path.exists():
        output_path.unlink()

    del jobs[job_id]
    save_jobs()
    return {"status": "deleted"}


@app.delete("/api/jobs")
async def clear_all_jobs():
    """Clear all jobs and files."""
    for job_id in list(jobs.keys()):
        for ext in [".jpg", ".jpeg", ".png", ".heic", ".webp"]:
            image_path = UPLOAD_DIR / f"{job_id}{ext}"
            if image_path.exists():
                image_path.unlink()

        output_path = OUTPUT_DIR / f"{job_id}.ply"
        if output_path.exists():
            output_path.unlink()

    jobs.clear()
    save_jobs()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
