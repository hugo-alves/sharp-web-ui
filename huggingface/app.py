"""SHARP Gradio app for HuggingFace Spaces deployment."""

import logging
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import spaces
import torch
import torch.nn.functional as F

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

# Global predictor (loaded once)
predictor = None


def load_model():
    """Load SHARP model."""
    global predictor
    if predictor is not None:
        return predictor

    LOGGER.info("Loading SHARP model...")
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    LOGGER.info("Model loaded successfully")
    return predictor


def predict_image(
    model,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in NDC space
    with torch.no_grad():
        gaussians_ndc = model(image_resized_pt, disparity_factor)

    intrinsics = (
        torch.tensor([
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metric space
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians


def create_preview_ply(gaussians: Gaussians3D, output_path: Path, max_points: int = 100000):
    """Create a simplified point cloud PLY for preview."""
    # Extract positions and colors
    positions = gaussians.mean_vectors[0].cpu().numpy()  # (N, 3)
    sh_coeffs = gaussians.sh_coefficients[0].cpu().numpy()  # (N, 3) for DC component

    # Convert SH DC to RGB
    def sh_to_rgb(sh):
        return np.clip(sh * 0.28209479177387814 + 0.5, 0, 1)

    colors = sh_to_rgb(sh_coeffs[:, :3])  # Take first 3 channels (RGB)

    # Subsample if needed
    total_points = len(positions)
    if total_points > max_points:
        indices = np.random.choice(total_points, max_points, replace=False)
        positions = positions[indices]
        colors = colors[indices]

    # Center the point cloud
    center = positions.mean(axis=0)
    positions = positions - center

    # Write simple PLY
    num_points = len(positions)
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(num_points):
            r, g, b = (colors[i] * 255).astype(np.uint8)
            f.write(f"{positions[i, 0]:.6f} {positions[i, 1]:.6f} {positions[i, 2]:.6f} {r} {g} {b}\n")

    return num_points


def create_preview_glb(gaussians: Gaussians3D, output_path: Path, max_points: int = 50000):
    """Create a GLB file with colored point spheres for preview."""
    import struct
    import json

    # Extract positions and colors
    positions = gaussians.mean_vectors[0].cpu().numpy()  # (N, 3)
    sh_coeffs = gaussians.sh_coefficients[0].cpu().numpy()  # (N, C)

    # Convert SH DC to RGB
    def sh_to_rgb(sh):
        return np.clip(sh * 0.28209479177387814 + 0.5, 0, 1)

    colors = sh_to_rgb(sh_coeffs[:, :3])

    # Subsample if needed
    total_points = len(positions)
    if total_points > max_points:
        indices = np.random.choice(total_points, max_points, replace=False)
        positions = positions[indices]
        colors = colors[indices]

    # Center and scale
    center = positions.mean(axis=0)
    positions = positions - center
    scale = np.abs(positions).max()
    if scale > 0:
        positions = positions / scale

    num_points = len(positions)

    # Create binary buffer with positions and colors
    positions_f32 = positions.astype(np.float32)
    colors_u8 = (colors * 255).astype(np.uint8)
    # Add alpha channel
    colors_rgba = np.hstack([colors_u8, np.full((num_points, 1), 255, dtype=np.uint8)])

    # Combine into buffer
    buffer_data = b''
    buffer_data += positions_f32.tobytes()
    buffer_data += colors_rgba.tobytes()

    positions_byte_length = positions_f32.nbytes
    colors_byte_length = colors_rgba.nbytes

    # Pad buffer to 4-byte alignment
    while len(buffer_data) % 4 != 0:
        buffer_data += b'\x00'

    # Calculate bounds
    pos_min = positions_f32.min(axis=0).tolist()
    pos_max = positions_f32.max(axis=0).tolist()

    # Build glTF JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "SHARP"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "COLOR_0": 1
                },
                "mode": 0  # POINTS
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": num_points,
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max
            },
            {
                "bufferView": 1,
                "componentType": 5121,  # UNSIGNED_BYTE
                "count": num_points,
                "type": "VEC4",
                "normalized": True
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": positions_byte_length
            },
            {
                "buffer": 0,
                "byteOffset": positions_byte_length,
                "byteLength": colors_byte_length
            }
        ],
        "buffers": [{
            "byteLength": len(buffer_data)
        }]
    }

    # Encode JSON
    gltf_json = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    # Pad JSON to 4-byte alignment
    while len(gltf_json) % 4 != 0:
        gltf_json += b' '

    # Build GLB
    glb_data = b''
    # Header
    glb_data += struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(gltf_json) + 8 + len(buffer_data))
    # JSON chunk
    glb_data += struct.pack('<II', len(gltf_json), 0x4E4F534A)  # JSON
    glb_data += gltf_json
    # Binary chunk
    glb_data += struct.pack('<II', len(buffer_data), 0x004E4942)  # BIN
    glb_data += buffer_data

    with open(output_path, 'wb') as f:
        f.write(glb_data)

    return num_points


@spaces.GPU(duration=120)
def process_image(input_image):
    """Process an image and return 3D Gaussian splat."""
    if input_image is None:
        raise gr.Error("Please upload an image")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    LOGGER.info(f"Using device: {device}")

    # Load model and move to device
    model = load_model()
    model.to(device)

    # Load and process image
    image = np.array(input_image)
    height, width = image.shape[:2]

    # Estimate focal length (common heuristic)
    f_px = max(height, width) * 1.2

    LOGGER.info(f"Processing image: {width}x{height}, f_px={f_px:.1f}")

    # Run prediction
    gaussians = predict_image(model, image, f_px, device)

    num_gaussians = len(gaussians.mean_vectors[0])
    LOGGER.info(f"Generated {num_gaussians:,} gaussians")

    # Save full Gaussian PLY for download
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        full_ply_path = Path(tmp.name)
    save_ply(gaussians, f_px, (height, width), full_ply_path)

    # Create preview GLB for 3D viewer
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        preview_path = Path(tmp.name)
    preview_points = create_preview_glb(gaussians, preview_path, max_points=50000)

    status = f"Generated {num_gaussians:,} 3D Gaussians (showing {preview_points:,} in preview)"

    return str(preview_path), str(full_ply_path), status


# Build Gradio interface
with gr.Blocks(title="SHARP - Single-Image 3D Gaussian Splatting") as demo:
    gr.Markdown("""
    # SHARP - Single-Image 3D Gaussian Splatting

    Upload a single image to generate a 3D Gaussian Splat representation.

    **[Paper](https://machinelearning.apple.com/research/sharp-monocular-view)** |
    **[GitHub](https://github.com/apple/ml-sharp)** |
    **[Project Page](https://apple.github.io/ml-sharp/)**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image",
                type="pil",
                height=400,
            )

            process_btn = gr.Button("Generate 3D Gaussians", variant="primary", size="lg")

            status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            model_3d = gr.Model3D(
                label="3D Preview (Point Cloud)",
                height=400,
                clear_color=[0.1, 0.1, 0.1, 1.0],
            )

            download_file = gr.File(label="Download Full Gaussian PLY")

    gr.Markdown("""
    ### Tips
    - Works best with images that have clear depth cues
    - Processing takes ~10-30 seconds on GPU, longer on CPU
    - The preview shows a simplified point cloud - download the PLY for full quality
    - Use a [3DGS viewer](https://antimatter15.com/splat/) to view the full Gaussian splat

    ### About
    SHARP generates 3D Gaussian Splats from a single image in under a second on GPU.
    """)

    # Wire up the interface
    process_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[model_3d, download_file, status_text],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
