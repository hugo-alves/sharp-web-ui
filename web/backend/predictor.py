"""SHARP model wrapper for web API."""

import os
import ssl
import torch
from pathlib import Path
from typing import Optional
import numpy as np
import urllib.request

from sharp.models import PredictorParams, create_predictor
from sharp.cli.predict import predict_image
from sharp.utils.io import load_rgb
from sharp.utils.gaussians import save_ply, Gaussians3D


DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def download_with_ssl_workaround(url: str, dest: Path) -> None:
    """Download file with SSL certificate workaround for macOS."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Try with default SSL first
    try:
        print(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, dest)
        return
    except ssl.SSLCertVerificationError:
        pass

    # Fallback: disable SSL verification (not ideal, but works for trusted Apple URL)
    print("SSL verification failed, retrying with workaround...")
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl_context)
    )
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, dest)
    print(f"Model downloaded to {dest}")


class SHARPPredictor:
    """Singleton wrapper for SHARP model with lazy loading."""

    _instance: Optional["SHARPPredictor"] = None
    _loading: bool = False

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load model if not already loaded."""
        if self._loaded:
            return

        print(f"Loading SHARP model on device: {self.device}")

        if self.checkpoint_path is None:
            # Check if cached
            cached_path = CACHE_DIR / "sharp_2572gikvuh.pt"
            if not cached_path.exists():
                download_with_ssl_workaround(DEFAULT_MODEL_URL, cached_path)

            state_dict = torch.load(cached_path, weights_only=True, map_location=self.device)
        else:
            state_dict = torch.load(self.checkpoint_path, weights_only=True, map_location=self.device)

        self.predictor = create_predictor(PredictorParams())
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        self.predictor.to(self.device)
        self._loaded = True
        print("SHARP model loaded successfully")

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[str] = None, device: Optional[str] = None) -> "SHARPPredictor":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(checkpoint_path, device)
        return cls._instance

    def predict_from_path(
        self,
        image_path: Path,
        output_path: Path
    ) -> dict:
        """Run prediction on image file and save PLY output.

        Args:
            image_path: Path to input image
            output_path: Path to save output PLY file

        Returns:
            dict with prediction metadata
        """
        self._ensure_loaded()

        image, icc_profile, f_px = load_rgb(image_path)
        height, width = image.shape[:2]

        with torch.no_grad():
            gaussians = predict_image(
                self.predictor,
                image,
                f_px,
                self.device
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_ply(gaussians, f_px, (height, width), output_path)

        num_gaussians = gaussians.mean_vectors.shape[1]

        return {
            "num_gaussians": num_gaussians,
            "image_width": width,
            "image_height": height,
            "focal_length_px": f_px,
            "output_path": str(output_path),
        }

    def predict_from_array(
        self,
        image: np.ndarray,
        f_px: float,
        output_path: Path
    ) -> dict:
        """Run prediction on numpy array and save PLY output.

        Args:
            image: RGB image as numpy array (H, W, 3) uint8
            f_px: Focal length in pixels
            output_path: Path to save output PLY file

        Returns:
            dict with prediction metadata
        """
        self._ensure_loaded()

        height, width = image.shape[:2]

        with torch.no_grad():
            gaussians = predict_image(
                self.predictor,
                image,
                f_px,
                self.device
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_ply(gaussians, f_px, (height, width), output_path)

        num_gaussians = gaussians.mean_vectors.shape[1]

        return {
            "num_gaussians": num_gaussians,
            "image_width": width,
            "image_height": height,
            "focal_length_px": f_px,
            "output_path": str(output_path),
        }
