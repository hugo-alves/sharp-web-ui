# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SHARP (Sharp Monocular View Synthesis in Less Than a Second) is an Apple ML Research project that generates 3D Gaussian Splatting representations from single images in under a second. The project consists of a core Python inference package, CLI tools, and a web UI with local/serverless backends.

## Build & Development Commands

```bash
# Python setup
pip install -r requirements.txt

# CLI usage
sharp --help
sharp predict -i <input_image> -o <output.ply>
sharp render -i <gaussians.ply> -o <video.mp4>

# Linting & formatting
ruff check src/
ruff format src/
mypy src/

# Frontend (web/frontend/)
npm install
npm run dev          # Dev server on port 5173
npm run build        # Production build
npm run lint         # ESLint

# Backend options
uvicorn web.backend.main:app --reload    # Local FastAPI
modal serve web/modal/app.py             # Modal serverless
```

## Architecture

### Core Prediction Pipeline (src/sharp/)

The inference flow in `RGBGaussianPredictor` (models/predictor.py):

1. **Input** → RGB image (B, 3, H, W)
2. **Initializer** → Base Gaussian parameters
3. **MonoDepth** → Scene depth prediction
4. **Feature Extraction** → Image feature encoding
5. **Gaussian Decoder** → Predicts Gaussian parameter deltas
6. **Composer** → Final 3D Gaussians (color, scale, rotation, opacity)
7. **Output** → 3D Gaussian Splatting (.ply)

**Key classes:**
- `RGBGaussianPredictor` - Main model orchestrator (models/predictor.py)
- `GaussianComposer` - Combines base + delta predictions (models/composer.py)
- `PredictorParams` - Configuration dataclass (models/params.py)
- `Gaussians3D` - 3D Gaussian representation (utils/gaussians.py)

### Model Components (src/sharp/models/)

- **encoders/** - Image feature extractors (ViT, UNet, MonoDepth, SPN)
- **decoders/** - Generate Gaussian parameters
- **presets/** - Pre-configured model architectures (monodepth, vit)
- **initializer.py** - Base Gaussian position/scale initialization
- **alignment.py** - Depth alignment utilities

### Utilities (src/sharp/utils/)

- **gaussians.py** - 3D math, PLY I/O, Gaussian operations
- **camera.py** - Camera matrices, trajectory generation (circular orbit)
- **gsplat.py** - Integration with gsplat renderer
- **io.py** - Image loading (PIL, HEIF support)
- **color_space.py** - sRGB ↔ linear conversions

### Web Architecture (web/)

```
Frontend (React/Vite) ←→ Backend (FastAPI on Modal T4 GPU)
    ↓
API Client (axios): /api/upload, /api/status, /api/download
    ↓
Job Queue: pending → processing → completed/failed
    ↓
3D Viewer (gsplat.js) renders PLY results
```

- **frontend/** - React 19 + TypeScript + Tailwind + gsplat.js
- **backend/** - Local FastAPI with job queue
- **modal/** - Serverless GPU deployment on Modal

## Key Patterns

**Python:**
- Google-style docstrings, strict type hints (pyright enabled)
- Configuration via dataclasses (`PredictorParams`)
- Factory functions (`create_predictor()`, `create_encoder()`)
- Forward pass signature: `(image, depth=None) → Gaussians3D`

**Frontend:**
- One component per file in src/components/
- Axios for API calls with session management
- `USE_LOCAL_BACKEND` flag in vite.config.ts toggles proxy target

## Output Format

- **Input:** Single RGB image or directory of images
- **Output:** 3D Gaussian Splat (.ply)
- **Coordinate system:** OpenCV (x right, y down, z forward), scene center at (0, 0, +z)

## Hardware Requirements

- **Prediction:** CPU, MPS (Apple Silicon), or CUDA
- **Rendering:** CUDA only (gsplat dependency)
- **Model checkpoint:** ~1GB, auto-downloads to ~/.cache/torch/hub/checkpoints/
