# SHARP Web UI

A web interface for [Apple's SHARP model](https://github.com/apple/ml-sharp) — turn any single photo into an interactive 3D Gaussian splat.

![SHARP Web UI](https://img.shields.io/badge/GPU-Modal-purple) ![Frontend](https://img.shields.io/badge/Frontend-Vercel-black) ![License](https://img.shields.io/badge/License-MIT-green)

**[Live Demo](https://frontend-woad-nu-76.vercel.app)** · **[SHARP Paper](https://machinelearning.apple.com/research/sharp)**

## What is this?

SHARP (Single-image 3D Gaussian Splatting) is a state-of-the-art model from Apple Research that generates 3D Gaussian splat reconstructions from a single image. This web UI makes it accessible to everyone — no coding required.

1. Upload any photo
2. Wait ~30-60 seconds for GPU processing
3. Explore the 3D result in your browser
4. Download the `.ply` file for use in other tools

## Architecture

```
┌─────────────────────┐         ┌─────────────────────────┐
│       Vercel        │         │         Modal           │
│     (Frontend)      │         │       (GPU API)         │
│                     │         │                         │
│   React + Vite      │◄───────►│  FastAPI + SHARP Model  │
│   gsplat.js viewer  │  HTTPS  │  T4 GPU (serverless)    │
│                     │         │                         │
└─────────────────────┘         └─────────────────────────┘
```

- **Frontend**: React + TypeScript + Tailwind CSS, with [gsplat.js](https://github.com/huggingface/gsplat.js) for real-time 3D rendering
- **Backend**: FastAPI on [Modal](https://modal.com) with serverless T4 GPU, scales to zero when idle
- **Privacy**: Session-based isolation — each browser only sees its own uploads

## Local Development

### Prerequisites

- Node.js 18+
- Python 3.11+
- Modal account (for GPU backend)

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend (Local)

For local development without GPU, you can run the original FastAPI backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Backend (Modal)

```bash
cd modal
pip install modal
modal token new  # Authenticate
modal serve app.py  # Dev mode with hot reload
```

## Deployment

### Deploy Backend to Modal

```bash
# Create API key secret
modal secret create sharp-api-key SHARP_API_KEY="your-secret-key"

# Deploy
cd modal
modal deploy app.py
```

### Deploy Frontend to Vercel

1. Push to GitHub
2. Import project in Vercel
3. Set environment variables:
   - `VITE_API_URL`: Your Modal API URL (e.g., `https://your-workspace--sharp-web-ui-fastapi-app.modal.run`)
   - `VITE_API_KEY`: Same key as Modal secret
4. Deploy

## Project Structure

```
.
├── frontend/           # React frontend
│   ├── src/
│   │   ├── api/       # API client with auth
│   │   ├── components/ # React components
│   │   └── types/     # TypeScript definitions
│   └── ...
├── modal/             # Modal serverless backend
│   └── app.py         # FastAPI + SHARP inference
└── backend/           # Original local backend (optional)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload images for processing |
| `/api/jobs` | GET | List jobs (session-filtered) |
| `/api/status/{id}` | GET | Get job status |
| `/api/download/{id}` | GET | Download PLY file |
| `/api/splat/{id}.splat` | GET | Get .splat for viewer |
| `/api/thumbnail/{id}` | GET | Get original image |

All endpoints require `X-API-Key` header or `api_key` query param.

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, gsplat.js
- **Backend**: FastAPI, Modal, PyTorch, SHARP
- **Deployment**: Vercel (frontend), Modal (GPU backend)

## Cost

- **Modal**: ~$0.59/hr for T4 GPU (only during processing), near-zero when idle
- **Vercel**: Free tier

## Credits

- [SHARP](https://github.com/apple/ml-sharp) by Apple Machine Learning Research
- [gsplat.js](https://github.com/huggingface/gsplat.js) by Hugging Face
- [Modal](https://modal.com) for serverless GPU infrastructure

## License

MIT
