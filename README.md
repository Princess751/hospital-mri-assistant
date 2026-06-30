# Hospital MRI Assistant

Fresh prototype to help radiologists identify brain tumors from MRI scan images.

## Features

- Upload MRI slices (PNG/JPG/TIFF)
- Classify **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**
- Confidence scores + **Grad-CAM** attention heatmap
- Study history and audit log (SQLite)

> Decision-support only — not a cleared medical device.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | React + Vite + TypeScript |
| Backend | FastAPI + PyTorch ResNet-18 |
| Dataset | [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (public) |

## Quick start

```powershell
cd C:\Users\Owner\Projects\hospital-mri-assistant
.\run.bat
```

Open **http://127.0.0.1:8781**

`run.bat` installs Python deps, downloads the public dataset, trains the model, builds the React UI, and starts the server.

### Manual setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python scripts\setup_lab.py
cd frontend && npm install && npm run build && cd ..
uvicorn app.main:app --host 127.0.0.1 --port 8781
```

### Frontend dev mode

```powershell
# Terminal 1
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8781

# Terminal 2
cd frontend && npm run dev
```

Dev UI: **http://127.0.0.1:5174**

## Project structure

```
hospital-mri-assistant/
├── app/           # FastAPI + SQLite
├── ml/            # Model, preprocessing, Grad-CAM
├── frontend/      # React UI
├── scripts/       # Download, train, setup
├── data/          # MRI images
├── weights/       # Trained model
└── run.bat
```

## Ethics

Use only de-identified data you are authorized to process. Comply with local health-data regulations.
