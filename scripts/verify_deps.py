import sys

required = ["fastapi", "torch", "torchvision", "cv2", "sqlalchemy", "PIL", "tqdm", "uvicorn"]
missing = []
for mod in required:
    try:
        __import__(mod if mod != "cv2" else "cv2")
    except ImportError:
        missing.append(mod)

if missing:
    print("Missing:", ", ".join(missing))
    sys.exit(1)
print("All dependencies OK")
