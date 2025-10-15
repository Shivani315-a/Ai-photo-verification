from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import cv2
from services import analyze_face

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    # Open the frontend in browser
    return FileResponse("static/index.html")


@app.post("/analyze-face/")
async def analyze_face_api(file: UploadFile = File(...)):
    try:
        img_id = str(uuid.uuid4())
        img_path = os.path.join(UPLOAD_DIR, f"{img_id}_{file.filename}")
        with open(img_path, "wb") as f:
            f.write(await file.read())

        img = cv2.imread(img_path)
        os.remove(img_path)

        if img is None:
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)

        result = analyze_face(img)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
