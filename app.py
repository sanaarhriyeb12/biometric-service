from fastapi import FastAPI, HTTPException, UploadFile, File
from face_extractor import extract_face
from verifier import verify_faces
from liveness import check_liveness
import os
import shutil
import uuid

app = FastAPI(title="Biometric Service", version="2.0")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/verify-upload")
async def verify_upload(
    image: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    image_id = str(uuid.uuid4())
    selfie_id = str(uuid.uuid4())

    image_path = os.path.join(UPLOAD_DIR, f"{image_id}_{image.filename}")
    selfie_path = os.path.join(UPLOAD_DIR, f"{selfie_id}_{selfie.filename}")

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(selfie.file, buffer)

    image_face_path = os.path.join(OUTPUT_DIR, f"{image_id}_face.jpg")
    selfie_face_path = os.path.join(OUTPUT_DIR, f"{selfie_id}_face.jpg")

    image_face = extract_face(image_path, image_face_path)
    selfie_face = extract_face(selfie_path, selfie_face_path)

    if image_face is None:
        raise HTTPException(status_code=400, detail="No face detected in image")

    if selfie_face is None:
        raise HTTPException(status_code=400, detail="No face detected in selfie")

    result = verify_faces(image_face, selfie_face)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result