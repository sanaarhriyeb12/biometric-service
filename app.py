from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from face_extractor import extract_face
from verifier import verify_faces
import os
import shutil
import uuid

app = FastAPI(title="Biometric Service", version="2.0")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

cors_origins_env = os.getenv("CORS_ORIGINS", "*")
cors_origins = (
    ["*"]
    if cors_origins_env.strip() in {"", "*"}
    else [o.strip() for o in cors_origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/biometric/verify")
async def biometric_verify(
    selfie: UploadFile = File(...),
    document_face: UploadFile = File(...),
):
    """
    Expected by backend:
    POST /biometric/verify (multipart/form-data)
      - selfie
      - document_face
    """
    request_id = str(uuid.uuid4())
    selfie_path = os.path.join(UPLOAD_DIR, f"{request_id}_selfie_{selfie.filename}")
    document_path = os.path.join(UPLOAD_DIR, f"{request_id}_document_{document_face.filename}")

    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(selfie.file, buffer)

    with open(document_path, "wb") as buffer:
        shutil.copyfileobj(document_face.file, buffer)

    selfie_face_path = os.path.join(OUTPUT_DIR, f"{request_id}_selfie_face.jpg")
    document_face_path = os.path.join(OUTPUT_DIR, f"{request_id}_document_face.jpg")

    selfie_face_extracted = extract_face(selfie_path, selfie_face_path)
    document_face_extracted = extract_face(document_path, document_face_path)

    if document_face_extracted is None:
        raise HTTPException(status_code=400, detail="No face detected in document_face")

    if selfie_face_extracted is None:
        raise HTTPException(status_code=400, detail="No face detected in selfie")

    result = verify_faces(document_face_extracted, selfie_face_extracted)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


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