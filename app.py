from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from face_extractor import extract_face
from verifier import verify_faces
import logging
import os
import shutil
import uuid

LOG_LEVEL = logging.DEBUG if os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"} else logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("biometric_service")

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


@app.get("/debug")
async def debug():
    return {
        "status": "debug",
        "routes": [route.path for route in app.router.routes],
        "upload_dir": UPLOAD_DIR,
        "output_dir": OUTPUT_DIR,
    }


def format_biometric_response(reason: str, request_id: str):
    return {
        "request_id": request_id,
        "verified": False,
        "distance": None,
        "score": 0.0,
        "decision": "REJECTED",
        "reason": reason,
    }


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

    logger.debug("biometric_verify start request_id=%s selfie=%s document_face=%s", request_id, selfie.filename, document_face.filename)

    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(selfie.file, buffer)

    with open(document_path, "wb") as buffer:
        shutil.copyfileobj(document_face.file, buffer)

    logger.debug("biometric_verify saved files selfie_path=%s document_path=%s", selfie_path, document_path)

    selfie_face_path = os.path.join(OUTPUT_DIR, f"{request_id}_selfie_face.jpg")
    document_face_path = os.path.join(OUTPUT_DIR, f"{request_id}_document_face.jpg")

    selfie_face_extracted = extract_face(selfie_path, selfie_face_path)
    document_face_extracted = extract_face(document_path, document_face_path)

    logger.debug("biometric_verify extracted selfie_face=%s document_face=%s", selfie_face_extracted, document_face_extracted)

    if document_face_extracted is None:
        logger.warning("biometric_verify no face in document_face request_id=%s", request_id)
        return format_biometric_response("No face detected in document_face", request_id)

    if selfie_face_extracted is None:
        logger.warning("biometric_verify no face in selfie request_id=%s", request_id)
        return format_biometric_response("No face detected in selfie", request_id)

    result = verify_faces(document_face_extracted, selfie_face_extracted)

    if "error" in result:
        logger.error("biometric_verify verification error request_id=%s error=%s", request_id, result["error"])
        return format_biometric_response(result["error"], request_id)

    result["request_id"] = request_id
    logger.info("biometric_verify success request_id=%s decision=%s distance=%s", request_id, result.get("decision"), result.get("distance"))
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

    logger.debug("verify_upload extracted image_face=%s selfie_face=%s", image_face, selfie_face)

    if image_face is None:
        logger.warning("verify_upload no face in image request_id=%s", image_id)
        return format_biometric_response("No face detected in image", image_id)

    if selfie_face is None:
        logger.warning("verify_upload no face in selfie request_id=%s", selfie_id)
        return format_biometric_response("No face detected in selfie", selfie_id)

    result = verify_faces(image_face, selfie_face)

    if "error" in result:
        return format_biometric_response(result["error"], image_id)

    result["request_id"] = image_id
    return result