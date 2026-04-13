from fastapi import FastAPI, HTTPException, Query
from face_extractor import extract_face
from verifier import verify_faces
from liveness import check_liveness
import os

app = FastAPI(title="Biometric Service", version="2.0")

OUTPUT_DIR = "output"
IMAGE_FACE = os.path.join(OUTPUT_DIR, "image_face.jpg")
SELFIE_FACE = os.path.join(OUTPUT_DIR, "selfie_face.jpg")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {
        "service": "biometric_service",
        "status": "running",
        "endpoints": ["/extract", "/verify", "/liveness"]
    }


@app.get("/extract")
def extract(
    image: str = Query(...),
    selfie: str = Query(...)
):
    ensure_output_dir()

    if not os.path.exists(image):
        raise HTTPException(status_code=404, detail=f"Image not found: {image}")

    if not os.path.exists(selfie):
        raise HTTPException(status_code=404, detail=f"Selfie not found: {selfie}")

    image_face = extract_face(image, IMAGE_FACE)
    selfie_face = extract_face(selfie, SELFIE_FACE)

    if image_face is None:
        raise HTTPException(status_code=400, detail="No face detected in image")

    if selfie_face is None:
        raise HTTPException(status_code=400, detail="No face detected in selfie")

    return {
        "image_face": image_face,
        "selfie_face": selfie_face
    }


@app.get("/verify")
def verify(
    image: str = Query(...),
    selfie: str = Query(...),
    force_reextract: bool = Query(False)
):
    ensure_output_dir()

    if not os.path.exists(image):
        raise HTTPException(status_code=404, detail=f"Image not found: {image}")

    if not os.path.exists(selfie):
        raise HTTPException(status_code=404, detail=f"Selfie not found: {selfie}")

    if force_reextract or not os.path.exists(IMAGE_FACE):
        image_face = extract_face(image, IMAGE_FACE)
        if image_face is None:
            raise HTTPException(status_code=400, detail="No face detected in image")

    if force_reextract or not os.path.exists(SELFIE_FACE):
        selfie_face = extract_face(selfie, SELFIE_FACE)
        if selfie_face is None:
            raise HTTPException(status_code=400, detail="No face detected in selfie")

    result = verify_faces(IMAGE_FACE, SELFIE_FACE)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.get("/liveness")
def liveness(
    selfie: str = Query(...)
):
    if not os.path.exists(selfie):
        raise HTTPException(status_code=404, detail=f"Selfie not found: {selfie}")

    result = check_liveness(selfie)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result