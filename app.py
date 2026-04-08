from fastapi import FastAPI, HTTPException, Query
from face_extractor import extract_face
from verifier import verify_faces
import os

app = FastAPI(title="Biometric Service", version="1.0.1")


OUTPUT_DIR = "output"
DOC_FACE_PATH = os.path.join(OUTPUT_DIR, "doc_face.png")
SELFIE_FACE_PATH = os.path.join(OUTPUT_DIR, "selfie_face.png")


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {
        "service": "biometric_service",
        "status": "running",
        "endpoints": ["/test-face", "/verify-face"]
    }


@app.get("/test-face")
def test_face(
    doc_path: str = Query("image.png"),
    selfie_path: str = Query("selfie.jpg")
):
    ensure_output_dir()

    if not os.path.exists(doc_path):
        raise HTTPException(404, f"Document image not found: {doc_path}")

    if not os.path.exists(selfie_path):
        raise HTTPException(404, f"Selfie image not found: {selfie_path}")

    # Extract faces
    doc_face = extract_face(doc_path, DOC_FACE_PATH, is_document=True)
    selfie_face = extract_face(selfie_path, SELFIE_FACE_PATH, is_document=False)

    if doc_face is None:
        raise HTTPException(400, "Failed to extract face from document")

    if selfie_face is None:
        raise HTTPException(400, "Failed to extract face from selfie")

    return {
        "doc_face": DOC_FACE_PATH,
        "selfie_face": SELFIE_FACE_PATH
    }


@app.get("/verify-face")
def verify_face(
    doc_path: str = Query("image.png"),
    selfie_path: str = Query("selfie.jpg"),
    force_reextract: bool = Query(False)
):
    ensure_output_dir()

    if not os.path.exists(doc_path):
        raise HTTPException(404, f"Document image not found: {doc_path}")

    if not os.path.exists(selfie_path):
        raise HTTPException(404, f"Selfie image not found: {selfie_path}")

    # 🔥 Only re-extract if needed
    if force_reextract or not os.path.exists(DOC_FACE_PATH):
        doc_face = extract_face(doc_path, DOC_FACE_PATH, is_document=True)
        if doc_face is None:
            raise HTTPException(400, "Failed to extract face from document")

    if force_reextract or not os.path.exists(SELFIE_FACE_PATH):
        selfie_face = extract_face(selfie_path, SELFIE_FACE_PATH, is_document=False)
        if selfie_face is None:
            raise HTTPException(400, "Failed to extract face from selfie")

    # 🔥 Verify using already extracted faces
    result = verify_faces(DOC_FACE_PATH, SELFIE_FACE_PATH)

    if result is None:
        raise HTTPException(500, "Verification failed")

    return result


@app.get("/liveness")
def liveness():
    return {
        "status": "NOT_IMPLEMENTED",
        "note": "Use video-based liveness (blink / head movement) for real systems."
    }