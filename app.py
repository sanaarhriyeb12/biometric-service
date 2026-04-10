from fastapi import FastAPI, HTTPException, Query
from face_extractor import extract_face
from verifier import verify_faces
import os

app = FastAPI(title="Biometric Service", version="2.0")

OUTPUT_DIR = "output"
DOC_FACE = os.path.join(OUTPUT_DIR, "image_face.jpg")
SELFIE_FACE = os.path.join(OUTPUT_DIR, "selfie_face.jpg")


def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/verify")
def verify(
    image: str = Query("image.jpg"),
    selfie: str = Query("selfie.jpg")
):
    ensure_output()

    if not os.path.exists(image):
        raise HTTPException(404, "Image not found")

    if not os.path.exists(selfie):
        raise HTTPException(404, "Selfie not found")

    # extraction
    face1 = extract_face(image, DOC_FACE)
    face2 = extract_face(selfie, SELFIE_FACE)

    if face1 is None:
        raise HTTPException(400, "No face in image")

    if face2 is None:
        raise HTTPException(400, "No face in selfie")

    result = verify_faces(face1, face2)

    return result