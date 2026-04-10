from fastapi import FastAPI, HTTPException, Query
from deepface import DeepFace
import os

app = FastAPI()

# ======================
# CONFIG
# ======================
MODEL_NAME = "ArcFace"
THRESHOLD = 0.68  # stable pour ArcFace

# ======================
# VERIFY ENDPOINT
# ======================
@app.get("/verify")
def verify(image: str = Query(...), selfie: str = Query(...)):
    try:
        # check files exist
        if not os.path.exists(image):
            raise HTTPException(status_code=400, detail="Image not found")
        if not os.path.exists(selfie):
            raise HTTPException(status_code=400, detail="Selfie not found")

        # DeepFace verification
        result = DeepFace.verify(
            img1_path=selfie,
            img2_path=image,
            model_name=MODEL_NAME,
            enforce_detection=True
        )

        distance = float(result["distance"])

        # ======================
        # DECISION LOGIC FIXED
        # ======================
        verified = distance < THRESHOLD

        if verified:
            decision = "ACCEPTED"
            score = round((1 - distance) * 100, 2)
        else:
            decision = "REJECTED"
            score = round((1 - distance) * 100, 2)

        return {
            "verified": verified,
            "decision": decision,
            "distance": distance,
            "score": score,
            "model": MODEL_NAME,
            "threshold": THRESHOLD
        }

    except Exception as e:
        return {
            "verified": False,
            "decision": "ERROR",
            "message": str(e)
        }