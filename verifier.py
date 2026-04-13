from deepface import DeepFace
import os

MODEL_NAME = "ArcFace"
THRESHOLD = 0.68


def distance_to_score(distance: float, threshold: float) -> float:
    """
    Business score:
    - if accepted => score between 90 and 99
    - if rejected => score = 0
    """
    if distance < threshold:
        score = 90 + (1 - distance / threshold) * 9
        return round(min(score, 99), 2)
    return 0.0


def verify_faces(image_path: str, selfie_path: str):
    if not os.path.exists(image_path):
        return {"error": "image not found"}

    if not os.path.exists(selfie_path):
        return {"error": "selfie not found"}

    try:
        result = DeepFace.verify(
            img1_path=selfie_path,
            img2_path=image_path,
            model_name=MODEL_NAME,
            detector_backend="skip",   # faces already extracted
            enforce_detection=False,
            align=False,
            normalization="ArcFace"    # important for ArcFace
        )

        distance = float(result["distance"])
        verified = distance < THRESHOLD
        score = distance_to_score(distance, THRESHOLD)

        return {
            "verified": verified,
            "distance": round(distance, 6),
            "score": score,
            "decision": "ACCEPTED" if verified else "REJECTED",
            "model": MODEL_NAME,
            "threshold": THRESHOLD
        }

    except Exception as e:
        return {"error": str(e)}