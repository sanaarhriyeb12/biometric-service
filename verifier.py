from deepface import DeepFace
import os

MODEL_NAME = "Facenet512"
THRESHOLD = 0.68


def distance_to_score(distance: float, threshold: float) -> float:
    """
    Convertit la distance DeepFace en score métier :
    - distance <= threshold  -> score entre 90 et 100
    - distance > threshold   -> score < 90
    """
    if distance <= threshold:
        score = 100 - (distance / threshold) * 10
    else:
        score = 90 - ((distance - threshold) / threshold) * 100

    return round(max(0, min(100, score)), 2)


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
            detector_backend="skip",   # car les visages sont déjà extraits
            enforce_detection=False,
            align=False,
            normalization="base"
        )

        distance = float(result["distance"])
        score = distance_to_score(distance, THRESHOLD)

        verified = score >= 90
        decision = "ACCEPTED" if verified else "REJECTED"

        return {
            "verified": verified,
            "distance": round(distance, 6),
            "score": score,
            "decision": decision,
            "model": MODEL_NAME,
            "threshold": THRESHOLD
        }

    except Exception as e:
        return {"error": str(e)}