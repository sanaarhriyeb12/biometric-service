from deepface import DeepFace
import os

MODEL_NAME = "Facenet512"

# Seuils métier
ACCEPT_THRESHOLD = 0.45
REVIEW_THRESHOLD = 0.67


def distance_to_score(distance: float) -> float:
    """
    Convert distance to business score:
    - distance < 0.45  -> score 90 to 99
    - 0.45 to 0.67     -> score 50 to 89
    - >= 0.67          -> score 0
    """
    if distance < ACCEPT_THRESHOLD:
        # Map [0 .. 0.45] -> [99 .. 90]
        score = 99 - (distance / ACCEPT_THRESHOLD) * 9
        return round(max(90, min(99, score)), 2)

    if distance < REVIEW_THRESHOLD:
        # Map [0.45 .. 0.67] -> [89 .. 50]
        ratio = (distance - ACCEPT_THRESHOLD) / (REVIEW_THRESHOLD - ACCEPT_THRESHOLD)
        score = 89 - ratio * 39
        return round(max(50, min(89, score)), 2)

    return 0.0


def get_decision(distance: float) -> str:
    if distance < ACCEPT_THRESHOLD:
        return "ACCEPTED"
    elif distance < REVIEW_THRESHOLD:
        return "REVIEW"
    else:
        return "REJECTED"


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
            normalization="base"
        )

        distance = float(result["distance"])
        decision = get_decision(distance)
        score = distance_to_score(distance)
        if decision == "REJECTED":
            score = 0.0

        return {
            "verified": decision == "ACCEPTED",
            "distance": round(distance, 6),
            "score": score,
            "decision": decision,
            "model": MODEL_NAME,
            "accept_threshold": ACCEPT_THRESHOLD,
            "review_threshold": REVIEW_THRESHOLD
        }

    except Exception as e:
        return {"error": str(e)}