from deepface import DeepFace
import cv2
import os


PRIMARY_MODEL = "Facenet512"
SECONDARY_MODEL = "ArcFace"

# More tolerant than the raw DeepFace threshold because of poor document quality
PRIMARY_ACCEPT_FACTOR = 1.25
PRIMARY_REVIEW_FACTOR = 1.80

SECONDARY_ACCEPT_FACTOR = 1.10
SECONDARY_REVIEW_FACTOR = 1.35

VERY_LOW_QUALITY_BLUR = 12.0


def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def run_model(img1_path: str, img2_path: str, model_name: str, accept_factor: float, review_factor: float):
    result = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend="skip",   # faces are already extracted and aligned
        enforce_detection=False,
        align=False,
        normalization="base"
    )

    distance = float(result.get("distance", 999.0))
    base_threshold = float(result.get("threshold", 0.5))

    accept_threshold = base_threshold * accept_factor
    review_threshold = base_threshold * review_factor

    if distance <= accept_threshold:
        decision = "ACCEPTED"
    elif distance <= review_threshold:
        decision = "REVIEW"
    else:
        decision = "REJECTED"

    # Confidence relative to the review threshold
    score = max(0.0, min(100.0, (1.0 - (distance / review_threshold)) * 100.0))

    return {
        "model": model_name,
        "verified_by_deepface": bool(result.get("verified", False)),
        "distance": round(distance, 4),
        "base_threshold": round(base_threshold, 4),
        "accept_threshold": round(accept_threshold, 4),
        "review_threshold": round(review_threshold, 4),
        "score": round(score, 2),
        "decision": decision
    }


def verify_faces(img1_path: str, img2_path: str):
    debug = {}

    try:
        # File existence
        debug["doc_exists"] = os.path.exists(img1_path)
        debug["selfie_exists"] = os.path.exists(img2_path)

        if not debug["doc_exists"] or not debug["selfie_exists"]:
            return {
                "error": "One or both extracted face images are missing",
                "debug": debug
            }

        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        debug["doc_readable"] = img1 is not None
        debug["selfie_readable"] = img2 is not None

        if img1 is None or img2 is None:
            return {
                "error": "Failed to read one or both extracted face images",
                "debug": debug
            }

        debug["doc_shape"] = list(img1.shape)
        debug["selfie_shape"] = list(img2.shape)

        debug["doc_blur_score"] = round(blur_score(img1), 2)
        debug["selfie_blur_score"] = round(blur_score(img2), 2)

        debug["doc_brightness"] = round(brightness_score(img1), 2)
        debug["selfie_brightness"] = round(brightness_score(img2), 2)

        # Run primary model
        primary = run_model(
            img1_path,
            img2_path,
            PRIMARY_MODEL,
            PRIMARY_ACCEPT_FACTOR,
            PRIMARY_REVIEW_FACTOR
        )

        # Run secondary model
        secondary = run_model(
            img1_path,
            img2_path,
            SECONDARY_MODEL,
            SECONDARY_ACCEPT_FACTOR,
            SECONDARY_REVIEW_FACTOR
        )

        # Final decision logic
        primary_decision = primary["decision"]
        secondary_decision = secondary["decision"]

        if primary_decision == "ACCEPTED":
            overall_decision = "ACCEPTED"
        elif primary_decision == "REVIEW":
            overall_decision = "REVIEW"
        elif secondary_decision in ("ACCEPTED", "REVIEW"):
            overall_decision = "REVIEW"
        else:
            overall_decision = "REJECTED"

        # If image quality is extremely poor, downgrade ACCEPTED to REVIEW
        if overall_decision == "ACCEPTED":
            if (
                debug["doc_blur_score"] < VERY_LOW_QUALITY_BLUR
                or debug["selfie_blur_score"] < VERY_LOW_QUALITY_BLUR
            ):
                overall_decision = "REVIEW"

        weighted_score = (primary["score"] * 0.7) + (secondary["score"] * 0.3)

        return {
            "verified": overall_decision == "ACCEPTED",
            "decision": overall_decision,
            "score": round(weighted_score, 2),
            "distance": primary["distance"],  # primary model distance
            "primary_model": PRIMARY_MODEL,
            "secondary_model": SECONDARY_MODEL,
            "models": {
                "primary": primary,
                "secondary": secondary
            },
            "debug": debug
        }

    except Exception as e:
        return {
            "error": str(e),
            "debug": debug
        }