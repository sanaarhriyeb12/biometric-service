import cv2
import numpy as np


def compute_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def compute_texture_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def check_liveness(image_path: str):
    try:
        img = cv2.imread(image_path)

        if img is None:
            return {"error": "image not readable"}

        blur = compute_blur(img)
        brightness = compute_brightness(img)
        texture = compute_texture_score(img)

        score = 0

        if blur > 100:
            score += 1

        if 40 < brightness < 200:
            score += 1

        if texture > 30:
            score += 1

        decision = "REAL" if score >= 2 else "FAKE"

        return {
            "liveness": decision,
            "score": score,
            "details": {
                "blur": round(blur, 2),
                "brightness": round(brightness, 2),
                "texture": round(texture, 2)
            }
        }

    except Exception as e:
        return {"error": str(e)}