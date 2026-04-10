import cv2
import numpy as np


def compute_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])


def compute_texture_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)


def check_liveness(image_path: str):
    try:
        img = cv2.imread(image_path)

        if img is None:
            return {"error": "image not readable"}

        blur = compute_blur(img)
        brightness = compute_brightness(img)
        texture = compute_texture_score(img)

        #  LOGIQUE SIMPLE MAIS EFFICACE
        score = 0

        # Netteté
        if blur > 100:
            score += 1

        # Luminosité
        if 40 < brightness < 200:
            score += 1

        # Texture (écran/photo souvent trop lisse)
        if texture > 30:
            score += 1

        #  Décision
        if score >= 2:
            decision = "REAL"
        else:
            decision = "FAKE"

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