from deepface import DeepFace
import cv2
import os

TARGET_SIZE = (160, 160)

def add_margin(face, margin=0.2):
    h, w, _ = face.shape
    pad_h = int(h * margin)
    pad_w = int(w * margin)

    return cv2.copyMakeBorder(
        face,
        pad_h, pad_h,
        pad_w, pad_w,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

def extract_face(image_path: str, output_path: str):
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            enforce_detection=True,
            align=True
        )

        if len(faces) == 0:
            return None

        faces = sorted(
            faces,
            key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"],
            reverse=True
        )

        face = faces[0]["face"]
        face = (face * 255).clip(0, 255).astype("uint8")
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

        face = add_margin(face, margin=0.2)
        face = cv2.resize(face, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        success = cv2.imwrite(output_path, face)
        if not success:
            return None

        return output_path

    except Exception as e:
        print("Extraction error:", e)
        return None