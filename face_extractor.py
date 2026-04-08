from deepface import DeepFace
import cv2
import os
import numpy as np

TARGET_SIZE = (160, 160)


def add_margin(face, margin=0.2):
    """
    Add margin around the face (important for embeddings)
    """
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


def enhance_document_face(face_bgr):
    """
    LIGHT enhancement for document (avoid over-processing)
    """
    # CLAHE (contrast)
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    face_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return face_bgr


def enhance_selfie_face(face_bgr):
    """
    Very light smoothing (keep details)
    """
    return cv2.bilateralFilter(face_bgr, 3, 20, 20)


def extract_face(image_path: str, output_path: str, is_document: bool = False):
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            enforce_detection=True,
            align=True
        )

        if len(faces) == 0:
            print("❌ Aucun visage détecté")
            return None

        # 🔥 Take biggest face
        faces = sorted(
            faces,
            key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"],
            reverse=True
        )

        face = faces[0]["face"]

        # 🔥 Convert float RGB → uint8
        face = (face * 255).clip(0, 255).astype("uint8")

        # 🔥 RGB → BGR
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

        # 🔥 Add margin (VERY IMPORTANT)
        face = add_margin(face, margin=0.2)

        # 🔥 Resize properly
        face = cv2.resize(face, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        # 🔥 Enhancement (light only)
        if is_document:
            face = enhance_document_face(face)
        else:
            face = enhance_selfie_face(face)

        # 🔥 Ensure folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 🔥 Save
        success = cv2.imwrite(output_path, face)

        if not success:
            print(f"❌ Failed to save: {output_path}")
            return None

        print(f"✅ Face saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error extracting face from {image_path}: {e}")
        return None