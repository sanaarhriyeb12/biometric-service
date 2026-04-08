import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


#  indices yeux (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]

    # vertical distances
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))

    # horizontal distance
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (A + B) / (2.0 * C)
    return ear


def liveness_blink():
    cap = cv2.VideoCapture(1)

    blink_count = 0
    EAR_THRESHOLD = 0.20
    CLOSED_FRAMES = 0

    with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:

                    left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
                    right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)

                    ear = (left_ear + right_ear) / 2.0

                    if ear < EAR_THRESHOLD:
                        CLOSED_FRAMES += 1
                    else:
                        if CLOSED_FRAMES > 2:
                            blink_count += 1
                        CLOSED_FRAMES = 0

                    cv2.putText(frame, f"Blinks: {blink_count}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.putText(frame, f"EAR: {round(ear,2)}", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Liveness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    #  décision finale
    if blink_count >= 1:
        return {"liveness": "REAL", "blinks": blink_count}
    else:
        return {"liveness": "FAKE", "blinks": blink_count}