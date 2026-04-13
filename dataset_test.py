import os
from face_extractor import extract_face
from verifier import verify_faces

DATASET = "dataset"
OUTPUT = "output_faces"

os.makedirs(OUTPUT, exist_ok=True)

total = 0
accepted = 0

for person in os.listdir(DATASET):
    person_path = os.path.join(DATASET, person)

    if not os.path.isdir(person_path):
        continue

    image_path = os.path.join(person_path, "image.jpg")

    if not os.path.exists(image_path):
        continue

    image_face = extract_face(image_path, f"{OUTPUT}/{person}_image.jpg")

    if image_face is None:
        continue

    for file in os.listdir(person_path):
        if "selfie" in file.lower():
            selfie_path = os.path.join(person_path, file)

            selfie_face = extract_face(
                selfie_path,
                f"{OUTPUT}/{person}_{file}"
            )

            if selfie_face is None:
                continue

            result = verify_faces(image_face, selfie_face)

            total += 1

            if result["decision"] == "ACCEPTED":
                accepted += 1

            print(f"{person} vs {file} → {result['score']}% → {result['decision']}")

# 📊 RESULTATS
print("\n====================")
print("TOTAL TESTS:", total)
print("ACCEPTED:", accepted)

if total > 0:
    accuracy = (accepted / total) * 100
    print("ACCURACY:", round(accuracy, 2), "%")