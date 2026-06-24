import os
import matplotlib.pyplot as plt
import numpy as np
from verifier import verify_faces

OUTPUT_DIR = "output_faces"
ARTIFACT_DIR = r"C:\Users\marwa\.gemini\antigravity\brain\3d9f8a0d-614a-4f41-8aa0-c99e1cdf0f47"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
PLOT_PATH = os.path.join(ARTIFACT_DIR, "evaluation_curves.png")

# Find all images and selfies in output_faces
files = os.listdir(OUTPUT_DIR)
images = {}
selfies = {}

for f in files:
    if f.endswith("_image.jpg"):
        # e.g., person_1_image.jpg -> person_1
        person_name = f.replace("_image.jpg", "")
        images[person_name] = os.path.join(OUTPUT_DIR, f)
    elif "selfie" in f:
        # e.g., person_1_selfie.jpg -> person_1
        # person_2_selfie1.jpg -> person_2
        parts = f.split("_")
        person_name = parts[0] + "_" + parts[1]
        if person_name not in selfies:
            selfies[person_name] = []
        selfies[person_name].append(os.path.join(OUTPUT_DIR, f))

# Let's perform all pairwise comparisons
pos_distances = []
neg_distances = []
all_results = []

print("Running pairwise face verification comparisons...")
for img_person, img_path in images.items():
    for selfie_person, selfie_paths in selfies.items():
        is_match = (img_person == selfie_person)
        for selfie_path in selfie_paths:
            res = verify_faces(img_path, selfie_path)
            if "error" in res:
                print(f"Error verifying {img_person} vs {selfie_path}: {res['error']}")
                continue
            dist = res["distance"]
            if is_match:
                pos_distances.append(dist)
            else:
                neg_distances.append(dist)
            
            all_results.append({
                "img_person": img_person,
                "selfie_person": selfie_person,
                "selfie_file": os.path.basename(selfie_path),
                "distance": dist,
                "is_match": is_match
            })

print(f"Completed! Found {len(pos_distances)} positive pairs and {len(neg_distances)} negative pairs.")

pos_distances = np.array(pos_distances)
neg_distances = np.array(neg_distances)

# Sweep thresholds to compute metrics
thresholds = np.linspace(0.0, 1.2, 200)
accuracies = []
far_list = []
frr_list = []
tpr_list = []
fpr_list = []

for t in thresholds:
    # Match if distance < t
    tp = np.sum(pos_distances < t)
    fn = np.sum(pos_distances >= t)
    fp = np.sum(neg_distances < t)
    tn = np.sum(neg_distances >= t)
    
    tpr = tp / len(pos_distances) if len(pos_distances) > 0 else 0
    fnr = fn / len(pos_distances) if len(pos_distances) > 0 else 0
    fpr = fp / len(neg_distances) if len(neg_distances) > 0 else 0
    tnr = tn / len(neg_distances) if len(neg_distances) > 0 else 0
    
    accuracy = (tp + tn) / (len(pos_distances) + len(neg_distances))
    
    accuracies.append(accuracy)
    far_list.append(fpr)  # FAR is FPR
    frr_list.append(fnr)  # FRR is FNR
    tpr_list.append(tpr)
    fpr_list.append(fpr)

accuracies = np.array(accuracies)
far_list = np.array(far_list)
frr_list = np.array(frr_list)
tpr_list = np.array(tpr_list)
fpr_list = np.array(fpr_list)

# Find optimal threshold (maximum accuracy)
opt_idx = np.argmax(accuracies)
opt_threshold = thresholds[opt_idx]
max_accuracy = accuracies[opt_idx]

# Find Equal Error Rate (EER) threshold (where FAR and FRR are closest)
eer_idx = np.argmin(np.abs(far_list - frr_list))
eer_threshold = thresholds[eer_idx]
eer_val = (far_list[eer_idx] + frr_list[eer_idx]) / 2

print(f"Optimal Threshold (Max Accuracy): {opt_threshold:.4f} (Accuracy: {max_accuracy*100:.2f}%)")
print(f"EER Threshold (FAR=FRR): {eer_threshold:.4f} (EER: {eer_val*100:.2f}%)")

# Let's plot 4 curves in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Courbes d'Évaluation du Service Biométrique (Facenet512)", fontsize=16, fontweight='bold', color='#1A365D')

# Plot 1: Distance Distribution
axs[0, 0].hist(pos_distances, bins=10, alpha=0.6, label='Matches (Même personne)', color='#3182CE', edgecolor='black')
axs[0, 0].hist(neg_distances, bins=15, alpha=0.6, label='Non-Matches (Personnes différentes)', color='#E53E3E', edgecolor='black')
axs[0, 0].axvline(opt_threshold, color='#2D3748', linestyle='--', linewidth=2, label=f'Seuil Optimal ({opt_threshold:.2f})')
axs[0, 0].set_title("Distribution des Distances", fontsize=12, fontweight='semibold')
axs[0, 0].set_xlabel("Distance (Facenet512)")
axs[0, 0].set_ylabel("Nombre de Paires")
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle=':', alpha=0.6)

# Plot 2: ROC Curve
axs[0, 1].plot(fpr_list, tpr_list, color='#805AD5', linewidth=3, label=f'ROC Curve')
axs[0, 1].plot([0, 1], [0, 1], color='#A0AEC0', linestyle='--', linewidth=1.5)
axs[0, 1].scatter(fpr_list[opt_idx], tpr_list[opt_idx], color='#D69E2E', s=100, zorder=5, label=f'Seuil Optimal ({opt_threshold:.2f})')
# Calculate AUC using trapezoid rule
auc = np.trapz(tpr_list, fpr_list)
axs[0, 1].set_title(f"Courbe ROC (AUC = {auc:.4f})", fontsize=12, fontweight='semibold')
axs[0, 1].set_xlabel("Taux de Faux Positifs (FPR / FAR)")
axs[0, 1].set_ylabel("Taux de Vrais Positifs (TPR / Rappel)")
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle=':', alpha=0.6)

# Plot 3: FAR and FRR vs Threshold
axs[1, 0].plot(thresholds, far_list, color='#DD6B20', linewidth=2.5, label='FAR (Taux Fausse Accept.)')
axs[1, 0].plot(thresholds, frr_list, color='#319795', linewidth=2.5, label='FRR (Taux Faux Rejet)')
axs[1, 0].axvline(eer_threshold, color='#E53E3E', linestyle=':', linewidth=2, label=f'Seuil EER ({eer_threshold:.2f})')
axs[1, 0].set_title(f"FAR & FRR vs Seuil (EER = {eer_val*100:.1f}%)", fontsize=12, fontweight='semibold')
axs[1, 0].set_xlabel("Seuil de Distance")
axs[1, 0].set_ylabel("Taux d'Erreur")
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle=':', alpha=0.6)

# Plot 4: Accuracy vs Threshold
axs[1, 1].plot(thresholds, accuracies * 100, color='#38A169', linewidth=2.5, label='Précision (Accuracy)')
axs[1, 1].scatter(opt_threshold, max_accuracy * 100, color='#D69E2E', s=100, zorder=5, label=f'Max ({max_accuracy*100:.1f}%)')
axs[1, 1].set_title("Précision Globale vs Seuil", fontsize=12, fontweight='semibold')
axs[1, 1].set_xlabel("Seuil de Distance")
axs[1, 1].set_ylabel("Précision (%)")
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"Saved evaluation curves plot to: {PLOT_PATH}")

# Save detailed results to a text report in the same artifact dir
REPORT_PATH = os.path.join(ARTIFACT_DIR, "evaluation_report.md")
with open(REPORT_PATH, "w", encoding="utf-8") as rf:
    rf.write("# Rapport d'Évaluation du Service Biométrique\n\n")
    rf.write(f"- **Modèle de Face Recognition** : `{verify_faces('','').get('model', 'Facenet512')}`\n")
    rf.write(f"- **Nombre de Paires Positives** (Même personne) : {len(pos_distances)}\n")
    rf.write(f"- **Nombre de Paires Négatives** (Personnes différentes) : {len(neg_distances)}\n")
    rf.write(f"- **Seuil Optimal de Distance** (Max Accuracy) : `{opt_threshold:.3f}`\n")
    rf.write(f"- **Précision Maximale** : `{max_accuracy*100:.2f}%`\n")
    rf.write(f"- **Seuil EER (Equal Error Rate)** : `{eer_threshold:.3f}`\n")
    rf.write(f"- **Equal Error Rate (EER)** : `{eer_val*100:.2f}%`\n")
    rf.write(f"- **Aire Sous la Courbe (AUC)** : `{auc:.4f}`\n\n")
    
    rf.write("## Performances pour Différents Seuils Clés\n\n")
    rf.write("| Seuil | FAR (%) | FRR (%) | Précision Globale (%) | Décision Métier |\n")
    rf.write("|---|---|---|---|---|\n")
    for t in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.67, 0.7, 0.8]:
        idx = np.argmin(np.abs(thresholds - t))
        far_pct = far_list[idx] * 100
        frr_pct = frr_list[idx] * 100
        acc_pct = accuracies[idx] * 100
        # Determine business meaning
        if t < 0.45:
            decision = "ACCEPTATION stricte"
        elif t < 0.67:
            decision = "REVUE manuelle"
        else:
            decision = "REJET strict"
        rf.write(f"| {t:.2f} | {far_pct:.1f}% | {frr_pct:.1f}% | {acc_pct:.1f}% | {decision} |\n")
        
    rf.write("\n## Détail de toutes les Paires Testées\n\n")
    rf.write("| Image Document | Selfie | Type de Paire | Distance Facenet512 | Décision Métier (Seuil 0.45/0.67) |\n")
    rf.write("|---|---|---|---|---|\n")
    for res in sorted(all_results, key=lambda x: (not x["is_match"], x["img_person"], x["selfie_file"])):
        pair_type = "**Match (Positif)**" if res["is_match"] else "Non-Match (Négatif)"
        dist = res["distance"]
        
        # Decide based on thresholds
        if dist < 0.45:
            decision = "✅ ACCEPTED"
        elif dist < 0.67:
            decision = "⚠️ REVIEW"
        else:
            decision = "❌ REJECTED"
            
        rf.write(f"| {res['img_person']}_image.jpg | {res['selfie_file']} | {pair_type} | {dist:.4f} | {decision} |\n")

print(f"Saved evaluation report to: {REPORT_PATH}")
