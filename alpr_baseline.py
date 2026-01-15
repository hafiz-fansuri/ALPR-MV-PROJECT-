import os
import cv2
import easyocr

# -----------------------------
# Paths
# -----------------------------
INPUT_DIR = "ALPR/testAll"
OUTPUT_DIR = "ALPR/testAll/output_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Initialize EasyOCR
# -----------------------------
reader = easyocr.Reader(['en'], gpu=True)

# -----------------------------
# Detection Statistics
# -----------------------------
total_images = 0
detected_images = 0

# -----------------------------
# Process Images
# -----------------------------
for img_name in os.listdir(INPUT_DIR):

    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    total_images += 1

    img_path = os.path.join(INPUT_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    # Resize if image is too large (baseline-safe)
h, w = image.shape[:2]
max_dim = 1000

if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    image = cv2.resize(image, None, fx=scale, fy=scale)


    # EasyOCR
    results = reader.readtext(image)

    # If any text detected → count as detected
    if len(results) > 0:
        detected_images += 1

    # Draw boxes
    for (bbox, text, conf) in results:
        (tl, tr, br, bl) = bbox
        tl = tuple(map(int, tl))
        br = tuple(map(int, br))

        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(
            image,
            text,
            (tl[0], tl[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Save output
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), image)

# -----------------------------
# Detection Rate
# -----------------------------
detection_rate = detected_images / total_images if total_images > 0 else 0

print("===== BASELINE ALPR RESULTS =====")
print(f"Total Images      : {total_images}")
print(f"Detected Images   : {detected_images}")
print(f"Detection Rate    : {detection_rate:.2f}")
