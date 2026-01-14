ALPR-MV-PROJECT-
# Automatic License Plate Recognition (ALPR) System
## 1. Project Overview
This project is a GUI-based Machine Vision system designed to detect and recognize Malaysian vehicle license plates. It utilizes a fine-tuned **YOLOv11** model for detection and **EasyOCR** for character recognition, integrated into a pipeline that addresses environmental challenges like low light and motion blur.

## 2. Methodology & Innovation

### **A. System Pipeline**
`Input` $\rightarrow$ `YOLOv11 Detection` $\rightarrow$ `Cropping (+10px padding)` $\rightarrow$ `Pre-processing` $\rightarrow$ `EasyOCR` $\rightarrow$ `Output`

### **B. Key Innovations**
1.  **Model Fine-Tuning:**
    * **Base Model:** YOLOv11
    * **Dataset:** Custom dataset of **158 images** (Malaysian plates).
    * **Training:** Trained for **100 epochs** via Roboflow.
    * **Inference Settings:** Confidence Threshold = 0.1.

2.  **Advanced Pre-processing (The "Secret Sauce"):**
    Before OCR reads the text, the plate image undergoes:
    * **Upscaling:** Increases resolution for better clarity.
    * **CLAHE:** Adaptive histogram equalization to fix lighting/glare.
    * **Sharpening:** Enhances character edges.
    * **Otsu's Thresholding:** Converts to binary (black & white) to remove noise.

3.  **Post-Processing:**
    * **Regex Filtering:** Removes special characters, keeping only A-Z and 0-9.
    * **Spatial Sorting:** Reads text in correct Y-axis order.

## 3. Results
* **Detection Rate:** **85%** (Tested on Week 12 Dataset).
* **Strengths:** Robust detection on clear Malaysian plates; handles angled views well.
* **Weaknesses:** OCR accuracy drops on extremely grainy night images or unusual vehicles (e.g., trucks).

---

## 4. How to Run (Step-by-Step)

### **Step 1: Prerequisites**
Ensure you have the following files in the same folder:
* `ALPR.py` (The main application code)
* `ALPR.pt` (The YOLOv11 model file)

### **Step 2: Install Libraries**
Open your terminal and run:
```bash
pip install opencv-python easyocr ultralytics numpy pillow
```
### **Step 3: Launch**
Open your terminal and run:
```bash
python ALPR.py
```
1. A GUI window will appear.
2. Click "Select Image Folder".
3. Choose the folder containing your test images.
4. View results live!

## 5.Tools Used
-Frameworks: Python, OpenCV, PyTorch.
-Models: Ultralytics YOLOv11, EasyOCR.
-Training Platform: Roboflow.
-Editor: VS Code.

cat << 'EOF' >> README.md

## 6. Project Poster
You can check our poster below:

![ALPR Poster](ALPR_POSTER.png)
EOF



