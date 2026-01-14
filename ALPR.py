import cv2
import easyocr
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO
import re

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.1   # Min confidence for EasyOCR text
YOLO_MODEL_PATH = 'ALPR.pt'  # <--- YOUR MODEL PATH
GPU_ENABLED = True           # Set False if you don't have an NVIDIA GPU

class ALPRSystem:
    def __init__(self):
        print("1. Loading YOLO Model...")
        try:
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load '{YOLO_MODEL_PATH}'.\nMake sure the file is in the same folder!")
            raise e

        print("2. Loading EasyOCR Model... (This may take a moment)")
        self.reader = easyocr.Reader(['en'], gpu=GPU_ENABLED)
        self.results_data = [] 

    def preprocess_plate(self, plate_img):
        """Advanced preprocessing to handle blur, dark lighting, and noise."""
        if plate_img.size == 0: return plate_img

        # 1. Grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # 2. Lighting Correction (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # 3. Upscaling (3x Zoom)
        scale_percent = 300 
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)

        # 4. Sharpening
        kernel = np.array([[0, -1, 0], 
                           [-1, 5,-1], 
                           [0, -1, 0]])
        sharpened = cv2.filter2D(resized, -1, kernel)

        # 5. Thresholding (Otsu)
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def process_folder(self, folder_path, update_callback):
        self.results_data = []
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        total_files = len(files)
        detected_count = 0
        
        # Define allowed characters (A-Z and 0-9 only)
        # This helps EasyOCR focus and improves accuracy significantly
        ALLOWED_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        for idx, filename in enumerate(files):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None: continue

            # --- STEP 1: YOLO DETECTION ---
            yolo_results = self.yolo_model(img, verbose=False)
            
            display_img = img.copy()
            found_plates_text = [] 
            found_plates_conf = []

            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Padding
                h, w, _ = img.shape
                padding = 10
                x1_p = max(0, x1 - padding)
                y1_p = max(0, y1 - padding)
                x2_p = min(w, x2 + padding)
                y2_p = min(h, y2 + padding)
                
                plate_crop = img[y1_p:y2_p, x1_p:x2_p]
                
                # --- STEP 2: PREPROCESSING ---
                processed_crop = self.preprocess_plate(plate_crop)

                # --- STEP 3: EASYOCR RECOGNITION (TUNED) ---
                # allowlist: Only looks for A-Z, 0-9
                # mag_ratio=1.5: Zooms in 1.5x more to find small text
                ocr_result = self.reader.readtext(
                    processed_crop, 
                    detail=1, 
                    allowlist=ALLOWED_CHARS, 
                    mag_ratio=1.5
                )

                plate_text_final = ""
                plate_conf_final = 0.0

                if ocr_result:
                    valid_results = [r for r in ocr_result if r[2] > CONFIDENCE_THRESHOLD]
                    
                    if valid_results:
                        # Sort Top-to-Bottom, then Left-to-Right
                        valid_results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
                        
                        # Join text
                        full_text_list = [res[1] for res in valid_results]
                        raw_text = "".join(full_text_list).upper()
                        
                        # --- REMOVE SPECIAL CHARACTERS (Safety Net) ---
                        # Keep only A-Z and 0-9
                        plate_text_final = re.sub(r'[^A-Z0-9]', '', raw_text)
                        
                        # Avg Confidence
                        confs = [res[2] for res in valid_results]
                        plate_conf_final = sum(confs) / len(confs)

                        # Draw Box (Green)
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Draw Text Background
                        (tw, th), _ = cv2.getTextSize(plate_text_final, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(display_img, (x1, y1 - 35), (x1 + tw, y1), (0, 255, 0), -1)
                        
                        # Draw Text
                        cv2.putText(display_img, plate_text_final, 
                                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.8, (0, 0, 0), 2)
                        
                        if plate_text_final:
                            found_plates_text.append(plate_text_final)
                            found_plates_conf.append(plate_conf_final)
                
                # If YOLO found a box, but OCR failed
                if not plate_text_final:
                     cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box

            if len(found_plates_text) > 0:
                detected_count += 1
            
            display_text = "\n".join(found_plates_text) if found_plates_text else "No Plate Found"
            avg_conf = sum(found_plates_conf) / len(found_plates_conf) if found_plates_conf else 0.0

            self.results_data.append({
                "filename": filename,
                "text": display_text,
                "confidence": avg_conf,
                "display_img": display_img
            })

            update_callback(idx + 1, total_files)

        detection_rate = (detected_count / total_files * 100) if total_files > 0 else 0
        return detection_rate, total_files

class ALPR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ALPR System (Multi-Car Support)")
        self.root.geometry("1100x750")
        
        self.alpr = ALPRSystem()
        self.current_index = 0
        
        self.create_widgets()

    def create_widgets(self):
        # 1. Top Control Panel
        frame_controls = tk.Frame(self.root, pady=10, bg="#e1e1e1")
        frame_controls.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_load = tk.Button(frame_controls, text="Select Image Folder", command=self.load_folder, 
                                  bg="white", font=("Arial", 10, "bold"), height=2, width=20)
        self.btn_load.pack(side=tk.LEFT, padx=20, pady=5)

        self.lbl_status = tk.Label(frame_controls, text="Status: Ready to load", bg="#e1e1e1", font=("Arial", 11))
        self.lbl_status.pack(side=tk.LEFT, padx=20)

        self.lbl_stats = tk.Label(frame_controls, text="Detection Rate: --%", 
                                  bg="#e1e1e1", fg="blue", font=("Arial", 12, "bold"))
        self.lbl_stats.pack(side=tk.RIGHT, padx=20)

        # 2. Main Image Area
        frame_image = tk.Frame(self.root, bg="#2b2b2b")
        frame_image.pack(expand=True, fill=tk.BOTH)
        
        self.lbl_img_display = tk.Label(frame_image, bg="#2b2b2b", text="No Image Loaded", fg="white")
        self.lbl_img_display.pack(expand=True)

        # 3. Bottom Info Panel
        frame_info = tk.Frame(self.root, height=150, bg="white", bd=2, relief=tk.RAISED) # Increased height
        frame_info.pack(side=tk.BOTTOM, fill=tk.X)

        lbl_plate_title = tk.Label(frame_info, text="License Plate(s):", bg="white", fg="#555", font=("Arial", 10))
        lbl_plate_title.place(x=20, y=10)
        
        # Result text label (Supports multiple lines now)
        self.lbl_result_text = tk.Label(frame_info, text="----", bg="white", fg="black", 
                                        font=("Arial", 18, "bold"), justify=tk.LEFT)
        self.lbl_result_text.place(x=20, y=35)
        
        self.lbl_conf = tk.Label(frame_info, text="Avg Confidence: --", bg="white", fg="#555", font=("Arial", 10))
        self.lbl_conf.place(x=20, y=110)

        # Navigation Buttons
        self.btn_prev = tk.Button(frame_info, text="< Previous", command=self.prev_img, state=tk.DISABLED, width=15, height=2)
        self.btn_prev.pack(side=tk.RIGHT, padx=20, pady=40)
        
        self.btn_next = tk.Button(frame_info, text="Next >", command=self.next_img, state=tk.DISABLED, width=15, height=2)
        self.btn_next.pack(side=tk.RIGHT, padx=5, pady=40)

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        self.lbl_status.config(text="Initializing...")
        self.btn_load.config(state=tk.DISABLED)
        threading.Thread(target=self.run_processing, args=(folder_path,)).start()

    def run_processing(self, folder_path):
        detection_rate, total = self.alpr.process_folder(folder_path, self.update_progress)
        self.root.after(0, lambda: self.finish_processing(detection_rate, total))

    def update_progress(self, current, total):
        self.lbl_status.config(text=f"Processing image {current} of {total}...")

    def finish_processing(self, detection_rate, total):
        self.lbl_status.config(text=f"Completed: {total} images processed.")
        self.lbl_stats.config(text=f"Detection Rate: {detection_rate:.1f}%")
        self.btn_load.config(state=tk.NORMAL)
        if self.alpr.results_data:
            self.current_index = 0
            self.update_display()
            self.btn_next.config(state=tk.NORMAL)
            self.btn_prev.config(state=tk.NORMAL)
        else:
            messagebox.showinfo("Info", "No valid images found.")

    def update_display(self):
        data = self.alpr.results_data[self.current_index]
        
        self.lbl_result_text.config(text=data['text'], fg="green" if "No Plate" not in data['text'] else "red")
        self.lbl_conf.config(text=f"Avg Confidence: {data['confidence']:.2f}")
        self.root.title(f"ALPR - Image {self.current_index + 1} / {len(self.alpr.results_data)} - {data['filename']}")

        img_rgb = cv2.cvtColor(data['display_img'], cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height() - 250
        img_w, img_h = img_pil.size
        ratio = min(win_w/img_w, win_h/img_h)
        new_size = (int(img_w * ratio), int(img_h * ratio))
        img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        self.lbl_img_display.config(image=img_tk)
        self.lbl_img_display.image = img_tk 

    def next_img(self):
        if self.current_index < len(self.alpr.results_data) - 1:
            self.current_index += 1
            self.update_display()

    def prev_img(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = ALPR_GUI(root)
    root.mainloop()
