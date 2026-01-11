import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from PIL import Image, ImageTk

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 AI Vision Monitor")
        self.root.geometry("1250x750")
        self.root.configure(bg="#1a1a2e")
        
        # --- Variables ---
        self.cap = None
        self.running = False
        self.model = YOLO("yolov8n.pt")  # Load Standard YOLOv8 Nano model
        self.detected_objects = set()    # To track what we've seen this session
        
        # --- UI Layout ---
        self.create_widgets()
        
    def create_widgets(self):
        # 1. Header
        header = tk.Frame(self.root, bg="#16213e", height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="👁️ AI Vision Monitor (YOLOv8)", 
                 font=("Segoe UI", 24, "bold"), bg="#16213e", fg="#00fff5").pack(pady=10)

        # 2. Main Layout
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # --- Left Side: Camera Feed ---
        left_panel = tk.Frame(main_frame, bg="#0f3460", bd=2, relief=tk.RIDGE)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video Placeholder
        self.video_label = tk.Label(left_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Controls
        btn_frame = tk.Frame(left_panel, bg="#0f3460")
        btn_frame.pack(pady=15)

        self.btn_start = tk.Button(btn_frame, text="▶ Start Detection", font=("Segoe UI", 12, "bold"),
                                   bg="#2ed573", fg="white", width=15, command=self.start_camera)
        self.btn_start.grid(row=0, column=0, padx=5)

        self.btn_stop = tk.Button(btn_frame, text="⏹ Stop", font=("Segoe UI", 12, "bold"),
                                  bg="#ff4757", fg="white", width=15, state=tk.DISABLED, command=self.stop_camera)
        self.btn_stop.grid(row=0, column=1, padx=5)

        # --- Right Side: Dashboard & Stats ---
        right_panel = tk.Frame(main_frame, bg="#16213e", width=350, bd=2, relief=tk.RIDGE)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="📊 Live Analytics", font=("Segoe UI", 14, "bold"), 
                 bg="#16213e", fg="white").pack(pady=15)

        # Stats Cards
        self.lbl_fps = self.create_stat_card(right_panel, "FPS", "#ffa502")
        self.lbl_obj_count = self.create_stat_card(right_panel, "Objects in View", "#2ed573")

        # Detection Log
        tk.Label(right_panel, text="📝 Detection Log", font=("Segoe UI", 12, "bold"), 
                 bg="#16213e", fg="#a4b0be").pack(pady=(20, 5))
        
        list_frame = tk.Frame(right_panel, bg="#0f3460")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_list = tk.Listbox(list_frame, font=("Consolas", 10), bg="#0f3460", fg="white",
                                   yscrollcommand=scrollbar.set, bd=0, highlightthickness=0)
        self.log_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_list.yview)

        # Bottom Buttons
        tk.Button(right_panel, text="🧹 Clear Log", bg="#535c68", fg="white", font=("Segoe UI", 10),
                  command=self.clear_log).pack(pady=15, fill=tk.X, padx=20)

    def create_stat_card(self, parent, title, color):
        frame = tk.Frame(parent, bg="#1a1a2e", pady=10)
        frame.pack(fill=tk.X, padx=15, pady=5)
        tk.Label(frame, text=title, font=("Segoe UI", 10), bg="#1a1a2e", fg="#a4b0be").pack()
        lbl = tk.Label(frame, text="0", font=("Segoe UI", 22, "bold"), bg="#1a1a2e", fg=color)
        lbl.pack()
        return lbl

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_fps.config(text="0")
        self.lbl_obj_count.config(text="0")

    def update_frame(self):
        if self.running and self.cap.isOpened():
            start_time = datetime.now()
            ret, frame = self.cap.read()
            
            if ret:
                # 1. Run YOLO Detection
                results = self.model(frame, verbose=False, conf=0.45) # conf=0.45 filters weak detections
                
                # 2. Get Plot (YOLO handles drawing boxes/labels automatically)
                annotated_frame = results[0].plot()
                
                # 3. Process Detections for GUI Stats
                detected_classes = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    detected_classes.append(class_name)
                    
                    # Log to listbox if seen for the first time in a while (simple logic)
                    log_entry = f"{class_name}"
                    # Simple duplication check for the UI list
                    if not self.log_list.get(0) or log_entry not in self.log_list.get(0):
                         current_time = datetime.now().strftime("%H:%M:%S")
                         self.log_list.insert(0, f"[{current_time}] Found: {class_name}")

                # 4. Update Stats on Right Panel
                self.lbl_obj_count.config(text=str(len(detected_classes)))
                
                # Calculate FPS
                process_time = (datetime.now() - start_time).total_seconds()
                fps = 1 / process_time if process_time > 0 else 0
                self.lbl_fps.config(text=f"{int(fps)}")

                # 5. Convert to Tkinter Image
                img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                # Resize to fit the label, keeping aspect ratio
                display_width = self.video_label.winfo_width()
                display_height = self.video_label.winfo_height()
                if display_width > 10 and display_height > 10: # ensure valid size
                    img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            # Recursive call
            self.root.after(10, self.update_frame)

    def clear_log(self):
        self.log_list.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_camera)
    root.mainloop()