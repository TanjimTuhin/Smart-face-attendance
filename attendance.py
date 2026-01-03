import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from PIL import Image, ImageTk
import threading

class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Face Attendance System")
        self.root.geometry("1200x750")
        self.root.configure(bg="#1a1a2e")
        
        # Variables
        self.cap = None
        self.running = False
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_path = os.path.join(self.script_dir, "ImagesAttendance")
        self.attendance_file = os.path.join(self.script_dir, "Attendance.csv")
        
        # Data
        self.images = []
        self.classNames = []
        self.encodeListKnown = []
        self.attendance_today = set()
        self.face_count = 0
        
        # Setup
        self.setup_directories()
        self.create_widgets()
        self.load_known_faces()
        self.update_stats()
        
    def setup_directories(self):
        """Create necessary directories"""
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
            
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w') as f:
                f.write("Name,Time,Date\n")
    
    def create_widgets(self):
        """Create all GUI elements"""
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#16213e", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame, 
            text="🎯 Smart Face Attendance System",
            font=("Segoe UI", 28, "bold"),
            bg="#16213e",
            fg="#00fff5"
        ).pack(pady=15)
        
        # Main Container
        main_container = tk.Frame(self.root, bg="#1a1a2e")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Panel - Video Feed
        left_panel = tk.Frame(main_container, bg="#0f3460", relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(
            left_panel,
            text="📹 Live Camera Feed",
            font=("Segoe UI", 16, "bold"),
            bg="#0f3460",
            fg="white"
        ).pack(pady=10)
        
        self.video_label = tk.Label(left_panel, bg="black", relief=tk.SUNKEN, bd=3)
        self.video_label.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(
            left_panel,
            text="● Camera Stopped",
            font=("Segoe UI", 12),
            bg="#0f3460",
            fg="#ff4757"
        )
        self.status_label.pack(pady=10)
        
        # Control Buttons
        control_frame = tk.Frame(left_panel, bg="#0f3460")
        control_frame.pack(pady=15)
        
        self.start_btn = tk.Button(
            control_frame, text="▶ Start Camera", font=("Segoe UI", 13, "bold"),
            bg="#2ed573", fg="white", activebackground="#26de81", relief=tk.FLAT,
            cursor="hand2", command=self.start_camera, width=15, height=2
        )
        self.start_btn.grid(row=0, column=0, padx=8)
        
        self.stop_btn = tk.Button(
            control_frame, text="⏸ Stop Camera", font=("Segoe UI", 13, "bold"),
            bg="#ff4757", fg="white", activebackground="#ee5a6f", relief=tk.FLAT,
            cursor="hand2", command=self.stop_camera, width=15, height=2, state=tk.DISABLED
        )
        self.stop_btn.grid(row=0, column=1, padx=8)
        
        # Right Panel - Info & Stats
        right_panel = tk.Frame(main_container, bg="#0f3460", width=350, relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        tk.Label(
            right_panel, text="📊 Statistics", font=("Segoe UI", 16, "bold"),
            bg="#0f3460", fg="white"
        ).pack(pady=15)
        
        # Stats Container
        stats_container = tk.Frame(right_panel, bg="#0f3460")
        stats_container.pack(fill=tk.X, padx=15)
        
        # Helper to create cards
        def create_card(parent, title, color):
            card = tk.Frame(parent, bg="#16213e", relief=tk.RAISED, bd=2)
            card.pack(fill=tk.X, pady=5)
            tk.Label(card, text=title, font=("Segoe UI", 11), bg="#16213e", fg="#a4b0be").pack(pady=(10, 5))
            lbl = tk.Label(card, text="0", font=("Segoe UI", 32, "bold"), bg="#16213e", fg=color)
            lbl.pack(pady=(0, 10))
            return lbl

        self.registered_count = create_card(stats_container, "Registered Faces", "#00fff5")
        self.attendance_count = create_card(stats_container, "Today's Attendance", "#2ed573")
        self.detected_count = create_card(stats_container, "Faces Detected", "#ffa502")
        
        # Recent Attendance List
        tk.Label(
            right_panel, text="📝 Recent Attendance", font=("Segoe UI", 14, "bold"),
            bg="#0f3460", fg="white"
        ).pack(pady=(20, 10))
        
        list_frame = tk.Frame(right_panel, bg="#16213e", relief=tk.SUNKEN, bd=2)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.attendance_listbox = tk.Listbox(
            list_frame, font=("Segoe UI", 10), bg="#16213e", fg="white",
            selectbackground="#00fff5", selectforeground="black",
            yscrollcommand=scrollbar.set, relief=tk.FLAT, highlightthickness=0
        )
        self.attendance_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.attendance_listbox.yview)
        
        # Action Buttons
        action_frame = tk.Frame(right_panel, bg="#0f3460")
        action_frame.pack(pady=10)
        
        buttons = [
            ("➕ Add Face", "#5f27cd", self.add_new_face, 0, 0),
            ("🔄 Refresh", "#ff6348", self.refresh_data, 0, 1),
            ("📄 View Records", "#1e90ff", self.view_attendance, 1, 0),
            ("🚪 Exit", "#535c68", self.exit_app, 1, 1)
        ]
        
        for text, color, cmd, r, c in buttons:
            tk.Button(
                action_frame, text=text, font=("Segoe UI", 10, "bold"),
                bg=color, fg="white", relief=tk.FLAT, cursor="hand2",
                command=cmd, width=12
            ).grid(row=r, column=c, padx=5, pady=5)
    
    def load_known_faces(self):
        """Load and encode known faces from directory"""
        self.images = []
        self.classNames = []
        
        if not os.path.exists(self.images_path):
            return
            
        files = [f for f in os.listdir(self.images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for file in files:
            img_path = os.path.join(self.images_path, file)
            curImg = cv2.imread(img_path)
            if curImg is not None:
                self.images.append(curImg)
                self.classNames.append(os.path.splitext(file)[0])
        
        if self.images:
            threading.Thread(target=self.encode_faces, daemon=True).start()
    
    def encode_faces(self):
        """Encode all loaded faces"""
        self.encodeListKnown = []
        for i, img in enumerate(self.images):
            try:
                # FIX 1: Use OpenCV conversion here too
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(img_rgb)
                
                if len(encodings) > 0:
                    self.encodeListKnown.append(encodings[0])
                else:
                    print(f"No face found in: {self.classNames[i]}")
            except Exception as e:
                print(f"Error encoding {self.classNames[i]}: {e}")
        
        self.update_stats()
    
    def mark_attendance(self, name):
        """Mark attendance for recognized person"""
        # Read existing file first to avoid duplicates in CSV
        nameList = []
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, "r") as f:
                myDataList = f.readlines()
                nameList = [line.split(",")[0] for line in myDataList]
            
        if name not in self.attendance_today:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dateString = now.strftime('%d-%m-%Y')
            
            # Write to file if not already in file for this session/day logic
            # (Simplistic check, ideal is checking date too)
            with open(self.attendance_file, "a") as f:
                 # Check if we should actually write (simple duplicate check)
                if name not in nameList: 
                    f.write(f'\n{name},{dtString},{dateString}')
            
            self.attendance_today.add(name)
            self.attendance_listbox.insert(0, f"✓ {name} - {dtString}")
            self.update_stats()
    
    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not access camera!")
                return
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="● Camera Running", fg="#2ed573")
            self.update_frame()
    
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="● Camera Stopped", fg="#ff4757")
    
    def update_frame(self):
        """Update video frame with FIX applied"""
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    # 1. Resize for speed
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    
                    # 2. CRITICAL FIX: Use OpenCV to convert BGR -> RGB
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # 3. Process Faces
                    facesCurFrame = face_recognition.face_locations(rgb_small_frame)
                    encodesCurFrame = face_recognition.face_encodings(rgb_small_frame, facesCurFrame)
                    
                    self.face_count = len(facesCurFrame)
                    self.detected_count.config(text=str(self.face_count))
                    
                    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                        matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                        
                        matchIndex = None
                        if len(faceDis) > 0:
                            matchIndex = np.argmin(faceDis)
                        
                        # Scale coordinates back up
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        
                        if matchIndex is not None and matches[matchIndex]:
                            name = self.classNames[matchIndex].upper()
                            self.mark_attendance(name)
                            
                            # Draw Green Box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        else:
                            # Draw Red Box (Unknown)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                            cv2.putText(frame, "UNKNOWN", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                            
                except Exception as e:
                    print(f"Frame Error: {e}")
                
                # 4. Display in GUI
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                img = ImageTk.PhotoImage(img)
                
                self.video_label.imgtk = img
                self.video_label.configure(image=img)
            
            self.video_label.after(10, self.update_frame)
    
    def update_stats(self):
        self.registered_count.config(text=str(len(self.encodeListKnown)))
        self.attendance_count.config(text=str(len(self.attendance_today)))
    
    def add_new_face(self):
        messagebox.showinfo("Add New Face", f"1. Save photo as 'Name.jpg'\n2. Put in folder: {self.images_path}\n3. Click Refresh")
    
    def refresh_data(self):
        self.load_known_faces()
        self.attendance_today.clear()
        self.attendance_listbox.delete(0, tk.END)
        messagebox.showinfo("Success", "Data refreshed!")
    
    def view_attendance(self):
        if os.path.exists(self.attendance_file):
            try:
                os.startfile(self.attendance_file)
            except AttributeError:
                messagebox.showerror("Error", "Feature available on Windows only")
        else:
            messagebox.showwarning("Warning", "No records found!")
    
    def exit_app(self):
        if messagebox.askyesno("Exit", "Are you sure?"):
            self.stop_camera()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_app)
    root.mainloop()