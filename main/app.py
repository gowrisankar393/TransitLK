import cv2
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import queue
from ultralytics import YOLO


class CrashDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Crash Detection System - YOLOv11")
        self.root.geometry("1000x700")

        # Model path
        self.model_path = "best.pt"
        self.model = None
        self.load_model()

        # Video variables
        self.cap = None
        self.is_playing = False
        self.video_path = None

        # Inference variables
        self.inference_interval = 0.5
        self.last_inference_time = 0
        self.prediction_queue = queue.Queue()

        self.create_ui()

        self.inference_thread = threading.Thread(
            target=self.inference_worker,
            daemon=True
        )
        self.inference_thread.start()

        # UI updater
        self.update_ui()
        self.model_lock = threading.Lock()


    def load_model(self):
        try:
            self.model = YOLO(self.model_path)
            self.model.fuse()
            print("YOLOv11 model loaded successfully!!!!!!!")
        except Exception as e:
            print("Model load error:", e)


    def create_ui(self):
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(fill=tk.X, padx=10)

        tk.Button(control_frame, text="📁 Upload Video",
                  command=self.upload_video, bg='#4CAF50', fg='white').pack(side=tk.LEFT, padx=5)

        self.play_btn = tk.Button(control_frame, text="▶️ Play",
                                  command=self.toggle_play, bg='#2196F3', fg='white', state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="⏹️ Stop",
                  command=self.stop_video, bg='#f44336', fg='white').pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(control_frame, text="No video loaded", fg='gray')
        self.status_label.pack(side=tk.RIGHT, padx=10)

        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_frame = tk.LabelFrame(content_frame, text="Video Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(content_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        self.prediction_label = tk.Label(right_frame, text="WAITING",
                                         font=('Arial', 24, 'bold'), fg='orange')
        self.prediction_label.pack(pady=20)

        self.confidence_label = tk.Label(right_frame, text="Confidence: --",
                                         font=('Arial', 12))
        self.confidence_label.pack()

        log_frame = tk.LabelFrame(right_frame, text="Detection Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, font=('Courier', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        

    # Video control
    def upload_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.status_label.config(text=self.video_path.split("/")[-1])
            self.play_btn.config(state=tk.NORMAL)

    def toggle_play(self):
        if not self.is_playing:
            self.start_video()
        else:
            self.pause_video()

    def start_video(self):
        if not self.video_path:
            print("No video selected")
            return

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print("ERROR: Cannot open video")
            return

        self.is_playing = True
        self.play_btn.config(text="⏸️ Pause")

        print("Video started")

        self.root.after(10, self.video_loop)

    def pause_video(self):
        self.is_playing = False
        self.play_btn.config(text="▶️ Play")

    def stop_video(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

    def video_loop(self):
        if not self.is_playing:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        current_time = time.time()
        if current_time - self.last_inference_time >= self.inference_interval:
            if self.prediction_queue.empty():
                self.prediction_queue.put(frame.copy())
            self.last_inference_time = current_time

        frame = self.draw_boxes(frame)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(10, self.video_loop)

    def inference_worker(self):
        while True:
            try:
                frame = self.prediction_queue.get(timeout=1)
                label, conf = self.run_inference(frame)
                self.root.after(0, self.update_prediction, label, conf)
            except queue.Empty:
                continue

    def run_inference(self, frame):
        try:
            with self.model_lock:
                results = self.model.predict(frame, verbose=False)[0]

            accident_detected = False
            max_conf = 0

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls]

                if "accident" in name:
                    accident_detected = True
                    max_conf = max(max_conf, conf)

            if accident_detected:
                return "CRASH", max_conf
            else:
                return "NORMAL", 0.90

        except Exception as e:
            print("Inference error:", e)
            return "ERROR", 0.0

    def draw_boxes(self, frame):
        results = self.model(frame, verbose=False)[0]
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]

            color = (0,0,255) if "accident" in label else (0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        return frame

    def update_prediction(self, label, conf):
        color = 'red' if label == "CRASH" else 'green'
        self.prediction_label.config(text=label, fg=color)
        self.confidence_label.config(text=f"Confidence: {conf:.2%}")

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {label} ({conf:.2%})\n")
        self.log_text.see(tk.END)

    def update_ui(self):
        self.root.after(100, self.update_ui)


if __name__ == "__main__":
    root = tk.Tk()
    app = CrashDetectionGUI(root)
    root.mainloop()
