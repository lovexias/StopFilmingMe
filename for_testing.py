import cv2
import time
import tkinter as tk
from tkinter import messagebox
from video_utils import WaveDetector, blur_faces_in_frame, get_video_rotation

video_path = "C:\\Users\\Layne\\Desktop\\RECORDINGS[CONFI]\\GH010052.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
rotation = get_video_rotation(video_path)
cap.release()

wave_detector = WaveDetector(video_path, fps)
wave_timestamps = wave_detector.detect_wave_timestamps(show_ui=True, frame_skip=3)
print("Wave timestamps (s):", wave_timestamps)

# UI confirmation flags
blur_enabled = False
blur_prompted = False

def ask_blur():
    global blur_enabled
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("Wave Detected", "Do you want to apply blur?")
    blur_enabled = result
    root.destroy()

cap = cv2.VideoCapture(video_path)
frame_count = 0
frame_time = 1.0 / fps
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    timestamp = frame_count * frame_time
    frame_count += 1

    # If wave detected and not asked yet, prompt
    if not blur_prompted and any(abs(ts - timestamp) < 0.1 for ts in wave_timestamps):
        ask_blur()
        blur_prompted = True  # Prevent re-asking

    if blur_enabled:
        frame = blur_faces_in_frame(frame)

    cv2.imshow("Test Video - Wave Detection + Blur", frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
