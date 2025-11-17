#run using pythong for_testing.py 2>nul to prevent unnecessary logs

import cv2
import time
import tkinter as tk
from tkinter import messagebox
from utilities import WaveDetector, blur_faces_of_person, get_video_rotation




video_path = "C:\\Users\\ramos\\Pictures\\Camera Roll\\test-people.mp4"


cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
rotation = get_video_rotation(video_path)
cap.release()

#IMPORTANT: I JUST CHANGE THE CLASS CALLED HERE FROMM HandOverFaceDetector to WaveDetector TO TEST EACH CLASS SEPARATELY
#also wave_detector to detect_wave_timestamps

wave_detector = WaveDetector(video_path, fps)
wave_timestamps = wave_detector.detect_wave_timestamps(show_ui=True, frame_skip=1)
print("Hand timestamps (s):", wave_timestamps)

# UI confirmation flags
blur_enabled = False
blur_prompted = False
target_pose = None

def ask_blur():
    global blur_enabled
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("Hand Detected", "Do you want to apply blur?")
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
    if not blur_prompted:
        match = next(((f, p) for f, p in wave_timestamps if abs(f - frame_count) < 3), None)
        if match:
            ask_blur()
            if blur_enabled:
                target_pose = match[1]  # store pose for reuse
            blur_prompted = True

    # Blur every frame if confirmed
    if blur_enabled and target_pose:
        # Force match always by setting tolerance high (to disable skipping)
        frame = blur_faces_of_person(frame, target_pose, tolerance=1.0)
        if frame is not None and target_pose:
            maybe_blurred = blur_faces_of_person(frame, target_pose)
            if maybe_blurred is not None:
                frame = maybe_blurred
        else:
            print(f"Skipping frame {frame_count}: frame is None or target_pose missing.")







    cv2.imshow("Test Video - Wave Detection + Blur", frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
