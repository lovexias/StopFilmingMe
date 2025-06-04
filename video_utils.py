# video_utils.py

import cv2
import subprocess
import json
import numpy as np
import mediapipe as mp

def get_video_rotation(path):
    """
    Uses ffprobe to read the 'rotate' metadata tag from the video file.
    Returns an integer rotation angle (0, 90, 180, 270).
    If ffprobe is not present or fails, returns 0.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "json", path
        ]
        output = subprocess.check_output(cmd).decode("utf-8")
        ffprobe_data = json.loads(output)
        tags = ffprobe_data.get("streams", [{}])[0].get("tags", {})
        return int(tags.get("rotate", 0))
    except:
        return 0


def generate_thumbnails(video_path, total_frames, rotation_angle, num_thumbs=10, thumb_size=(80, 45)):
    """
    Generate up to num_thumbs equally‐spaced thumbnails from the video.
    Returns a list of tuples: [(frame_index, numpy_rgb_thumbnail), ...].
    Each thumbnail is resized to thumb_size (width, height) in RGB.
    """
    thumbs = []
    if total_frames <= 0:
        return thumbs

    step = max(1, total_frames // num_thumbs)
    cap = cv2.VideoCapture(video_path)

    for i in range(num_thumbs):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Apply rotation if needed
        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert to RGB and resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(frame_rgb, thumb_size)
        thumbs.append((frame_idx, small.copy()))

    cap.release()
    return thumbs


class WaveDetector:
    """
    Runs MediaPipe Hands on the entire video to detect simple wave gestures.
    A wave is detected when the average X‐coordinate of hand landmarks oscillates
    left/right enough times within a short time window (1 second).
    """
    def __init__(self, video_path, fps, detection_confidence=0.8):
        self.video_path = video_path
        self.fps = fps or 30.0
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=detection_confidence
        )

    def detect_wave_timestamps(self):
        """
        Process the entire video frame by frame, detect hand landmarks, and
        return a list of timestamps (in seconds) when a “wave” was detected.
        """
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        detected_timestamps = []
        last_x = None
        movement_history = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            timestamp = frame_count / self.fps
            frame_count += 1

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    media_x = sum(xs) / len(xs)
                    direction = None
                    if last_x is not None:
                        if media_x < last_x - 0.02:
                            direction = "left"
                        elif media_x > last_x + 0.02:
                            direction = "right"
                        if direction and (not movement_history or movement_history[-1][0] != direction):
                            movement_history.append((direction, timestamp))
                    # Keep only movements in the last 1 second
                    movement_history = [(d, t) for d, t in movement_history if timestamp - t <= 1.0]
                    if len(movement_history) >= 4:
                        detected_timestamps.append(timestamp)
                        movement_history.clear()
                    last_x = media_x

        cap.release()
        self.hands.close()
        return detected_timestamps


def blur_faces_in_frame(frame):
    """
    Uses MediaPipe FaceDetection to find faces in this BGR frame,
    blurs each detected face region, and returns the new BGR frame.
    """
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(frame_rgb)
    mp_face.close()

    if not results.detections:
        return frame

    h, w, _ = frame.shape
    mask = np.zeros_like(frame)
    for detection in results.detections:
        box = detection.location_data.relative_bounding_box
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        w_box = int(box.width * w)
        h_box = int(box.height * h)
        x, y = max(0, x), max(0, y)
        w_box = min(w_box, w - x)
        h_box = min(h_box, h - y)
        mask[y : y + h_box, x : x + w_box] = 255

    blurred = cv2.GaussianBlur(frame, (55, 55), 0)
    return np.where(mask == 255, blurred, frame)
