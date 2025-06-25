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
    def __init__(self, video_path, fps, detection_confidence=0.8):
        self.video_path = video_path
        self.fps = fps or 30.0
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=detection_confidence
        )
        self.drawer = mp.solutions.drawing_utils

    def detect_wave_timestamps(self, show_ui=True, frame_skip=3):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        detected_timestamps = []
        last_x = None
        movement_history = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to reduce processing
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Resize to speed up processing
            frame = cv2.resize(frame, (640, 360))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            timestamp = frame_count / self.fps
            frame_count += 1

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand skeleton
                    if show_ui:
                        self.drawer.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    # Wave detection logic
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    media_x = sum(xs) / len(xs)
                    direction = None
                    if last_x is not None:
                        # More sensitive threshold for higher FPS
                        threshold = 0.005
                        if media_x < last_x - threshold:
                            direction = "left"
                        elif media_x > last_x + threshold:
                            direction = "right"
                        if direction and (not movement_history or movement_history[-1][0] != direction):
                            movement_history.append((direction, timestamp))
                    movement_history = [(d, t) for d, t in movement_history if timestamp - t <= 1.0]
                    if len(movement_history) >= 4:
                        print(f"Wave detected at {timestamp:.2f}s")
                        detected_timestamps.append(timestamp)
                        movement_history.clear()
                    last_x = media_x

            if show_ui:
                cv2.imshow("Wave Detection", frame)
                if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                    break


        cap.release()
        self.hands.close()
        if show_ui:
            cv2.destroyAllWindows()
        return detected_timestamps

class HandOverFaceDetector:
    def __init__(self, video_path, fps, detection_confidence=0.5):
        self.video_path = video_path
        self.fps = fps or 30.0
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=detection_confidence)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=detection_confidence
        )
        self.drawer = mp.solutions.drawing_utils

    def detect_hand_over_face_frames(self, show_ui=True, frame_skip=3):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        hand_over_face_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            frame = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_result = self.pose.process(frame_rgb)
            hands_result = self.hands.process(frame_rgb)

            if pose_result.pose_landmarks:
                nose_landmark = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                nose_x, nose_y = int(nose_landmark.x * frame.shape[1]), int(nose_landmark.y * frame.shape[0])

                if hands_result.multi_hand_landmarks:
                    for hand_landmarks in hands_result.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            hand_x = int(lm.x * frame.shape[1])
                            hand_y = int(lm.y * frame.shape[0])
                            dist = np.hypot(nose_x - hand_x, nose_y - hand_y)

                            if dist < 40:  # pixel threshold; adjust as needed
                                hand_over_face_frames.append(frame_count / self.fps)
                                print(f"Hand over face at frame {frame_count}")
                                break  # One close landmark is enough
                        else:
                            continue
                        break

            if show_ui:
                if pose_result.pose_landmarks:
                    self.drawer.draw_landmarks(frame, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                if hands_result.multi_hand_landmarks:
                    for hand_landmarks in hands_result.multi_hand_landmarks:
                        self.drawer.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                cv2.imshow("Hand Over Face Detection", frame)
                if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        self.pose.close()
        self.hands.close()
        if show_ui:
            cv2.destroyAllWindows()

        return hand_over_face_frames

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
