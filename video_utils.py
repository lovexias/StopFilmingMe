# video_utils.py

import cv2
import subprocess
import json
import numpy as np
import mediapipe as mp



# Reads the rotation angle metadata from a video file using ffprobe.
# Returns 0 if no rotation metadata is found.
def get_video_rotation(path):
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

# Generates equally spaced thumbnails from the video for preview UI.
def generate_thumbnails(video_path, total_frames, rotation_angle, num_thumbs=10, thumb_size=(80, 45)):
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

        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(frame_rgb, thumb_size)
        thumbs.append((frame_idx, small.copy()))

    cap.release()
    return thumbs

# Detects wave gestures based only on horizontal movement of hand landmarks (no pose).
class WaveDetector:
    def __init__(self, video_path, fps, detection_confidence=0.8):
        self.video_path = video_path
        self.fps = fps or 30.0
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=detection_confidence
        )
        self.pose = mp.solutions.pose.Pose()
        self.drawer = mp.solutions.drawing_utils

    def detect_wave_timestamps(self, show_ui=True, frame_skip=3):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        detected = []
        last_x = None
        movement_history = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            frame = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)

            if hand_results.multi_hand_landmarks and pose_results.pose_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if show_ui:
                        self.drawer.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    xs = [lm.x for lm in hand_landmarks.landmark]
                    media_x = sum(xs) / len(xs)
                    direction = None
                    if last_x is not None:
                        threshold = 0.005
                        if media_x < last_x - threshold:
                            direction = "left"
                        elif media_x > last_x + threshold:
                            direction = "right"
                        if direction and (not movement_history or movement_history[-1][0] != direction):
                            movement_history.append((direction, frame_count, pose_results.pose_landmarks.landmark))

                    movement_history = [(d, f, l) for d, f, l in movement_history if frame_count - f <= self.fps]
                    if len(movement_history) >= 4:
                        print(f"Wave detected at frame {frame_count}")
                        detected.append((frame_count, pose_results.pose_landmarks.landmark))
                        movement_history.clear()
                    last_x = media_x

                if show_ui:
                    self.drawer.draw_landmarks(frame, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            if show_ui:
                cv2.imshow("Wave Detection", frame)
                if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        self.hands.close()
        self.pose.close()
        if show_ui:
            cv2.destroyAllWindows()

        return detected

# Detects if a hand is near the face by checking hand and nose landmark proximity.
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
                nose = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                nose_x, nose_y = int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])

                if hands_result.multi_hand_landmarks:
                    for hand_landmarks in hands_result.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            hand_x = int(lm.x * frame.shape[1])
                            hand_y = int(lm.y * frame.shape[0])
                            dist = np.hypot(nose_x - hand_x, nose_y - hand_y)

                            if dist < 40:
                                hand_over_face_frames.append(frame_count / self.fps)
                                print(f"Hand over face at frame {frame_count}")
                                break
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

# Blurs only the face of a person whose pose matches the target skeleton landmarks.
# Used after detecting gesture to blur only that specific individual.
last_face_box = None  # Global cache for the last known face box

def blur_faces_of_person(frame, target_landmarks, tolerance=0.05):

    global last_face_box

    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
    mp_pose = mp.solutions.pose.Pose()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = mp_pose.process(frame_rgb)
    face_result = mp_face.process(frame_rgb)
    h, w, _ = frame.shape

    if not pose_result.pose_landmarks:
        mp_face.close()
        mp_pose.close()
        return frame

    curr_landmarks = pose_result.pose_landmarks.landmark
    total_diff = sum(np.hypot(curr_landmarks[i].x - target_landmarks[i].x,
                              curr_landmarks[i].y - target_landmarks[i].y)
                     for i in range(len(curr_landmarks)))
    avg_diff = total_diff / len(curr_landmarks)

    if avg_diff > tolerance:
        mp_face.close()
        mp_pose.close()
        return frame

    # If no face detection now, try to initialize from earlier frame
    if not face_result.detections and last_face_box is None:
        for _ in range(5):
            mp_face_retry = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
            frame_rgb_retry = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            retry_result = mp_face_retry.process(frame_rgb_retry)
            mp_face_retry.close()

            if retry_result.detections:
                detection = retry_result.detections[0]
                box = detection.location_data.relative_bounding_box
                x = int(box.xmin * w)
                y = int(box.ymin * h)
                w_box = int(box.width * w)
                h_box = int(box.height * h)
                last_face_box = (x, y, w_box, h_box)
                break

    mask = np.zeros_like(frame)

    if face_result.detections:
        detection = face_result.detections[0]
        box = detection.location_data.relative_bounding_box
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        w_box = int(box.width * w)
        h_box = int(box.height * h)
        x, y = max(0, x), max(0, y)
        w_box = min(w_box, w - x)
        h_box = min(h_box, h - y)
        last_face_box = (x, y, w_box, h_box)
        mask[y:y + h_box, x:x + w_box] = 255
    elif last_face_box:
        x, y, w_box, h_box = last_face_box
        mask[y:y + h_box, x:x + w_box] = 255

    blurred = cv2.GaussianBlur(frame, (55, 55), 0)

    mp_face.close()
    mp_pose.close()
    return np.where(mask == 255, blurred, frame)


# NEW 7/2
def blur_faces_in_frame(frame):
    """
    Detect **all** faces in the frame and Gaussian-blur them.
    """
    
    # Initialize FaceDetection once per call
    face_detector = mp.solutions.face_detection.FaceDetection(
        min_detection_confidence=0.7
    )

    # Run detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(frame_rgb)
    h, w, _ = frame.shape

    # Build a mask of all face-regions
    mask = np.zeros_like(frame)
    if results.detections:
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)
            x, y = max(0, x), max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            mask[y:y+bh, x:x+bw] = 255

    # Blur everywhere the mask is set
    blurred = cv2.GaussianBlur(frame, (55, 55), 0)
    face_detector.close()
    return np.where(mask == 255, blurred, frame)
